import unittest

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsQuantInfo
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK, TopKOutputFormat
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    "Triton MXFP4 dot_scaled requires SM90 or newer",
)
def test_triton_mxfp4_fused_moe_matches_torch_reference():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    torch.manual_seed(1)

    num_tokens = 17
    hidden_size = 64
    intermediate_size = 64
    num_experts = 2
    top_k = 2
    scale_value = 0.0625
    e2m1_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device="cuda",
    )

    def make_mxfp4_weight(num_rows: int, logical_k: int):
        codes = torch.randint(
            0,
            16,
            (num_experts, num_rows, logical_k),
            device="cuda",
            dtype=torch.uint8,
        )
        packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
        scales = torch.full(
            (num_experts, num_rows, logical_k // 32),
            scale_value,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        dequantized = (e2m1_values[codes.long()] * scale_value).to(torch.bfloat16)
        return (
            packed.contiguous(),
            scales.view(torch.uint8).contiguous(),
            dequantized,
        )

    hidden_states = torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    hidden_states_ref = hidden_states.clone()
    w1, w1_scale, w1_ref = make_mxfp4_weight(2 * intermediate_size, hidden_size)
    w2, w2_scale, w2_ref = make_mxfp4_weight(hidden_size, intermediate_size)

    router_logits = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )
    topk_weights, topk_ids = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    topk_output = StandardTopKOutput(
        topk_weights,
        topk_ids.to(torch.int32),
        router_logits,
    )
    runner_config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
        inplace=True,
    )

    output = fused_moe(
        hidden_states,
        w1,
        w2,
        topk_output,
        moe_runner_config=runner_config,
        use_mxfp4_w4a16=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=[0, 32],
    )

    reference = []
    for token in range(num_tokens):
        token_output = torch.zeros(hidden_size, device="cuda", dtype=torch.float32)
        for route in range(top_k):
            expert = int(topk_ids[token, route])
            gate_up = (hidden_states_ref[token].float() @ w1_ref[expert].float().T).to(
                torch.bfloat16
            )
            activated = (
                F.silu(gate_up[:intermediate_size]) * gate_up[intermediate_size:]
            ).to(torch.bfloat16)
            down = (activated.float() @ w2_ref[expert].float().T).to(torch.bfloat16)
            token_output += topk_weights[token, route] * down.float()
        reference.append(token_output)
    reference = torch.stack(reference).to(torch.bfloat16)

    torch.testing.assert_close(output, reference, rtol=5e-2, atol=0.08)


class TestFusedMOE(CustomTestCase):
    NUM_EXPERTS = [8, 64]
    TOP_KS = [2, 4]

    @staticmethod
    def create_random_cuda_tensor(shape, dtype, mean=0, std=0.01):
        """Create a random CUDA tensor

        Args:
            shape: Tensor shape
            dtype: Data type
            mean: Mean value
            std: Standard deviation

        Returns:
            torch.Tensor: Randomly initialized CUDA tensor
        """
        return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

    def get_tolerance(self, dtype):
        """Get tolerance values for different data types

        Args:
            dtype: Data type

        Returns:
            tuple: (relative tolerance, absolute tolerance)
        """
        if dtype == torch.float32:
            return 1e-5, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-5, 1e-5
        else:
            return 1e-2, 1e-2  # Default values for other types

    def torch_naive_moe(
        self,
        a,
        w1,
        w2,
        score,
        topk,
        return_per_expert: bool = False,
    ):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)

        if w1.dtype == torch.float8_e4m3fn:
            w1_compute = w1.to(a.dtype)
            w2_compute = w2.to(a.dtype)
        else:
            w1_compute = w1
            w2_compute = w2

        for i in range(w1_compute.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(
                    a[mask] @ w1_compute[i].transpose(0, 1)
                ) @ w2_compute[i].transpose(0, 1)

        weighted = out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(
            out.dtype
        )

        if return_per_expert:
            return weighted

        return weighted.sum(dim=1)

    def _test_case(self, m, n, k, e, topk, dtype):
        rtol, atol = self.get_tolerance(dtype)

        a = self.create_random_cuda_tensor((m, k), dtype)
        w1 = self.create_random_cuda_tensor((e, 2 * n, k), dtype)
        w2 = self.create_random_cuda_tensor((e, k, n), dtype)
        w1_tri = w1.clone()
        w2_tri = w2.clone()
        w1_tri = w1_tri.transpose(-2, -1).contiguous()
        w2_tri = w2_tri.transpose(-2, -1).contiguous()
        score = self.create_random_cuda_tensor((m, e), dtype)

        topk_op = TopK(
            top_k=topk,
            renormalize=False,
            use_grouped_topk=False,
        )
        topk_op.topk_config.output_format = TopKOutputFormat.TRITON_KERNEL
        triton_topk_output = topk_op.forward_cuda(
            hidden_states=a,
            router_logits=score,
        )

        quant_info = TritonKernelsQuantInfo(w13_weight=w1_tri, w2_weight=w2_tri)

        dispatch_output = StandardDispatchOutput(
            hidden_states=a, hidden_states_scale=None, topk_output=triton_topk_output
        )

        torch_per_expert = self.torch_naive_moe(
            a, w1, w2, score, topk, return_per_expert=True
        )
        torch_combined = torch_per_expert.sum(dim=1)

        def run_runner(config):
            runner = MoeRunner(MoeRunnerBackend.TRITON_KERNELS, config)
            result = runner.run(dispatch_output, quant_info)
            return result.hidden_states

        # Combined output (no_combine=False)
        non_fused_config = MoeRunnerConfig(inplace=False)
        non_fused_output = run_runner(non_fused_config)
        torch.testing.assert_close(
            non_fused_output, torch_combined, rtol=rtol, atol=atol
        )

        # Per-expert output (no_combine=True)
        non_fused_no_combine_config = MoeRunnerConfig(
            inplace=False, no_combine=True, top_k=topk
        )
        non_fused_no_combine_output = run_runner(non_fused_no_combine_config)
        torch.testing.assert_close(
            non_fused_no_combine_output, torch_per_expert, rtol=rtol, atol=atol
        )

    def test_various_configurations(self):
        m_values = [1, 32, 64, 256]
        n_values = [128, 1024]
        k_values = [128, 512, 1024]
        dtypes = [torch.bfloat16]

        # Calculate total number of tests
        total_tests = (
            len(m_values)
            * len(n_values)
            * len(k_values)
            * len(self.NUM_EXPERTS)
            * len(self.TOP_KS)
            * len(dtypes)
        )

        # Create progress bar
        with tqdm(total=total_tests, desc="Running MoE tests") as pbar:
            for m in m_values:
                for n in n_values:
                    for k in k_values:
                        for e in self.NUM_EXPERTS:
                            for topk in self.TOP_KS:
                                for dtype in dtypes:
                                    with self.subTest(
                                        m=m,
                                        n=n,
                                        k=k,
                                        e=e,
                                        topk=topk,
                                        dtype=dtype,
                                    ):
                                        self._test_case(
                                            m,
                                            n,
                                            k,
                                            e,
                                            topk,
                                            dtype,
                                        )
                                        torch.cuda.empty_cache()
                                    pbar.update(1)


if __name__ == "__main__":
    unittest.main()
