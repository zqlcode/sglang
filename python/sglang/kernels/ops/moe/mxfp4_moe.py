from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl

MXFP4_BLOCK_SIZE = 32


@triton.jit
def fused_moe_kernel_mxfp4(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    filter_expert: tl.constexpr,
):
    """Routed BF16 x packed MXFP4 GEMM.

    Triton expands packed E2M1 values and row-major UE8M0 scales tile-locally.
    The full expert weights are never dequantized or materialized.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if filter_expert and off_experts == -1:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, 0.0, mask=c_mask)
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    EVEN_N: tl.constexpr = N % BLOCK_SIZE_N == 0
    if not EVEN_N:
        offs_bn %= N

    MXFP4_PACK_FACTOR_CONST: tl.constexpr = 2
    MXFP4_BLOCK_SIZE_CONST: tl.constexpr = 32
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K // MXFP4_PACK_FACTOR_CONST)
    offs_sf_k = tl.arange(0, BLOCK_SIZE_K // MXFP4_BLOCK_SIZE_CONST)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am
            + (k_start + offs_k[None, :]) * stride_ak
        )
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)

        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (k_start // MXFP4_PACK_FACTOR_CONST + offs_k_packed[:, None]) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b = tl.load(b_ptrs).to(tl.uint8)

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[:, None] * stride_bsn
            + (k_start // MXFP4_BLOCK_SIZE_CONST + offs_sf_k[None, :]) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs).to(tl.uint8)

        accumulator = tl.dot_scaled(
            a,
            None,
            "bf16",
            b,
            b_scale,
            "e2m1",
            acc=accumulator,
            rhs_k_pack=True,
        )

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    if EVEN_N:
        c_mask = token_mask[:, None]
    else:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask)


def _validate_mxfp4_kernel_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
    config: Dict[str, Any],
    logical_k: int,
    *,
    a_use_tma: bool,
    b_use_tma: bool,
    c_sorted: bool,
    fuse_sum_all_reduce: bool,
) -> None:
    if A.dtype != torch.bfloat16:
        raise TypeError("Triton MXFP4 only supports BF16 activations")
    if B.dtype != torch.uint8:
        raise TypeError("MXFP4 weights must be packed uint8")
    if B_scale is None or B_scale.dtype != torch.uint8:
        raise TypeError("MXFP4 requires raw UE8M0 scales viewed as uint8")
    if block_shape != [0, MXFP4_BLOCK_SIZE]:
        raise ValueError(
            f"MXFP4 requires block_shape=[0, {MXFP4_BLOCK_SIZE}], " f"got {block_shape}"
        )
    if bias is not None:
        raise NotImplementedError("Triton MXFP4 does not support bias")
    if B_zp is not None:
        raise ValueError("MXFP4 does not use zero points")
    if fuse_sum_all_reduce or a_use_tma or b_use_tma or c_sorted:
        raise NotImplementedError(
            "Triton MXFP4 does not support TMA, sorted output, or fused all-reduce"
        )

    assert logical_k == A.shape[1] and logical_k % MXFP4_BLOCK_SIZE == 0
    assert config["BLOCK_SIZE_K"] % MXFP4_BLOCK_SIZE == 0
    assert logical_k % config["BLOCK_SIZE_K"] == 0


def invoke_mxfp4_moe_kernel(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    block_shape: Optional[List[int]],
    logical_k: int,
    a_use_tma: bool,
    b_use_tma: bool,
    c_sorted: bool,
    filter_expert: bool,
    fuse_sum_all_reduce: bool,
) -> None:
    _validate_mxfp4_kernel_inputs(
        A,
        B,
        bias,
        B_scale,
        B_zp,
        block_shape,
        config,
        logical_k,
        a_use_tma=a_use_tma,
        b_use_tma=b_use_tma,
        c_sorted=c_sorted,
        fuse_sum_all_reduce=fuse_sum_all_reduce,
    )
    assert B_scale is not None

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    fused_moe_kernel_mxfp4[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        logical_k,
        sorted_token_ids.shape[0],
        num_valid_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        filter_expert=filter_expert,
        **config,
    )
