import types
from unittest.mock import patch

import pytest
import torch

from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU


def _build_fake_fused_moe_layer(
    *,
    tp_size: int = 1,
    fully_sharded: bool = False,
    hidden_size: int = 8,
    intermediate_size_per_partition: int = 4,
    num_experts: int = 2,
    max_rank: int = 4,
):
    layer = object.__new__(FusedMoEWithLoRA)
    layer.base_layer = types.SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
    )
    layer.tp_size = tp_size
    layer.tp_rank = 0
    layer.fully_sharded = fully_sharded
    layer._w13_slices = 1
    layer.w13_lora_a_stacked = (
        torch.zeros((2, num_experts, max_rank, hidden_size)),
    )
    layer.w13_lora_b_stacked = (
        torch.zeros((2, num_experts, intermediate_size_per_partition, max_rank)),
    )
    layer.w2_lora_a_stacked = (
        torch.zeros((2, num_experts, max_rank, intermediate_size_per_partition)),
    )
    layer.w2_lora_b_stacked = (torch.zeros((2, num_experts, hidden_size, max_rank)),)
    layer.adapter_enabled = torch.zeros((3,), dtype=torch.int)
    return layer


def _make_valid_loras(
    *,
    num_experts: int,
    hidden_size: int,
    rank: int,
    tp_size: int,
    intermediate_size_per_partition: int,
):
    intermediate_full = intermediate_size_per_partition * tp_size
    w13_out_full = intermediate_size_per_partition * tp_size
    return (
        torch.zeros((num_experts, rank, hidden_size)),
        torch.zeros((num_experts, rank, intermediate_full)),
        torch.zeros((num_experts, rank, hidden_size)),
    ), (
        torch.zeros((num_experts, w13_out_full, rank)),
        torch.zeros((num_experts, hidden_size, rank)),
        torch.zeros((num_experts, w13_out_full, rank)),
    )


def test_fused_moe_set_lora_rejects_wrong_lora_structure():
    layer = _build_fake_fused_moe_layer()
    lora_a, lora_b = _make_valid_loras(
        num_experts=2,
        hidden_size=8,
        rank=2,
        tp_size=1,
        intermediate_size_per_partition=4,
    )

    with pytest.raises(ValueError, match="expects exactly 3 tensors"):
        layer.set_lora(index=0, lora_a=list(lora_a[:2]), lora_b=list(lora_b))


def test_fused_moe_set_lora_rejects_non_divisible_w2_input_for_tp():
    layer = _build_fake_fused_moe_layer(tp_size=2)
    lora_a, lora_b = _make_valid_loras(
        num_experts=2,
        hidden_size=8,
        rank=2,
        tp_size=2,
        intermediate_size_per_partition=4,
    )
    lora_a = list(lora_a)
    # Make w2 input dimension non-divisible by tp_size=2.
    lora_a[1] = torch.zeros((2, 2, 7))

    with pytest.raises(ValueError, match="w2 LoRA input dim must be divisible"):
        layer.set_lora(index=0, lora_a=lora_a, lora_b=list(lora_b))


class _FakeTokenMappingMeta:
    def __init__(self, token_lora_mapping: torch.Tensor, lora_ids: torch.Tensor):
        self._token_lora_mapping = token_lora_mapping
        self._lora_ids = lora_ids

    def meta_args(self, *_args, **_kwargs):
        return (
            self._token_lora_mapping,
            None,
            None,
            None,
            self._lora_ids,
            None,
            None,
        )


def _build_fake_punica_wrapper(max_loras: int = 4):
    wrapper = types.SimpleNamespace()
    wrapper.lora_config = types.SimpleNamespace(specialize_active_lora=False)
    wrapper._moe_align_buffers = {}
    token_lora_mapping = torch.zeros((8,), dtype=torch.int32)
    lora_ids = torch.arange(max_loras, dtype=torch.int32)
    wrapper.token_mapping_meta = _FakeTokenMappingMeta(token_lora_mapping, lora_ids)
    return wrapper


def test_moe_lora_align_block_size_reuses_buffers_for_same_shape():
    wrapper = _build_fake_punica_wrapper(max_loras=4)
    topk_ids = torch.zeros((2, 2), dtype=torch.int32)
    adapter_enabled = torch.ones((4,), dtype=torch.int32)

    with patch(
        "vllm.lora.punica_wrapper.punica_gpu.ops.moe_lora_align_block_size",
        return_value=None,
    ):
        _, sorted_ids_1, expert_ids_1, num_tokens_post_pad_1 = (
            PunicaWrapperGPU.moe_lora_align_block_size(
                wrapper,
                topk_ids=topk_ids,
                num_tokens=2,
                block_size=4,
                num_experts=8,
                max_loras=4,
                adapter_enabled=adapter_enabled,
            )
        )
        _, sorted_ids_2, expert_ids_2, num_tokens_post_pad_2 = (
            PunicaWrapperGPU.moe_lora_align_block_size(
                wrapper,
                topk_ids=topk_ids,
                num_tokens=2,
                block_size=4,
                num_experts=8,
                max_loras=4,
                adapter_enabled=adapter_enabled,
            )
        )

    assert sorted_ids_1 is sorted_ids_2
    assert expert_ids_1 is expert_ids_2
    assert num_tokens_post_pad_1 is num_tokens_post_pad_2
