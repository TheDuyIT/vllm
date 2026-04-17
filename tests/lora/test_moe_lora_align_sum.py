# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def sample_data(num_experts, max_loras, num_tokens, topk_num):
    topk_ids = torch.zeros((num_tokens, topk_num), dtype=torch.int32)
    token_lora_mapping = torch.zeros((num_tokens,), dtype=torch.int32)

    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)
        for j in range(topk_num):
            topk_ids[i, j] = pool[j]
        token_lora_mapping[i] = random.randint(0, max_loras - 1)

    return topk_ids.to(DEVICE_TYPE), token_lora_mapping.to(DEVICE_TYPE)


@pytest.mark.parametrize("num_tokens", [100, 200, 1024, 4096])  # 81920
@pytest.mark.parametrize("topk_num", [6])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 512])
@pytest.mark.parametrize("max_loras", [2, 32])
@pytest.mark.parametrize("block_size", [16])
def test_moe_lora_align_block_size(
    num_tokens, topk_num, num_experts, max_loras, block_size
):
    # sample data
    random.seed(1)
    topk_ids, token_lora_mapping = sample_data(
        num_experts, max_loras, num_tokens, topk_num
    )

    # compute paddings
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # init output tensors
    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    expert_ids = torch.full(
        (max_loras * max_num_m_blocks,),
        num_experts,
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    num_tokens_post_pad = torch.zeros(
        (max_loras,), dtype=torch.int32, device=DEVICE_TYPE
    )
    adapter_enabled = torch.ones(
        (max_loras + 1,), dtype=torch.int32, device=DEVICE_TYPE
    )
    lora_ids = torch.arange(max_loras + 2, dtype=torch.int32, device=DEVICE_TYPE)

    # call kernel
    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
    )

    # verify values
    expert_ids = expert_ids.view(max_loras, -1)
    sorted_token_ids = sorted_token_ids.view(max_loras, -1, block_size)

    for lora_idx in range(max_loras):
        for token_idx in range(sorted_token_ids.size(1)):
            block = sorted_token_ids[lora_idx][token_idx]
            indices = block[block != topk_ids.numel()]
            if indices.numel() > 0:
                expert_id = expert_ids[lora_idx][token_idx]
                assert torch.all(topk_ids.view(-1)[indices] == expert_id)


@pytest.mark.parametrize(
    "max_loras,num_lora_tokens,num_base_tokens",
    [
        # The minimal failing configuration from vllm-project/vllm#32235 class:
        # max_loras == number_of_real_active_loras AND the batch contains
        # base-model tokens (lora_id=-1). With an old grid of `max_loras`
        # (resp. `max_loras * 2`), iterating over active_lora_ids hits -1 at
        # position 0 and the real LoRA slot (pushed to position max_loras) is
        # never processed, leaving the output buffers uninitialized.
        (1, 8, 8),  # mixed base + 1 LoRA, max_loras=1 (was broken)
        (1, 8, 0),  # LoRA only,                     max_loras=1 (should work)
        (2, 8, 8),  # mixed base + 1 LoRA, max_loras=2 (should work)
    ],
)
@pytest.mark.parametrize("topk_num", [6])
@pytest.mark.parametrize("num_experts", [64, 128])
@pytest.mark.parametrize("block_size", [16])
def test_moe_lora_align_block_size_mixed_base_and_lora(
    max_loras, num_lora_tokens, num_base_tokens, topk_num, num_experts, block_size
):
    """Regression test for issue #32235 on the C++ align-kernel side.

    Constructs `active_lora_ids` exactly the way
    `LoRAKernelMeta.prepare_tensors` does -- via
    `torch.unique(token_lora_mapping, sorted=True)` -- so that -1
    (base-model tokens) occupies position 0 of `lora_ids` whenever the
    batch is mixed. Output buffers are initialized with sentinel
    values so we can distinguish "kernel wrote this" from "kernel
    skipped this slot". The test fails on the old grid sizing
    (`max_loras` / `max_loras * 2`) because the real LoRA slot gets
    pushed to position `max_loras` and is never processed; the fix
    bumps the grid to `max_loras + 1` so that slot is covered.
    """
    random.seed(1)

    # We only use LoRA slot 0 as "the one real active LoRA" so that
    # num_unique_real_loras == 1. When num_base_tokens > 0, active_lora_ids
    # becomes [-1, 0, ...] after sorted-unique, triggering the bug when
    # max_loras == 1.
    num_tokens = num_lora_tokens + num_base_tokens
    assert num_lora_tokens > 0, "test requires at least one LoRA-tagged token"

    topk_ids = torch.zeros((num_tokens, topk_num), dtype=torch.int32)
    token_lora_mapping = torch.empty((num_tokens,), dtype=torch.int32)
    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)
        for j in range(topk_num):
            topk_ids[i, j] = pool[j]
        token_lora_mapping[i] = 0 if i < num_lora_tokens else -1

    topk_ids = topk_ids.to(DEVICE_TYPE)
    token_lora_mapping = token_lora_mapping.to(DEVICE_TYPE)

    # Compute paddings (same convention as the production call path).
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # Mirror LoRAKernelMeta: lora_ids has length max_loras + 1, pre-filled
    # with -1, then first N entries overwritten with sorted-unique values
    # of token_lora_mapping.
    lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32, device=DEVICE_TYPE)
    unique_ids = torch.unique(token_lora_mapping, sorted=True)
    lora_ids[: unique_ids.numel()] = unique_ids.to(torch.int32)

    # Sanity: the layout we are specifically trying to test.
    if num_base_tokens > 0:
        assert lora_ids[0].item() == -1, (
            "prepare_tensors layout mismatch: -1 expected at position 0"
        )

    # All real LoRA slots are enabled.
    adapter_enabled = torch.ones(
        (max_loras + 1,), dtype=torch.int32, device=DEVICE_TYPE
    )

    # Initialize output buffers with distinctive SENTINELS rather than zeros
    # so that "kernel never wrote this position" is observable. These
    # sentinels must be out-of-domain for valid values the kernel would
    # write, so we can detect a skipped slot.
    SENTINEL_EXPERT = -2  # kernel writes either a real expert id [0, num_experts) or -1
    SENTINEL_TOKEN = -7  # kernel writes a token index or the `numel` padding value
    SENTINEL_NPAD = -13  # kernel writes 0 on init then the cumsum total

    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        SENTINEL_TOKEN,
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    expert_ids = torch.full(
        (max_loras * max_num_m_blocks,),
        SENTINEL_EXPERT,
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    num_tokens_post_pad = torch.full(
        (max_loras,), SENTINEL_NPAD, dtype=torch.int32, device=DEVICE_TYPE
    )

    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
    )

    # Subsequent `.item()` and elementwise comparisons below implicitly
    # synchronize, so an explicit sync here is not needed.

    # Slot 0 is the real LoRA in all parametrized cases (see construction).
    real_slot = 0

    # 1. num_tokens_post_pad[real_slot] must have been overwritten.
    #    (Without the fix, it stayed at SENTINEL_NPAD for max_loras=1+mixed.)
    post_pad_val = num_tokens_post_pad[real_slot].item()
    assert post_pad_val != SENTINEL_NPAD, (
        f"num_tokens_post_pad[{real_slot}] was never written by the kernel "
        f"(still {SENTINEL_NPAD}); the align kernel skipped the real LoRA slot."
    )
    # And must be a valid block-aligned count.
    assert 0 < post_pad_val <= max_num_tokens_padded, (
        f"num_tokens_post_pad[{real_slot}]={post_pad_val} out of range "
        f"(expected 0 < x <= {max_num_tokens_padded})."
    )
    assert post_pad_val % block_size == 0, (
        f"num_tokens_post_pad[{real_slot}]={post_pad_val} not a multiple "
        f"of block_size={block_size}."
    )

    # 2. expert_ids row for the real slot must be fully written (no sentinels).
    expert_ids_2d = expert_ids.view(max_loras, -1)
    row = expert_ids_2d[real_slot]
    assert (row != SENTINEL_EXPERT).all(), (
        f"expert_ids row for slot {real_slot} has unwritten entries "
        f"(sentinel={SENTINEL_EXPERT}); the align kernel skipped the real "
        f"LoRA slot."
    )
    # All written entries must be either a real expert id or -1 (inactive).
    assert ((row >= 0) & (row < num_experts) | (row == -1)).all()

    # 3. sorted_token_ids row must be fully written too.
    sorted_2d = sorted_token_ids.view(max_loras, -1)
    row = sorted_2d[real_slot]
    assert (row != SENTINEL_TOKEN).all(), (
        f"sorted_token_ids row for slot {real_slot} has unwritten entries "
        f"(sentinel={SENTINEL_TOKEN}); the align kernel skipped the real "
        f"LoRA slot."
    )

    # 4. Cross-check: the per-block expert_id must match the topk_id of every
    #    valid token index inside that block (same invariant as the existing
    #    test). This also implicitly validates the fix's output correctness.
    sorted_block = sorted_token_ids.view(max_loras, -1, block_size)[real_slot]
    expert_block = expert_ids_2d[real_slot]
    flat_topk = topk_ids.view(-1)
    for b in range(sorted_block.size(0)):
        blk = sorted_block[b]
        valid = blk[blk != topk_ids.numel()]
        if valid.numel() > 0:
            assert torch.all(flat_topk[valid] == expert_block[b]), (
                f"block {b} contains tokens not routed to expert "
                f"{expert_block[b].item()}"
            )


@pytest.mark.parametrize("max_loras", [1, 2])
@pytest.mark.parametrize("num_experts", [64, 128])
@pytest.mark.parametrize("topk_num", [6])
@pytest.mark.parametrize("block_size", [16])
def test_moe_lora_align_block_size_disabled_adapter_untouched(
    max_loras, num_experts, topk_num, block_size
):
    """Disabled adapter slot must not be touched by any of the three align kernels.

    Constructs a batch where slot 0 is present in `active_lora_ids` (i.e. tokens
    are routed to it from `token_lora_mapping`) but `adapter_enabled[0] == 0`.
    The align kernel (`moe_lora_align_block_size_kernel`) already early-returns
    for disabled adapters, leaving `token_mask` uninitialized for that row.
    Before this test / guard, the sort kernel
    (`lora_count_and_sort_expert_tokens_kernel`) did not check
    `adapter_enabled` and would traverse that garbage mask, polluting
    `sorted_token_ids` and `cumsum_buffer` for the disabled slot. The pollution
    was dormant (downstream consumers also skip disabled slots), but it breaks
    the expected invariant that disabled-slot output rows remain untouched.

    This test pins that invariant by initializing the output buffers with
    distinctive sentinels and asserting the disabled-slot rows are *unchanged*
    after the op completes.
    """
    random.seed(2)

    num_tokens = 16
    topk_ids = torch.zeros((num_tokens, topk_num), dtype=torch.int32)
    token_lora_mapping = torch.zeros((num_tokens,), dtype=torch.int32)
    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)
        for j in range(topk_num):
            topk_ids[i, j] = pool[j]
        # Route every token to the disabled slot 0 so that slot 0 actually
        # appears in active_lora_ids (otherwise the -1 / >=max_loras guards
        # alone would make this test uninteresting).
        token_lora_mapping[i] = 0

    topk_ids = topk_ids.to(DEVICE_TYPE)
    token_lora_mapping = token_lora_mapping.to(DEVICE_TYPE)

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32, device=DEVICE_TYPE)
    unique_ids = torch.unique(token_lora_mapping, sorted=True)
    lora_ids[: unique_ids.numel()] = unique_ids.to(torch.int32)
    # Slot 0 is present in active_lora_ids...
    assert (lora_ids == 0).any().item(), "test setup requires slot 0 to be active"

    # ...but disabled.
    adapter_enabled = torch.ones(
        (max_loras + 1,), dtype=torch.int32, device=DEVICE_TYPE
    )
    adapter_enabled[0] = 0

    SENTINEL_EXPERT = -2
    SENTINEL_TOKEN = -7
    SENTINEL_NPAD = -13

    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        SENTINEL_TOKEN,
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    expert_ids = torch.full(
        (max_loras * max_num_m_blocks,),
        SENTINEL_EXPERT,
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    num_tokens_post_pad = torch.full(
        (max_loras,), SENTINEL_NPAD, dtype=torch.int32, device=DEVICE_TYPE
    )

    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
    )

    # Disabled slot 0: output rows must be completely untouched.
    #
    # num_tokens_post_pad[0] must remain at the sentinel; a non-sentinel
    # value would indicate the align kernel ran for the disabled slot.
    assert num_tokens_post_pad[0].item() == SENTINEL_NPAD, (
        f"num_tokens_post_pad[0]={num_tokens_post_pad[0].item()} was modified "
        f"for a disabled adapter slot; the align kernel must skip "
        f"adapter_enabled==0 slots."
    )

    # expert_ids row for slot 0: must remain all-sentinel.
    expert_row0 = expert_ids.view(max_loras, -1)[0]
    assert (expert_row0 == SENTINEL_EXPERT).all(), (
        "expert_ids row for disabled slot 0 was partially written; the align "
        "kernel must skip adapter_enabled==0 slots."
    )

    # sorted_token_ids row for slot 0: must remain all-sentinel. This is the
    # assertion the sort-kernel guard specifically protects: without the
    # adapter_enabled check in lora_count_and_sort_expert_tokens_kernel, the
    # sort kernel reads uninitialized token_mask values and writes indices
    # into this row.
    sorted_row0 = sorted_token_ids.view(max_loras, -1)[0]
    assert (sorted_row0 == SENTINEL_TOKEN).all(), (
        "sorted_token_ids row for disabled slot 0 was polluted by the sort "
        "kernel; lora_count_and_sort_expert_tokens_kernel must skip "
        "adapter_enabled==0 slots."
    )


if __name__ == "__main__":
    pytest.main([__file__])
