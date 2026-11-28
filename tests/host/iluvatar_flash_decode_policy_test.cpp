#include <cassert>
#include <limits>

#include "native/cuda/iluvatar/ops/flash_attn_with_kvcache/reduction_policy.h"

namespace detail = infini::ops::flash_attn_with_kvcache_iluvatar_detail;

int main() {
  detail::DecodeQkConfig config{};
  config.all_bf16 = config.int32_indices = config.contiguous_indices = true;
  config.batch = 2;
  config.heads = 32;
  config.kv_heads = 2;
  config.head_size = 128;
  config.blocks_per_sequence = 1;
  config.page_size = 4096;
  config.q_batch_stride = config.out_batch_stride = 4096;
  config.q_head_stride = config.out_head_stride = 128;
  config.k_block_stride = config.v_block_stride = 1048576;
  config.k_token_stride = config.v_token_stride = 128;
  config.k_head_stride = config.v_head_stride = 524288;
  config.q_address = config.k_address = config.v_address = 0x1000;
  assert(detail::CanBatchDecodeQkReduction(config));
  assert(detail::DecodeNumSplits(config) == 4);
  for (std::size_t batch : {1, 2, 4, 8, 16}) {
    for (std::size_t heads : {8, 16, 32, 64, 128}) {
      auto variant = config;
      variant.batch = batch;
      variant.heads = heads;
      variant.q_batch_stride = variant.out_batch_stride = heads * 128 + 128;
      variant.blocks_per_sequence = 32;
      variant.page_size = 256;
      assert(detail::CanBatchDecodeQkReduction(variant));
      assert(detail::DecodeNumSplits(variant) == (batch <= 4 ? 8 : 4));
    }
  }
  for (auto member : {&detail::DecodeQkConfig::all_bf16,
                      &detail::DecodeQkConfig::int32_indices,
                      &detail::DecodeQkConfig::contiguous_indices}) {
    auto variant = config;
    variant.*member = false;
    assert(!detail::CanBatchDecodeQkReduction(variant));
  }
  auto rejected = [&](auto member, auto value) {
    auto variant = config;
    variant.*member = value;
    assert(!detail::CanBatchDecodeQkReduction(variant));
  };
  rejected(&detail::DecodeQkConfig::has_alibi, true);
  rejected(&detail::DecodeQkConfig::batch, 0);
  rejected(&detail::DecodeQkConfig::batch, 17);
  rejected(&detail::DecodeQkConfig::heads, 0);
  rejected(&detail::DecodeQkConfig::heads, 129);
  rejected(&detail::DecodeQkConfig::kv_heads, 0);
  rejected(&detail::DecodeQkConfig::kv_heads, 3);
  rejected(&detail::DecodeQkConfig::head_size, 64);
  rejected(&detail::DecodeQkConfig::blocks_per_sequence, 0);
  rejected(&detail::DecodeQkConfig::page_size, 0);
  rejected(&detail::DecodeQkConfig::page_size, 257);
  rejected(&detail::DecodeQkConfig::page_size,
           std::numeric_limits<std::size_t>::max() - 255);
  rejected(&detail::DecodeQkConfig::q_batch_stride, 4095);
  rejected(&detail::DecodeQkConfig::q_head_stride, 129);
  rejected(&detail::DecodeQkConfig::out_batch_stride, 4095);
  rejected(&detail::DecodeQkConfig::out_head_stride, 129);
  rejected(&detail::DecodeQkConfig::k_token_stride, 129);
  rejected(&detail::DecodeQkConfig::v_token_stride, 129);
  rejected(&detail::DecodeQkConfig::k_head_stride, -128);
  rejected(&detail::DecodeQkConfig::v_block_stride, 0);
  rejected(&detail::DecodeQkConfig::q_address, 0x1002);
  rejected(&detail::DecodeQkConfig::k_address, 0x1008);
  rejected(&detail::DecodeQkConfig::v_address, 0x1008);
}
