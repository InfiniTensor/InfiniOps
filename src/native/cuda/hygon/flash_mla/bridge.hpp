#pragma once

#include <cstddef>
#include <cstdint>

namespace infini::ops::hygon_flash_mla {

struct Pages {
  const void* data;
  const int32_t* indices;
  const int32_t* lengths;
  int blocks;
  int page_tokens;
  int page_stride;
  int index_width;
};

struct Arguments {
  const void* query;
  const float* sink;
  void* output;
  void* workspace;
  size_t workspace_bytes;
  void* stream;
  int rows;
  int heads;
  int partitions;
  float scale;
  Pages swa;
  Pages extra;
};

inline size_t workspace_bytes(size_t rows, size_t partitions) {
  constexpr size_t heads = 64, dim = 512;
  const size_t queries = rows * heads;
  const size_t splits = (rows + partitions) * heads;
  // Each allocation in the bridge starts at a 256-byte boundary.
  return 8 * 256 + queries * dim * 4 + queries * 4 +
      splits * (dim + 1) * 4 + partitions * 32 + (rows + 1) * 4 + heads * 4;
}

void run(const Arguments& args);

}  // namespace infini::ops::hygon_flash_mla
