#include <torchtext/csrc/export.h>

#include <cstdint>
#include <string>
#include <vector>

namespace torchtext {

namespace impl {
TORCHTEXT_API int64_t divup(int64_t x, int64_t y);
TORCHTEXT_API void infer_offsets(
    const std::string& file_path,
    int64_t num_lines,
    int64_t chunk_size,
    std::vector<size_t>& offsets,
    int64_t num_header_lines = 0);
} // namespace impl
} // namespace torchtext
