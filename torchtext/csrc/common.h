namespace torchtext {

namespace impl {
int64_t divup(int64_t x, int64_t y);
void infer_offsets(const std::string &file_path, int64_t num_lines,
                   int64_t chunk_size, std::vector<size_t> &offsets,
                   int64_t num_header_lines = 0);
} // namespace impl
} // namespace torchtext
