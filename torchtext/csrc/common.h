#pragma once
#include <string>
#include <vector>
#include <c10/util/string_view.h>

namespace torchtext {
// typedef c10::string_view const_string;
// typedef std::vector<std::string> StringList;
// // typedef std::vector<const_string> ConstStringList;
// typedef std::vector<std::string> ConstStringList;

using const_string = c10::string_view;
using StringList = std::vector<std::string>;
using ConstStringList = std::vector<std::string>;
// typedef std::vector<const_string> ConstStringList;

namespace impl {
int64_t divup(int64_t x, int64_t y);
void infer_offsets(const std::string &file_path, int64_t num_lines,
                   int64_t chunk_size, std::vector<size_t> &offsets,
                   int64_t num_header_lines = 0);
} // namespace impl
} // namespace torchtext
