#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace torchtext {
namespace impl {

int64_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }

void infer_offsets(const std::string &file_path, int64_t num_lines,
                   int64_t chunk_size, std::vector<size_t> &offsets,
                   int64_t num_header_lines) {
  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  while (num_header_lines > 0) {
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    num_header_lines--;
  }
  offsets.push_back(fin.tellg());
  size_t offset = 0;
  while (fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n')) {
    offset++;
    if (offset % chunk_size == 0) {
      offsets.push_back(fin.tellg());
    }
  }
}

} // namespace impl
} // namespace torchtext
