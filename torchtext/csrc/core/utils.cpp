#include "utils.h"

namespace torch {
namespace text {
namespace core {
namespace impl {

std::vector<std::string> split(const std::string& text, char splitter) {
  std::string token;
  std::istringstream stream(text);
  std::vector<std::string> list;

  while (getline(stream, token, splitter))
    if (!token.empty())
      list.push_back(token);

  return list;
}

std::vector<std::string> split_tokenizer(const std::string& line) {
  return split(line, ' ');
}

} // namespace impl
} // namespace core
} // namespace text
} // namespace torch
