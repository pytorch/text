#ifndef UTILS_H
#define UTILS_H

#include <regex>
#include <vector>

namespace torch {
namespace text {
namespace core {
namespace impl {

std::vector<std::string> split(const std::string& text, char splitter);
std::vector<std::string> split_tokenizer(const std::string& line);

} // namespace impl
} // namespace core
} // namespace text
} // namespace torch

#endif
