#ifndef UTILS_H
#define UTILS_H

#include <torch/extensions.h>
#include <regex>
#include <vector>

namespace torch {
namespace text {
namespace core {
namespace impl {
std::vector<std::string> basic_english_normalize(std::string line);

std::vector<std::string> split(const std::string& text, char splitter);

}  // namespace impl
}  // namespace core
}  // namespace text
}  // namespace torch

#endif
