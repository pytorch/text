#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>
#include <regex>
#include <vector>

namespace torch {
namespace text {
namespace core {
namespace impl {
std::vector<std::string> basic_english_normalize(std::string line);

std::vector<std::string> split(
    const std::string& text,
    const std::string& splitter,
    bool skipEmptyParts = false);

} // namespace impl
} // namespace core
} // namespace text
} // namespace torch

#endif
