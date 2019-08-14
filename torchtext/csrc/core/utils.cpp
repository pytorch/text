#include "utils.h"

namespace torch {
namespace text {
namespace core {
namespace impl {

std::vector<std::string> basic_english_normalize(std::string line) {
  // std::vector<std::pair<std::regex, std::string>> _patterns = {
  //     {std::regex("\\\'"), " \'  "}, {std::regex("\\\""), ""},
  //     {std::regex("\\."), " . "},    {std::regex("<br \\/>"), " "},
  //     {std::regex(","), " , "},      {std::regex("\\("), " ( "},
  //     {std::regex("\\)"), " ) "},    {std::regex("\\!"), " ! "},
  //     {std::regex("\\?"), " ? "},    {std::regex("\\;"), " "},
  //     {std::regex("\\:"), " "},      {std::regex("\\s+"), " "}};
  std::vector<std::pair<std::regex, std::string>> _patterns = {};
  std::transform(line.begin(), line.end(), line.begin(), tolower);
  for (const auto& P : _patterns)
    line = std::regex_replace(line, P.first, P.second);
  return split(line, ' ');
}

std::vector<std::string> split(const std::string& text, char splitter) {
  std::string string;
  std::vector<std::string> list;
  size_t previous = 0, current = 0;

  while (current != std::string::npos) {
    current = text.find(splitter, previous);
    string = text.substr(previous, current - previous);
    if (!string.empty()) list.push_back(string);
    previous = current + 1;
  }

  return list;
}

}  // namespace impl
}  // namespace core
}  // namespace text
}  // namespace torch
