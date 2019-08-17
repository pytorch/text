#include "utils.h"

namespace torch {
namespace text {
namespace core {
namespace impl {
static std::vector<std::pair<std::regex, std::string>> _patterns = {
    {std::regex("\\\'"), " \'  "},
    {std::regex("\\\""), ""},
    {std::regex("\\."), " . "},
    {std::regex("<br \\/>"), " "},
    {std::regex(","), " , "},
    {std::regex("\\("), " ( "},
    {std::regex("\\)"), " ) "},
    {std::regex("\\!"), " ! "},
    {std::regex("\\?"), " ? "},
    {std::regex("\\;"), " "},
    {std::regex("\\:"), " "},
    {std::regex("\\s+"), " "}};

std::vector<std::string> basic_english_normalize(std::string line) {
  std::transform(line.begin(), line.end(), line.begin(), tolower);
  for (const auto& P : _patterns)
    line = std::regex_replace(line, P.first, P.second);
  return split(line, ' ');
}

std::vector<std::string> split(const std::string& text, char splitter) {
  std::string token;
  std::istringstream stream(text);
  std::vector<std::string> list;

  while (getline(stream, token, splitter))
    if (!token.empty())
      list.push_back(token);

  return list;
}

} // namespace impl
} // namespace core
} // namespace text
} // namespace torch
