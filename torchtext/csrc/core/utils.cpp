#include "utils.h"

static std::vector<std::pair<std::string, std::string>> _patterns = {
    {"\\\'", " \'  "},
    {"\\\"", ""},
    {"\\.", " . "},
    {"<br \\/>", " "},
    {",", " , "},
    {"\\(", " ( "},
    {"\\)", " ) "},
    {"\\!", " ! "},
    {"\\?", " ? "},
    {"\\;", " "},
    {"\\:", " "},
    {"\\s+", " "}};

std::vector<std::string> torch::text::core::impl::basic_english_normalize(
    std::string line) {
  std::transform(line.begin(), line.end(), line.begin(), tolower);
  for (const auto& P : _patterns)
    line = std::regex_replace(line, std::regex(P.first), P.second);
  return split(line, " ", true);
}

std::vector<std::string> torch::text::core::impl::split(
    const std::string& text,
    const std::string& splitter,
    bool skipEmptyParts) {
  std::vector<std::string> vector;
  size_t previous = 0;

  auto splitterSize = splitter.size();
  auto current = text.find(splitter);

  while (current != text.npos) {
    auto str = text.substr(previous, current - previous);
    if (!(skipEmptyParts && str.empty()))
      vector.push_back(str);

    previous = current + splitterSize;
    current = text.find(splitter, previous);
  }

  auto str = text.substr(previous, current - previous);
  if (!(skipEmptyParts && str.empty()))
    vector.push_back(str);

  return vector;
}
