#include "regex.h"
#include <sstream>
#include <torch/script.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace torchtext {
namespace {

struct BasicEnglishNormalize : torch::CustomClassHolder {
private:
  std::vector<std::string> patterns_{"'",   "\"",  "\\.", "<br \\/>",
                                     ",",   "\\(", "\\)", "\\!",
                                     "\\?", "\\;", "\\:", "\\s+"};
  std::vector<std::string> replacements_{
      " '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "};
  std::vector<Regex> regex_objects_;

  // std::map<char[], std::string> patterns_list_ = {{R"'", ""}, {R"\'", " \'
  // "}};
  // , {"\.", " . "}, {"<br \/>", " "},
  // {",", " , "},  {"\(", " ( "},   {"\)", " ) "}, {"\!", " ! "},
  // {"\?", " ? "}, {"\;", " "},     {"\:", " "},   {"\s+", " "}};

  //       std::map<Regex, std::string> patterns_list_ =
  //       {
  //           {Regex("\'"), " \'  "},
  //           {Regex("\""), ""},
  //           {Regex("\."), " . "},
  //           {Regex("<br \/>"), " "},
  //           {Regex(","), " , "},
  //           {Regex("\("), " ( "},
  //           {Regex("\)"), " ) "},
  //           {Regex("\!"), " ! "},
  //           {Regex("\?"), " ? "},
  //           {Regex("\;"), " "},
  //           {Regex("\:"), " "},
  //           {Regex("\s+"), " ")};
  // };
  // std::vector<Regex> regex_objects_;

public:
  explicit BasicEnglishNormalize() {
    for (const auto &pattern : patterns_) {
      regex_objects_.push_back(Regex(pattern));
    }
  }

  std::vector<std::string> split_(std::string &str,
                                  const char &delimiter = ' ') const {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
      tokens.push_back(token);
    }

    return tokens;
  }

  std::vector<std::string> forward(const std::string &str) {
    std::string str_copy = str;

    // str tolower
    std::transform(str_copy.begin(), str_copy.end(), str_copy.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    for (size_t i = 0; i < regex_objects_.size(); i++) {
      str_copy = regex_objects_[i].Sub(str_copy, replacements_[i]);
    }

    std::cout << "[str_copy] " << str_copy << std::endl;
    return split_(str_copy);
  }
};

// Registers our custom class with torch.
static auto basic_english_normalize =
    torch::class_<BasicEnglishNormalize>("torchtext", "BasicEnglishNormalize")
        .def(torch::init<>())
        .def("forward", &BasicEnglishNormalize::forward);

} // namespace
} // namespace torchtext

// using namespace torchtext;
// namespace py = pybind11;

// PYBIND11_MODULE(_torchtext, m) {
//   py::class_<BasicEnglishNormalize>(m, "BasicEnglishNormalize")
//       .def(py::init<>())
//       .def("forward", &BasicEnglishNormalize::forward);
// }