#include "regex.h"
#include <re2/re2.h>

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

namespace torchtext {

Regex::Regex(const std::string &re_str) : re_str_(re_str) {}

std::string Regex::Sub(const std::string &str, const std::string &repl) const {
  std::string mutable_str = str;
  RE2::GlobalReplace(&mutable_str, re_str_, repl);
  return mutable_str;
}

// Registers our custom class with torch.
static auto regex =
    torch::class_<Regex>("torchtext", "Regex")
        .def(torch::init<std::string>())
        .def("Sub", &Regex::Sub)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<Regex> &self) -> std::string {
              return self->re_str_;
            },
            // __setstate__
            [](std::string state) -> c10::intrusive_ptr<Regex> {
              return c10::make_intrusive<Regex>(std::move(state));
            });

} // namespace torchtext

// using namespace torchtext;
// namespace py = pybind11;

// PYBIND11_MODULE(_torchtext, m) {
//   py::class_<Regex>(m, "Regex")
//       .def(py::init<std::string>())
//       .def("Sub", &Regex::Sub);
// .def_pickle(
//     // __getstate__
//     [](const c10::intrusive_ptr<Regex> &self) -> std::string {
//       return self->re_str_;
//     },
//     // __setstate__
//     [](std::string state) -> c10::intrusive_ptr<Regex> {
//       return c10::make_intrusive<Regex>(std::move(state));
//     });
// }