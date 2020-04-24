#include <torch/extension.h>
namespace torch {
namespace text {

    std::string get_s() {
        return "asdf";
    }
}
}

PYBIND11_MODULE(_torchtext, m) { m.def("get_random", torch::text::get_s); }
