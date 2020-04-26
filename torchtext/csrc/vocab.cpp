#include <torch/extension.h>
namespace torch {
namespace text {

    std::string get_s() {
        return "asdf";
    }
}
}

struct Vocab {
  Vocab(py::dict stoi, at::Tensor vectors, at::Tensor unk_vector, int64_t dim)
      : _stoi(stoi), _vectors(vectors), _unk_vector(unk_vector), _dim(dim) {}

  at::Tensor __getitem__(std::string token) {
      if (_stoi.contains(py::cast(token))) {
        return _vectors[at::Scalar(py::cast<int64_t>(_stoi[py::cast(token)]))];
      }
      return _unk_vector;
  }
  int64_t __len__() { return _vectors.size(0); }

  py::dict _stoi;
  at::Tensor _vectors;
  at::Tensor _unk_vector;
  int64_t _dim;
};

PYBIND11_MODULE(_torchtext, m) { 
    auto c = py::class_<Vocab>(m, "Vocab");
    c.def(py::init<
            py::dict, // stoi
            at::Tensor, // vectors
            at::Tensor, // unk_vector
            int64_t>() // dim
         );
    c.def("__getitem__", &Vocab::__getitem__);
    c.def("__len__", &Vocab::__len__);
}
