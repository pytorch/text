#include <torch/extension.h>
#include <map>
namespace torch {
namespace text {

    std::string get_s() {
        return "asdf";
    }
}
}

struct Vocab {
  Vocab(py::dict stoi, at::Tensor vectors, at::Tensor unk_vector, int64_t dim)
      : _vectors(vectors), _unk_vector(unk_vector), _dim(dim) {
        for (auto item :stoi) {
          std::string k = py::cast<std::string>(item.first);
          int64_t i = py::cast<int64_t>(item.second);
          _map.insert({k, i});
        }
      }

  at::Tensor __getitem__(const std::string& token) {
      py::object key = py::cast(token);
      if (_stoi.contains(key)) {
        return _vectors[at::Scalar(py::cast<int64_t>(_stoi[key]))];
      }
      return _unk_vector;
  }
  int64_t __len__() { return _vectors.size(0); }
  at::Tensor get_vecs_by_tokens(const std::vector<std::string> &tokens) {
    std::vector<at::Tensor> indices;
    for (const std::string &token : tokens) {
      indices.push_back(__getitem__(token));
    }
    return at::stack(indices);
  }

  std::map<std::string, int64_t> _map;
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
    c.def("get_vecs_by_tokens", &Vocab::get_vecs_by_tokens);
}
