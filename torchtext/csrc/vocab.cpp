#include <torch/extension.h>
#include <unordered_map>

struct Vocab {
  Vocab(std::vector<std::string> itos, at::Tensor vectors,
        at::Tensor unk_vector)
      : _unk_vector(unk_vector) {

    int64_t index = 0;
    _map.reserve(itos.size());
    for (const std::string & t : itos) {
      _map.insert({t, index});
      index++;
    }

    index = 0;
    _vector_list.resize(vectors.size(0));
    for (at::Tensor t : vectors.unbind(0)) {
      _vector_list[index] = t;
      index++;
    }
  }

  at::Tensor __getitem__(const std::string &token) {
    auto search = _map.find(token);
    if (search == _map.end()) {
      return _unk_vector;
    }
    return _vector_list[search->second];
  }
  int64_t __len__() { return _vector_list.size(); }
  at::Tensor get_vecs_by_tokens(const std::vector<std::string> &tokens) {
    std::vector<at::Tensor> indices;
    for (const std::string &token : tokens) {
      indices.push_back(__getitem__(token));
    }
    return at::stack(indices);
  }

  std::unordered_map<std::string, int64_t> _map;
  std::vector<at::Tensor> _vector_list;
  at::Tensor _unk_vector;
  int64_t _dim;
};

PYBIND11_MODULE(_torchtext, m) { 
    auto c = py::class_<Vocab>(m, "Vocab");
    c.def(py::init<
            const std::vector<std::string>&, // stoi
            at::Tensor, // vectors
            at::Tensor>() // unk_vector
         );
    c.def("__getitem__", &Vocab::__getitem__);
    c.def("__len__", &Vocab::__len__);
    c.def("get_vecs_by_tokens", &Vocab::get_vecs_by_tokens);
}
