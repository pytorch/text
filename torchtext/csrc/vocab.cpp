#include <torch/extension.h>
#include <unordered_map>

struct Vocab {
  Vocab(std::vector<std::string> itos, at::Tensor vectors,
        at::Tensor unk_vector)
      : _vectors(
            at::cat({vectors, unk_vector.reshape({1, unk_vector.size(0)})})),
        _unk_index(vectors.size(0)) {
    int64_t index = 0;
    _map.reserve(itos.size());
    for (const std::string & t : itos) {
      _map.insert({t, index});
      index++;
    }
  }
  at::Tensor __getitem__(const std::string &token) {
    auto search = _map.find(token);
    if (search == _map.end()) {
      return _vectors[_vectors.size(0) - 1].clone();
    }
    return _vectors[search->second];
  }
  int64_t __len__() { return _unk_index; }
  at::Tensor get_vecs_by_tokens(const std::vector<std::string> &tokens) {
    int64_t index = 0;
    at::Tensor indices = torch::empty({int64_t(tokens.size())},
                                      at::TensorOptions(torch::Dtype::Long));
    auto indices_accessor = indices.accessor<int64_t, 1>();
    size_t len = tokens.size();
    for (size_t i = 0; i < len; i++) {
      auto search = _map.find(tokens[i]);
      if (search != _map.end()) {
        indices_accessor[index] = search->second;
      } else {
        indices_accessor[index] = _unk_index;
      }
      index++;
    }
    return at::index_select(_vectors, 0, indices);
  }

private:
  std::unordered_map<std::string, int64_t> _map;
  at::Tensor _vectors;
  int64_t _unk_index;
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
