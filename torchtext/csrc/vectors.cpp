#include <stdexcept>
#include <string>
#include <torch/script.h>

using c10::Dict;

namespace torchtext {
namespace {

typedef Dict<std::string, torch::Tensor> VectorsDict;
typedef Dict<std::string, int64_t> IndexDict;
typedef std::tuple<std::string, std::vector<int64_t>, std::vector<std::string>,
                   std::vector<torch::Tensor>>
    VectorsStates;

struct Vectors : torch::CustomClassHolder {
public:
  const std::string version_str_ = "0.0.1";

  IndexDict stoindex_;
  VectorsDict stovec_;

  torch::Tensor vectors_;
  torch::Tensor unk_tensor_;

  explicit Vectors(const std::vector<std::string> &tokens,
                   const torch::Tensor &vectors,
                   const torch::Tensor &unk_tensor)
      : vectors_(std::move(vectors)), unk_tensor_(std::move(unk_tensor)) {
    // guarding against size mismatch of vectors and tokens
    if (static_cast<int>(tokens.size()) != vectors.size(0)) {
      throw std::runtime_error(
          "Mismatching sizes for tokens and vectors. Size of tokens: " +
          std::to_string(tokens.size()) +
          ", size of vectors: " + std::to_string(vectors.size(0)) + ".");
    }

    stoindex_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stoindex_.contains(tokens[i])) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stoindex_.insert(std::move(tokens[i]), i);
    }
  }

  torch::Tensor __getitem__(const std::string &token) const {
    const auto &item = stovec_.find(token);
    if (item != stovec_.end()) {
      return item->value();
    }

    const auto &item_index = stoindex_.find(token);
    if (item_index != stoindex_.end()) {
      auto vector = vectors_[item_index->value()];
      stovec_.insert(token, vector);
      return vector;
    }

    return unk_tensor_;
  }

  torch::Tensor lookup_vectors(const std::vector<std::string> &tokens) {
    std::vector<torch::Tensor> vectors;
    for (const std::string &token : tokens) {
      vectors.push_back(__getitem__(token));
    }

    return torch::stack(vectors, 0);
  }

  void __setitem__(const std::string &token, const torch::Tensor &vector) {
    const auto &item_index = stoindex_.find(token);
    if (item_index != stoindex_.end()) {
      stovec_.insert_or_assign(token, vector);
      vectors_[item_index->value()] = vector;
    } else {
      stoindex_.insert_or_assign(token, vectors_.size(0));
      stovec_.insert_or_assign(token, vector);
      // TODO: This could be done lazily during serialization (if necessary).
      // We would cycle through the vectors and concatenate those that aren't
      // views.
      vectors_ = at::cat({vectors_, vector.unsqueeze(0)});
    }
  }

  int64_t __len__() { return stovec_.size(); }
};

VectorsStates _set_vectors_states(const c10::intrusive_ptr<Vectors> &self) {
  std::vector<std::string> tokens(self->stoindex_.size());
  // reconstruct tokens list
  for (const auto &item : self->stoindex_) {
    tokens[item.value()] = item.key();
  }

  std::vector<int64_t> integers;
  std::vector<std::string> strings = std::move(tokens);
  std::vector<torch::Tensor> tensors{self->vectors_, self->unk_tensor_};

  VectorsStates states =
      std::make_tuple(self->version_str_, std::move(integers),
                      std::move(strings), std::move(tensors));

  return states;
}

c10::intrusive_ptr<Vectors> _get_vectors_from_states(VectorsStates states) {
  auto state_size = std::tuple_size<decltype(states)>::value;
  if (state_size != 4) {
    throw std::runtime_error(
        "Expected deserialized Vectors to have 4 states but found only " +
        std::to_string(state_size) + " states.");
  }

  auto &version_str = std::get<0>(states);
  auto &integers = std::get<1>(states);
  auto &strings = std::get<2>(states);
  auto &tensors = std::get<3>(states);

  // check integers are empty
  if (integers.size() != 0) {
    throw std::runtime_error("Expected `integers` states to be empty.");
  }

  if (version_str.compare("0.0.1") >= 0) {
    return c10::make_intrusive<Vectors>(
        std::move(strings), std::move(tensors[0]), std::move(tensors[1]));
  }

  throw std::runtime_error(
      "Found unexpected version for serialized Vector: " + version_str + ".");
}

// Registers our custom class with torch.
static auto vectors =
    torch::class_<Vectors>("torchtext", "Vectors")
        .def(torch::init<std::vector<std::string>, torch::Tensor,
                         torch::Tensor>())
        .def("__getitem__", &Vectors::__getitem__)
        .def("lookup_vectors", &Vectors::lookup_vectors)
        .def("__setitem__", &Vectors::__setitem__)
        .def("__len__", &Vectors::__len__)
        .def_pickle(
            // __setstate__
            [](const c10::intrusive_ptr<Vectors> &self) -> VectorsStates {
              return _set_vectors_states(self);
            },
            // __getstate__
            [](VectorsStates states) -> c10::intrusive_ptr<Vectors> {
              return _get_vectors_from_states(states);
            });

} // namespace
} // namespace torchtext
