#include <stdexcept>
#include <string>
#include <torch/script.h>

using c10::Dict;

namespace torchtext {
namespace {

typedef std::tuple<std::string, std::vector<int64_t>, std::vector<std::string>,
                   std::vector<torch::Tensor>>
    VectorsStates;

struct Vectors : torch::CustomClassHolder {
public:
  const std::string version_str_ = "0.0.1";

  ska_ordered::order_preserving_flat_hash_map<std::string, torch::Tensor>
      stovec_;
  // Dict<std::string, torch::Tensor> stovec_;
  std::vector<std::string> tokens_;
  torch::Tensor vectors_;
  torch::Tensor unk_tensor_;

  explicit Vectors(const std::vector<std::string> &tokens,
                   const torch::Tensor &vectors,
                   const torch::Tensor &unk_tensor)
      : tokens_(std::move(tokens)), vectors_(std::move(vectors)),
        unk_tensor_(std::move(unk_tensor)) {
    // guarding against size mismatch of vectors and tokens
    if (static_cast<int>(tokens.size()) != vectors.size(0)) {
      throw std::runtime_error(
          "Mismatching sizes for tokens and vectors. Size of tokens: " +
          std::to_string(tokens.size()) + ", size of vectors: " +
          std::to_string(vectors.size(0)) + ".");
    }

    stovec_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stovec_.find(tokens[i]) != stovec_.end()) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stovec_.insert_or_assign(std::move(tokens[i]), vectors_.select(0, i));
    }
  }

  torch::Tensor __getitem__(const std::string &token) const {
    const auto &item = stovec_.find(token);
    if (item != stovec_.end()) {
      return item->second;
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
    const auto &item = stovec_.find(token);
    if (item != stovec_.end()) {
      item->second = vector;
    } else {
      tokens_.push_back(token);
      vectors_ = torch::cat({vectors_, torch::unsqueeze(vector, /*dim=*/0)},
                            /*dim=*/0);
      stovec_.insert_or_assign(token, vectors_.select(0, stovec_.size()));
    }
  }

  int64_t __len__() { return stovec_.size(); }
};

VectorsStates _set_vectors_states(const c10::intrusive_ptr<Vectors> &self) {
  std::vector<int64_t> integers;
  std::vector<std::string> strings = self->tokens_;
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

  throw std::runtime_error("Found unexpected version for serialized Vector: " +
                           version_str + ".");
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
