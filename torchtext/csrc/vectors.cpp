#include <stdexcept>
#include <string>
#include <torch/script.h>

using c10::Dict;

namespace torchtext {
namespace {

struct Vectors : torch::CustomClassHolder {
public:
  Dict<std::string, torch::Tensor> stovec_;
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
      stovec_.insert(std::move(tokens[i]), vectors_.select(0, i));
    }
  }

  torch::Tensor __getitem__(const std::string &token) const {
    const auto &item = stovec_.find(token);
    if (item != stovec_.end()) {
      return item->value();
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
      item->value() = vector;
    } else {
      tokens_.push_back(token);
      vectors_ = torch::cat({vectors_, torch::unsqueeze(vector, /*dim=*/0)},
                            /*dim=*/0);
      stovec_.insert_or_assign(token, vectors_.select(0, stovec_.size()));
    }
  }

  int64_t __len__() { return stovec_.size(); }
};

c10::intrusive_ptr<Vectors>
_get_vectors_from_states(std::vector<c10::IValue> states) {
  if (states.size() <= 1) {
    throw std::runtime_error("Expected deserialized Vectors to have 2 or "
                             "more states but found only " +
                             std::to_string(states.size()) + " states.");
  }

  // version string is greater than or equal
  if (states[0].toStringRef().compare("0.0.1") >= 0) {
    std::vector<std::string> tokens =
        c10::toTypedList(states[1].toList()).vec();

    return c10::make_intrusive<Vectors>(std::move(tokens),
                                        std::move(states[2].toTensor()),
                                        std::move(states[3].toTensor()));
  }

  throw std::runtime_error("Found unexpected version for serialized Vector: " +
                           states[0].toStringRef() + ".");
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
            [](const c10::intrusive_ptr<Vectors> &self)
                -> std::vector<c10::IValue> {
              c10::IValue version_str("0.0.1");
              c10::IValue tokens(self->tokens_);
              c10::IValue vectors(self->vectors_);
              c10::IValue unk_tensor(self->unk_tensor_);

              std::vector<c10::IValue> states{
                  std::move(version_str), std::move(tokens), std::move(vectors),
                  std::move(unk_tensor)};
              return states;
            },
            // __getstate__
            [](std::vector<c10::IValue> states) -> c10::intrusive_ptr<Vectors> {

              return _get_vectors_from_states(states);
            });
// .def_pickle(
//     // __setstate__
//     [](const c10::intrusive_ptr<Vectors> &self) -> std::tuple<
//         std::vector<std::string>, torch::Tensor, torch::Tensor> {
//       std::tuple<std::vector<std::string>, torch::Tensor, torch::Tensor>
//           states(self->tokens_, self->vectors_, self->unk_tensor_);
//       return states;
//     },
//     // __getstate__
//     [](std::tuple<std::vector<std::string>, torch::Tensor,
//                   torch::Tensor>
//            states) -> c10::intrusive_ptr<Vectors> {
//       return c10::make_intrusive<Vectors>(
//           std::move(std::get<0>(states)),
//           std::move(std::get<1>(states)),
//           std::move(std::get<2>(states)));
//     });

} // namespace
} // namespace torchtext
