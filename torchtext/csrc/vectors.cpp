#include <stdexcept>
#include <string>
#include <torch/script.h>

using c10::Dict;

namespace torchtext {
namespace {

struct Vectors : torch::CustomClassHolder {
public:
  Dict<std::string, int64_t> stoi_;
  torch::Tensor vectors_;
  torch::Tensor unk_tensor_;

  explicit Vectors(const std::vector<std::string> &tokens,
                   const torch::Tensor &vectors,
                   const torch::Tensor &unk_tensor)
      : vectors_(std::move(vectors)), unk_tensor_(std::move(unk_tensor)) {
    // guarding against size mismatch of vectors and tokens
    if (tokens.size() != vectors.size(0)) {
      throw std::runtime_error(
          "Mismatching sizes for tokens and vectors. Size of tokens: " +
          std::to_string(tokens.size()) + ", size of vectors: " +
          std::to_string(vectors.size(0)) + ".");
    }

    stoi_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stoi_.find(tokens[i]) != stoi_.end()) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stoi_.insert(std::move(tokens[i]), i);
    }
  }

  // constructor for loading serialized object
  explicit Vectors(const Dict<std::string, int64_t> &stoi,
                   const torch::Tensor &vectors,
                   const torch::Tensor &unk_tensor)
      : stoi_(std::move(stoi)), vectors_(std::move(vectors)),
        unk_tensor_(std::move(unk_tensor)){};

  torch::Tensor __getitem__(const std::string &token) const {
    const auto &item = stoi_.find(token);
    if (item != stoi_.end()) {
      return vectors_[item->value()];
    }
    return unk_tensor_;
  }

  void __setitem__(const std::string &token, const torch::Tensor &vector) {
    const auto &item = stoi_.find(token);
    if (item != stoi_.end()) {
      vectors_[item->value()] = vector;
    } else {
      stoi_.insert(token, stoi_.size());
      vectors_ = torch::cat({vectors_, vector}, /*dim=*/0);
    }
  }

  int64_t __len__() { return stoi_.size(); }
};

// Registers our custom class with torch.
static auto vectors =
    torch::class_<Vectors>("torchtext", "Vectors")
        .def(torch::init<std::vector<std::string>, torch::Tensor,
                         torch::Tensor>())
        .def("__getitem__", &Vectors::__getitem__)
        .def("__setitem__", &Vectors::__setitem__)
        .def("__len__", &Vectors::__len__)
        .def_pickle(
            // __setstate__
            [](const c10::intrusive_ptr<Vectors> &self) -> std::tuple<
                Dict<std::string, int64_t>, torch::Tensor, torch::Tensor> {
              std::tuple<Dict<std::string, int64_t>, torch::Tensor,
                         torch::Tensor>
                  states(self->stoi_, self->vectors_, self->unk_tensor_);
              return states;
            },
            // __getstate__
            [](std::tuple<Dict<std::string, int64_t>, torch::Tensor,
                          torch::Tensor>
                   states) -> c10::intrusive_ptr<Vectors> {
              return c10::make_intrusive<Vectors>(
                  std::move(std::get<0>(states)),
                  std::move(std::get<1>(states)),
                  std::move(std::get<2>(states)));
            });

} // namespace
} // namespace torchtext
