#include <stdexcept>
#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

// size_t str_hash(const std::string &str) {
//   size_t h = 2166136261;
//   for (size_t i = 0; i < str.size(); i++) {
//     h = h ^ uint32_t(uint8_t(str[i]));
//     h = h * 16777619;
//   }
//   return h;
// }

struct Vectors : torch::CustomClassHolder {
private:
  // Using unordered_map stovec
  std::unordered_map<std::string, torch::Tensor> stoi_;

  // Using unordered_map stovec and custom hash function
  // std::unordered_map<std::string, torch::Tensor, decltype(&str_hash)> stoi_;

  // Using unordered_map stoi and std::vector of vectors
  // std::unordered_map<std::string, int> stoi_;

  // Using map stoi and std::vector of vectors
  // std::map<std::string, int> stoi_;

  // Using unordered_map stoi and std::vector of vectors and custom hash
  // function
  // std::unordered_map<std::string, int, decltype(&str_hash)> stoi_;

public:
  // tokens_, vectors_, and unk_tensor_ holds the serialized params passed in
  // during initialization. We need this because we need to be able to serialize
  // the model so that we can save the scripted object. Pickle will get the
  // serialized model from these members, thus they needs to be public.
  std::vector<std::string> tokens_;
  std::vector<torch::Tensor> vectors_;
  torch::Tensor unk_tensor_;

  explicit Vectors(const std::vector<std::string> &tokens,
                   const std::vector<torch::Tensor> &vectors,
                   const torch::Tensor &unk_tensor)
      : tokens_(tokens), vectors_(vectors), unk_tensor_(unk_tensor) {
    // guarding against size mismatch of vectors and tokens
    if (tokens.size() != vectors.size()) {
      throw std::runtime_error(
          "Mismatching sizes for tokens and vectors. Size of tokens: " +
          std::to_string(tokens.size()) + ", size of vectors: " +
          std::to_string(vectors.size()) + ".");
    }

    stoi_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stoi_.find(tokens[i]) != stoi_.end()) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stoi_[tokens[i]] = vectors_[i];
      // stoi_[tokens[i]] = i;
    }
  }

  torch::Tensor GetItem(const std::string &token) const {
    if (stoi_.find(token) != stoi_.end()) {
      return stoi_.at(token);
      // return vectors_.at(stoi_.at(token));
    }
    return unk_tensor_;
  }

  void AddItem(const std::string &token, const torch::Tensor &vector) {
    stoi_[token] = vector;
    // stoi_[token] = vectors_.size();
    // vectors_.push_back(vector);
  }
};

// Registers our custom class with torch.
static auto vectors =
    torch::class_<Vectors>("torchtext", "Vectors")
        .def(torch::init<std::vector<std::string>, std::vector<torch::Tensor>,
                         torch::Tensor>())
        .def("GetItem", &Vectors::GetItem)
        .def("AddItem", &Vectors::AddItem)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<Vectors> &self)
                -> std::tuple<std::vector<std::string>,
                              std::vector<torch::Tensor>, torch::Tensor> {
              std::tuple<std::vector<std::string>, std::vector<torch::Tensor>,
                         torch::Tensor>
                  states(self->tokens_, self->vectors_, self->unk_tensor_);
              return states;
            },
            // __setstate__
            [](std::tuple<std::vector<std::string>, std::vector<torch::Tensor>,
                          torch::Tensor>
                   states) -> c10::intrusive_ptr<Vectors> {
              return c10::make_intrusive<Vectors>(
                  std::move(std::get<0>(states)),
                  std::move(std::get<1>(states)),
                  std::move(std::get<2>(states)));
            });

} // namespace
} // namespace torchtext
