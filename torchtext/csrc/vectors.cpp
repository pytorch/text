#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

struct Vectors : torch::CustomClassHolder {
private:
  std::unordered_map<std::string, int> stoi_;

public:
  // tokens_ and vectors_ holds the serialized params passed in during
  // initialization. We need this because we need to be able to serialize the
  // model so that we can save the scripted object. Pickle will get the
  // serialized model from these members, thus they needs to be public.
  std::vector<std::string> tokens_;
  std::vector<torch::Tensor> vectors_;

  explicit Vectors(std::vector<std::string> tokens,
                   std::vector<torch::Tensor> vectors)
      : tokens_(tokens), vectors_(vectors) {
    for (std::size_t i = 0; i < tokens.size(); i++) {
      stoi_[tokens[i]] = static_cast<int>(i);
    }
  }

  torch::Tensor GetItem(const std::string &token) const {
    return vectors_.at(stoi_.at(token));
  }

  bool TokenExists(const std::string &token) const {
    return stoi_.find(token) != stoi_.end();
  }
};

// Registers our custom class with torch.
static auto vectors =
    torch::class_<Vectors>("torchtext", "Vectors")
        .def(
            torch::init<std::vector<std::string>, std::vector<torch::Tensor>>())
        .def("GetItem", &Vectors::GetItem)
        .def("TokenExists", &Vectors::TokenExists)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<Vectors> &self) -> std::pair<
                std::vector<std::string>, std::vector<torch::Tensor>> {
              std::pair<std::vector<std::string>, std::vector<torch::Tensor>>
                  pair(self->tokens_, self->vectors_);
              return pair;
            },
            // __setstate__
            [](std::pair<std::vector<std::string>, std::vector<torch::Tensor>>
                   states) -> c10::intrusive_ptr<Vectors> {
              return c10::make_intrusive<Vectors>(std::move(states.first),
                                                  std::move(states.second));
            });

} // namespace
} // namespace torchtext
