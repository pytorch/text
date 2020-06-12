#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

struct Vectors : torch::CustomClassHolder {
private:
  std::unordered_map<std::string, torch::Tensor> stovectors_;

public:
  // tokens_, vectors_, and unk_tensor_ holds the serialized params passed in
  // during initialization. We need this because we need to be able to serialize
  // the model so that we can save the scripted object. Pickle will get the
  // serialized model from these members, thus they needs to be public.
  std::vector<std::string> tokens_;
  std::vector<torch::Tensor> vectors_;
  torch::Tensor unk_tensor_;

  explicit Vectors(std::vector<std::string> tokens,
                   std::vector<torch::Tensor> vectors, torch::Tensor unk_tensor)
      : tokens_(tokens), vectors_(vectors), unk_tensor_(unk_tensor) {
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // guarding against size mismatch of vectors and tokens
      if (i >= vectors.size()) {
        break;
      }

      stovectors_[tokens[i]] = vectors_[i];
    }
  }

  torch::Tensor GetItem(const std::string &token) const {
    if (stovectors_.find(token) != stovectors_.end()) {
      return stovectors_.at(token);
    }
    return unk_tensor_;
  }
};

// Registers our custom class with torch.
static auto vectors =
    torch::class_<Vectors>("torchtext", "Vectors")
        .def(torch::init<std::vector<std::string>, std::vector<torch::Tensor>,
                         torch::Tensor>())
        .def("GetItem", &Vectors::GetItem)
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
