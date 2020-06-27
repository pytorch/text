#include <stdexcept>
#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

struct Vectors : torch::CustomClassHolder {
public:
  // stovectors_ holds the serialized params passed in during initialization. We
  // need this because we need to be able to serialize the model so that we can
  // save the scripted object. Pickle will get the serialized model from these
  // members, thus they needs to be public.
  // std::vector<std::string> tokens_;
  // std::vector<torch::Tensor> vectors_;
  std::unordered_map<std::string, torch::Tensor> stovectors_;
  torch::Tensor unk_tensor_;

  explicit Vectors(const std::vector<std::string> &tokens,
                   const std::vector<torch::Tensor> &vectors,
                   const torch::Tensor &unk_tensor)
      : unk_tensor_(unk_tensor) {
    // guarding against size mismatch of vectors and tokens
    if (tokens.size() != vectors.size()) {
      throw std::runtime_error(
          "Mismatching sizes for tokens and vectors. Size of tokens: " +
          std::to_string(tokens.size()) + ", size of vectors: " +
          std::to_string(vectors.size()) + ".");
    }

    stovectors_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stovectors_.find(tokens[i]) != stovectors_.end()) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stovectors_[tokens[i]] = vectors[i];
    }
  }

  // constructor for loading serialized object
  explicit Vectors(
      const std::unordered_map<std::string, torch::Tensor> &stovectors,
      const torch::Tensor &unk_tensor)
      : stovectors_(stovectors), unk_tensor_(unk_tensor){};

  torch::Tensor __getitem__(const std::string &token) const {
    const auto &item = stovectors_.find(token);
    if (item != stovectors_.end()) {
      return item->second;
    }
    return unk_tensor_;
  }

  void __setitem__(const std::string &token, const torch::Tensor &vector) {
    stovectors_[token] = vector;
  }

  int64_t __len__() { return stovectors_.size(); }
};

// Registers our custom class with torch.
static auto vectors =
    torch::class_<Vectors>("torchtext", "Vectors")
        .def(torch::init<std::vector<std::string>, std::vector<torch::Tensor>,
                         torch::Tensor>())
        .def("__getitem__", &Vectors::__getitem__)
        .def("__setitem__", &Vectors::__setitem__)
        .def("__len__", &Vectors::__len__)
        .def_pickle(
            // __setstate__
            // [](const c10::intrusive_ptr<Vectors> &self)
            //     -> std::tuple<std::vector<std::string>,
            //                   std::vector<torch::Tensor>, torch::Tensor> {
            //   std::vector<std::string> tokens;
            //   std::vector<torch::Tensor> vectors;
            //   tokens.reserve(self->stovectors_.size());

            //   for (const auto &kv : self->stovectors_) {
            //     tokens.push_back(kv.first);
            //     vectors.push_back(kv.second);
            //   }

            //   std::tuple<std::vector<std::string>,
            //   std::vector<torch::Tensor>,
            //              torch::Tensor>
            //       states(tokens, vectors, self->unk_tensor_);
            //   return states;
            // },
            [](const c10::intrusive_ptr<Vectors> &self) -> std::tuple<
                std::unordered_map<std::string, torch::Tensor>, torch::Tensor> {

              std::tuple<std::unordered_map<std::string, torch::Tensor>,
                         torch::Tensor>
                  states(self->stovectors_, self->unk_tensor_);
              return states;
            },
            // __getstate__
            // [](std::tuple<std::vector<std::string>,
            // std::vector<torch::Tensor>,
            //               torch::Tensor>
            //        states) -> c10::intrusive_ptr<Vectors> {
            //   return c10::make_intrusive<Vectors>(
            //       std::move(std::get<0>(states)),
            //       std::move(std::get<1>(states)),
            //       std::move(std::get<2>(states)));
            // }
            [](std::tuple<std::unordered_map<std::string, torch::Tensor>,
                          torch::Tensor>
                   states) -> c10::intrusive_ptr<Vectors> {
              return c10::make_intrusive<Vectors>(
                  std::move(std::get<0>(states)),
                  std::move(std::get<1>(states)));
            });

} // namespace
} // namespace torchtext
