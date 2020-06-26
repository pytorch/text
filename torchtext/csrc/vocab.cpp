#include <stdexcept>
#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

struct Vocab : torch::CustomClassHolder {
private:
  std::unordered_map<std::string, int64_t> stoi_;
  std::string unk_token_;
  int64_t unk_index_;

public:
  // // tokens_, Vocab_, and unk_tensor_ holds the serialized params passed in
  // // during initialization. We need this because we need to be able to
  // serialize
  // // the model so that we can save the scripted object. Pickle will get the
  // // serialized model from these members, thus they needs to be public.
  // std::vector<std::string> tokens_;
  // std::vector<torch::Tensor> Vocab_;
  // torch::Tensor unk_tensor_;

  explicit Vocab(const std::vector<std::string> &tokens,
                 const std::string unk_token, const int64_t &unk_index)
      : unk_token_(unk_token), unk_index_(unk_index) {
    stoi_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stoi_.find(tokens[i]) != stoi_.end()) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stoi_[tokens[i]] = i;
    }
  }

  int64_t GetItem(const std::string &token) const {
    auto &item = stoi_.find(token);
    if (item != stoi_.end()) {
      return item->second;
    }
    return unk_index_;
  }

  void AddItem(const std::string &token) {
    if (stoi_.find(token) == stoi_.end()) {
      stoi_[token] = stoi_.size();
    }
  }

  void AddItemToIndex(const std::string &token, const int64_t &index) {
    if (index < 0 || index > stoi_.size()) {
      throw std::runtime_error(
          "Specified index " + std::to_string(index) +
          " is out of bounds of the size of stoi dictionary: " +
          std::to_string(stoi_.size()) + ".");
    }

    auto &item = stoi_.find(token);
    // inserting into end of stoi_
    if (item == stoi_.end()) {
      stoi_[token] = index;
    }
    // need to offset all tokens greater than index by 1
    else {
      for (auto &entry : stoi_) {
        if (entry.second >= index) {
          entry.second++;
        }
      }
      stoi_[token] = index;
    }

    // need to update unk_index if unk_token
    if (token == unk_token_) {
      unk_index_ = index;
    }
  }
};

// Registers our custom class with torch.
static auto Vocab =
    torch::class_<Vocab>("torchtext", "Vocab")
        .def(torch::init<std::vector<std::string>, std::string, std::int64_t>,
             torch::Tensor > ())
        .def("GetItem", &Vocab::GetItem)
        .def("AddItem", &Vocab::AddItem)
        .def("AddItemToIndex", &Vocab::AddItemToIndex)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<Vocab> &self)
                -> std::tuple<std::vector<std::string>,
                              std::vector<torch::Tensor>, torch::Tensor> {
              std::tuple<std::vector<std::string>, std::vector<torch::Tensor>,
                         torch::Tensor>
                  states(self->tokens_, self->Vocab_, self->unk_tensor_);
              return states;
            },
            // __setstate__
            [](std::tuple<std::vector<std::string>, std::vector<torch::Tensor>,
                          torch::Tensor>
                   states) -> c10::intrusive_ptr<Vocab> {
              return c10::make_intrusive<Vocab>(std::move(std::get<0>(states)),
                                                std::move(std::get<1>(states)),
                                                std::move(std::get<2>(states)));
            });

} // namespace
} // namespace torchtext
