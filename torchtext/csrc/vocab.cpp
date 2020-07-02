#include <stdexcept>
#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

using c10::Dict;

struct Vocab : torch::CustomClassHolder {
private:
  int64_t unk_index_;
  Dict<std::string, int64_t> stoi_;

public:
  // stoi_, and unordered_map holds the serialized params passed in
  // during initialization. We need this because we need to be able to serialize
  // the model so that we can save the scripted object. Pickle will get the
  // serialized model from these members, thus they needs to be public.
  std::vector<std::string> itos_;
  std::string unk_token_;

  explicit Vocab(const std::vector<std::string> &tokens,
                 const std::string &unk_token)
      : itos_(std::move(tokens)), unk_token_(std::move(unk_token)) {
    stoi_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stoi_.find(tokens[i]) != stoi_.end()) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stoi_.insert(std::move(tokens[i]), i);
    }
    unk_index_ = stoi_.find(unk_token)->value();
  }

  int64_t __len__() const { return stoi_.size(); }

  int64_t __getitem__(const std::string &token) const {
    const auto &item = stoi_.find(token);
    if (item != stoi_.end()) {
      return item->value();
    }
    return unk_index_;
  }

  void __setitem__(const std::string &token, const int64_t &index) {
    if (index < 0 || index > static_cast<int64_t>(stoi_.size())) {
      throw std::runtime_error(
          "Specified index " + std::to_string(index) +
          " is out of bounds of the size of stoi dictionary: " +
          std::to_string(stoi_.size()) + ".");
    }

    const auto &item = stoi_.find(token);
    // if item already in stoi we throw an error
    if (item != stoi_.end()) {
      throw std::runtime_error("Token " + token +
                               " already exists in the Vocab with index: " +
                               std::to_string(item->value()) + ".");
    }

    // need to offset all tokens greater than or equal index by 1
    for (auto &entry : stoi_) {
      if (entry.value() >= index) {
        stoi_.insert_or_assign(entry.key(), std::move(entry.value() + 1));
      }
    }
    stoi_.insert(std::move(token), std::move(index));

    // need to update unk_index in case token equals unk_token or token inserted
    // before unk_token
    unk_index_ = stoi_.find(unk_token_)->value();
  }

  void addToken(const std::string &token) {
    if (stoi_.find(token) == stoi_.end()) {
      stoi_.insert(std::move(token), stoi_.size());
    }
  }

  std::string lookupToken(const int64_t &index) {
    if (index < 0 || index > static_cast<int64_t>(itos_.size())) {
      throw std::runtime_error(
          "Specified index " + std::to_string(index) +
          " is out of bounds of the size of itos dictionary: " +
          std::to_string(itos_.size()) + ".");
    }

    return itos_[index];
  }

  std::vector<std::string> lookupTokens(const std::vector<int64_t> &indices) {
    std::vector<std::string> tokens(indices.size());
    for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); i++) {
      tokens[i] = lookupToken(indices[i]);
    }
    return tokens;
  }

  std::vector<int64_t> lookupIndices(const std::vector<std::string> &tokens) {
    std::vector<int64_t> indices(tokens.size());
    for (int64_t i = 0; i < static_cast<int64_t>(tokens.size()); i++) {
      indices[i] = __getitem__(tokens[i]);
    }
    return indices;
  }

  Dict<std::string, int64_t> getStoi() const { return stoi_; }
  std::vector<std::string> getItos() const { return itos_; }
};

// Registers our custom class with torch.
static auto vocab =
    torch::class_<Vocab>("torchtext", "Vocab")
        .def(torch::init<std::vector<std::string>, std::string>())
        .def("__getitem__", &Vocab::__getitem__)
        .def("__len__", &Vocab::__len__)
        .def("__setitem__", &Vocab::__setitem__)
        .def("addToken", &Vocab::addToken)
        .def("lookupToken", &Vocab::lookupToken)
        .def("lookupTokens", &Vocab::lookupTokens)
        .def("lookupIndices", &Vocab::lookupIndices)
        .def("getStoi", &Vocab::getStoi)
        .def("getItos", &Vocab::getItos)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<Vocab> &self)
                -> std::tuple<std::vector<std::string>, std::string> {
              std::tuple<std::vector<std::string>, std::string> states(
                  self->itos_, self->unk_token_);
              return states;
            },
            // __setstate__
            [](std::tuple<std::vector<std::string>, std::string> states)
                -> c10::intrusive_ptr<Vocab> {
              return c10::make_intrusive<Vocab>(std::move(std::get<0>(states)),
                                                std::move(std::get<1>(states)));
            });
} // namespace
} // namespace torchtext
