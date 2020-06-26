#include <stdexcept>
#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

struct Vocab : torch::CustomClassHolder {
private:
  int64_t unk_index_;
  std::unordered_map<std::string, int64_t> stoi_;

public:
  // stoi_, and unordered_map holds the serialized params passed in
  // during initialization. We need this because we need to be able to serialize
  // the model so that we can save the scripted object. Pickle will get the
  // serialized model from these members, thus they needs to be public.
  std::vector<std::string> itos_;
  std::string unk_token_;

  explicit Vocab(const std::vector<std::string> &tokens,
                 const std::string &unk_token)
      : itos_(tokens), unk_token_(unk_token) {
    stoi_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stoi_.find(tokens[i]) != stoi_.end()) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stoi_[tokens[i]] = i;
    }
    unk_index_ = stoi_[unk_token];
  }

  int64_t __len__() const { return stoi_.size(); }

  int64_t __getitem__(const std::string &token) const {
    const auto &item = stoi_.find(token);
    if (item != stoi_.end()) {
      return item->second;
    }
    return unk_index_;
  }

  void addToken(const std::string &token) {
    if (stoi_.find(token) == stoi_.end()) {
      stoi_[token] = stoi_.size();
    }
  }

  void __setitem__(const std::string &token, const int64_t &index) {
    if (index < 0 || index > stoi_.size()) {
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
                               std::to_string(item->second) + ".");
    }

    // need to offset all tokens greater than or equal index by 1
    for (auto &entry : stoi_) {
      if (entry.second >= index) {
        entry.second++;
      }
    }
    stoi_[token] = index;

    // need to update unk_index in case token equals unk_token or token inserted
    // before unk_token
    unk_index_ = stoi_[unk_token_];
  }

  std::unordered_map<std::string, int64_t> getStoi() const { return stoi_; }
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
        .def("getStoi", &Vocab::getStoi)
        .def("getItos", &Vocab::getItos)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<Vocab> &self)
                -> std::tuple<std::vector<std::string>, std::string> {
              // std::vector<std::string> tokens;
              // tokens.reserve(self->stoi_.size());

              // for (const auto &kv : self->stoi_) {
              //   tokens.push_back(kv.first);
              // }

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
