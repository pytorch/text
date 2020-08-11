#include <ATen/Parallel.h>
#include <stdexcept>
#include <string>
#include <vocab.h>

namespace torchtext {

Vocab::Vocab(const StringList &tokens, const IndexDict &stoindex,
             const std::string &unk_token, const int64_t unk_index)
    : itos_(std::move(tokens)), stoi_(std::move(stoindex)),
      unk_index_(std::move(unk_index)), unk_token_(std::move(unk_token)) {}

Vocab::Vocab(const StringList &tokens, const std::string &unk_token)
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

int64_t Vocab::__len__() const { return stoi_.size(); }

int64_t Vocab::__getitem__(const std::string &token) const {
  const auto &item = stoi_.find(token);
  if (item != stoi_.end()) {
    return item->value();
  }
  return unk_index_;
}

void Vocab::append_token(const std::string &token) {
  if (stoi_.find(token) == stoi_.end()) {
    stoi_.insert(std::move(token), stoi_.size());
  }
}

void Vocab::insert_token(const std::string &token, const int64_t &index) {
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
  for (size_t i = index; i < itos_.size(); i++) {
    stoi_.insert_or_assign(itos_[i], std::move(i + 1));
  }
  stoi_.insert(std::move(token), std::move(index));
  itos_.insert(itos_.begin() + index, std::move(token));

  // need to update unk_index in case token equals unk_token or token inserted
  // before unk_token
  unk_index_ = stoi_.find(unk_token_)->value();
}

std::string Vocab::lookup_token(const int64_t &index) {
  if (index < 0 || index > static_cast<int64_t>(itos_.size())) {
    throw std::runtime_error(
        "Specified index " + std::to_string(index) +
        " is out of bounds of the size of itos dictionary: " +
        std::to_string(itos_.size()) + ".");
  }

  return itos_[index];
}

StringList Vocab::lookup_tokens(const std::vector<int64_t> &indices) {
  std::vector<std::string> tokens(indices.size());
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); i++) {
    tokens[i] = lookup_token(indices[i]);
  }
  return tokens;
}

std::vector<int64_t> Vocab::lookup_indices(const StringList &tokens) {
  std::vector<int64_t> indices(tokens.size());
  for (int64_t i = 0; i < static_cast<int64_t>(tokens.size()); i++) {
    indices[i] = __getitem__(tokens[i]);
  }
  return indices;
}

c10::Dict<std::string, int64_t> Vocab::get_stoi() const { return stoi_; }
StringList Vocab::get_itos() const { return itos_; }

VocabStates _set_vocab_states(const c10::intrusive_ptr<Vocab> &self) {
  std::vector<int64_t> integers;
  StringList strings = self->itos_;
  strings.push_back(self->unk_token_);
  std::vector<torch::Tensor> tensors;

  VocabStates states = std::make_tuple(self->version_str_, std::move(integers),
                                       std::move(strings), std::move(tensors));
  return states;
}

c10::intrusive_ptr<Vocab> _get_vocab_from_states(VocabStates states) {
  auto state_size = std::tuple_size<decltype(states)>::value;
  if (state_size != 4) {
    throw std::runtime_error(
        "Expected deserialized Vocab to have 4 states but found only " +
        std::to_string(state_size) + " states.");
  }

  auto &version_str = std::get<0>(states);
  auto &integers = std::get<1>(states);
  auto &strings = std::get<2>(states);
  auto &tensors = std::get<3>(states);

  // check integers and tensors are empty
  if (integers.size() != 0 || tensors.size() != 0) {
    throw std::runtime_error(
        "Expected `integers` and `tensors` states to be empty.");
  }

  if (version_str.compare("0.0.1") >= 0) {
    std::string unk_token = strings.back();
    strings.pop_back(); // remove last element which is unk_token

    return c10::make_intrusive<Vocab>(std::move(strings), std::move(unk_token));
  }

  throw std::runtime_error(
      "Found unexpected version for serialized Vocab: " + version_str + ".");
}
} // namespace torchtext
