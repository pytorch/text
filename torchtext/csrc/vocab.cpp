#include <ATen/Parallel.h> // @manual
#include <torch/torch.h> // @manual
#include <torchtext/csrc/common.h>
#include <torchtext/csrc/vocab.h> // @manual

#include <iostream>
#include <stdexcept>
#include <string>
namespace torchtext {

Vocab::Vocab(StringList tokens, const c10::optional<int64_t>& default_index)
    : stoi_(MAX_VOCAB_SIZE, -1), default_index_{default_index} {
  for (auto& token : tokens) {
    // throw error if duplicate token is found
    auto id = _find(c10::string_view{token});
    TORCH_CHECK(
        stoi_[id] == -1, "Duplicate token found in tokens list: " + token);

    _add(std::move(token));
  }
}

Vocab::Vocab(StringList tokens) : Vocab(std::move(tokens), {}) {}

int64_t Vocab::__len__() const {
  return itos_.size();
}

bool Vocab::__contains__(const c10::string_view& token) const {
  int64_t id = _find(token);
  if (stoi_[id] != -1) {
    return true;
  }
  return false;
}

int64_t Vocab::__getitem__(const c10::string_view& token) const {
  int64_t id = _find(token);
  if (stoi_[id] != -1)
    return stoi_[id];

  // throw error if default_index_ is not set
  TORCH_CHECK(
      default_index_.has_value(),
      "Token " + std::string(token) +
          " not found and default index is not set");

  // return default index if token is OOV
  return default_index_.value();
}

void Vocab::set_default_index(c10::optional<int64_t> index) {
  default_index_ = index;
}

c10::optional<int64_t> Vocab::get_default_index() const {
  return default_index_;
}

void Vocab::append_token(std::string token) {
  // throw error if token already exist in vocab
  auto id = _find(c10::string_view{token});
  TORCH_CHECK(
      stoi_[id] == -1,
      "Token " + token + " already exists in the Vocab with index: " +
          std::to_string(stoi_[id]));

  _add(std::move(token));
}

void Vocab::insert_token(std::string token, const int64_t& index) {
  // throw error if index is not valid
  TORCH_CHECK(
      index >= 0 && index <= __len__(),
      "Specified index " + std::to_string(index) +
          " is out of bounds for vocab of size " + std::to_string(__len__()));

  // throw error if token already present
  TORCH_CHECK(!__contains__(token), "Token " + token + " not found in Vocab");

  // need to offset all tokens greater than or equal index by 1
  for (size_t i = index; i < __len__(); i++) {
    stoi_[_find(c10::string_view{itos_[i]})] = i + 1;
  }

  stoi_[_find(c10::string_view{token})] = index;
  itos_.insert(itos_.begin() + index, std::move(token));
}

std::string Vocab::lookup_token(const int64_t& index) {
  // throw error if index is not valid
  TORCH_CHECK(
      index >= 0 && index < __len__(),
      "Specified index " + std::to_string(index) +
          " is out of bounds for vocab of size " + std::to_string(__len__()));

  return itos_[index];
}

StringList Vocab::lookup_tokens(const std::vector<int64_t>& indices) {
  // throw error if indices are not valid
  for (size_t i = 0; i < indices.size(); i++) {
    TORCH_CHECK(
        indices[i] >= 0 && indices[i] < __len__(),
        "Specified index " + std::to_string(indices[i]) + " at position " +
            std::to_string(i) + " is out of bounds for vocab of size " +
            std::to_string(__len__()));
  }

  std::vector<std::string> tokens(indices.size());
  for (size_t i = 0; i < indices.size(); i++) {
    tokens[i] = itos_[indices[i]];
  }
  return tokens;
}

std::vector<int64_t> Vocab::lookup_indices(
    const std::vector<c10::string_view>& tokens) {
  std::vector<int64_t> indices(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    indices[i] = __getitem__(tokens[i]);
  }
  return indices;
}

std::unordered_map<std::string, int64_t> Vocab::get_stoi() const {
  std::unordered_map<std::string, int64_t> stoi;
  // construct tokens and index list
  for (const auto& item : itos_) {
    stoi[item] = __getitem__(c10::string_view{item});
  }
  return stoi;
}

StringList Vocab::get_itos() const {
  return itos_;
}

VocabStates _serialize_vocab(const c10::intrusive_ptr<Vocab>& self) {
  std::vector<int64_t> integers;
  StringList strings = self->itos_;
  std::vector<torch::Tensor> tensors;

  if (self->default_index_.has_value()) {
    integers.push_back(self->default_index_.value());
  }

  VocabStates states = std::make_tuple(
      self->version_str_,
      std::move(integers),
      std::move(strings),
      std::move(tensors));
  return states;
}

c10::intrusive_ptr<Vocab> _deserialize_vocab(VocabStates states) {
  auto state_size = std::tuple_size<decltype(states)>::value;
  TORCH_CHECK(
      state_size == 4,
      "Expected deserialized Vocab to have 4 states but found " +
          std::to_string(state_size) + " states");

  auto& version_str = std::get<0>(states);
  auto& integers = std::get<1>(states);
  auto& strings = std::get<2>(states);
  auto& tensors = std::get<3>(states);

  // check tensors are empty
  TORCH_CHECK(tensors.size() == 0, "Expected `tensors` states to be empty");

  // throw error if version is not compatible
  TORCH_CHECK(
      version_str.compare("0.0.2") >= 0,
      "Found unexpected version for serialized Vocab: " + version_str);

  c10::optional<int64_t> default_index = {};
  if (integers.size() > 0) {
    default_index = integers[0];
  }
  return c10::make_intrusive<Vocab>(std::move(strings), default_index);
}

} // namespace torchtext
