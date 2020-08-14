#include <ATen/Parallel.h>
#include <common.h>
#include <stdexcept>
#include <string>
#include <vocab.h>

namespace torchtext {

Vocab::Vocab(const StringList &tokens, const IndexDict &stoi,
             const std::string &unk_token, const int64_t unk_index)
    : itos_(std::move(tokens)), stoi_(std::move(stoi)),
      unk_index_(std::move(unk_index)), unk_token_(std::move(unk_token)) {}

Vocab::Vocab(const StringList &tokens, const std::string &unk_token)
    : itos_(std::move(tokens)), unk_token_(std::move(unk_token)) {
  stoi_.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); i++) {
    // tokens should not have any duplicates
    if (stoi_.find(tokens[i]) != stoi_.end()) {
#ifdef _MSC_VER
      std::cerr << "[RuntimeError] Duplicate token found in tokens list: "
                << tokens[i] << std::endl;
#endif
      throw std::runtime_error("Duplicate token found in tokens list: " +
                               tokens[i]);
    }
    stoi_[std::move(tokens[i])] = i;
  }
  unk_index_ = stoi_.find(unk_token)->second;
}

int64_t Vocab::__len__() const { return stoi_.size(); }

int64_t Vocab::__getitem__(const std::string &token) const {
  const auto &item = stoi_.find(token);
  if (item != stoi_.end()) {
    return item->second;
  }
  return unk_index_;
}

void Vocab::append_token(const std::string &token) {
  if (stoi_.find(token) == stoi_.end()) {
    // Note: we can't do `stoi_[token] = stoi_.size()` because of a bug
    // on Windows where the size gets updated before the assign occurs.
    // For example if the size of `stoi_` is 2, doing
    // `stoi_["test"] = stoi_.size()` will set `stoi_["test"]` to a
    // value of 3 instead of 2 on Windows stoi_[token] = itos_.size();
    stoi_[token] = itos_.size();
    itos_.push_back(token);
  }
}

void Vocab::insert_token(const std::string &token, const int64_t &index) {
  if (index < 0 || index > static_cast<int64_t>(stoi_.size())) {
#ifdef _MSC_VER
    std::cerr << "[RuntimeError] Specified index " << index
              << " is out of bounds of the size of stoi dictionary: "
              << stoi_.size() << std::endl;
#endif
    throw std::runtime_error(
        "Specified index " + std::to_string(index) +
        " is out of bounds of the size of stoi dictionary: " +
        std::to_string(stoi_.size()) + ".");
  }

  const auto &item = stoi_.find(token);
  // if item already in stoi we throw an error
  if (item != stoi_.end()) {
#ifdef _MSC_VER
    std::cerr << "[RuntimeError] Token " << token
              << " already exists in the Vocab with index: " << item->second
              << std::endl;
#endif
    throw std::runtime_error("Token " + token +
                             " already exists in the Vocab with index: " +
                             std::to_string(item->second) + ".");
  }

  // need to offset all tokens greater than or equal index by 1
  for (size_t i = index; i < itos_.size(); i++) {
    stoi_[itos_[i]] = i + 1;
  }
  stoi_[token] = index;
  itos_.insert(itos_.begin() + index, token);

  // need to update unk_index in case token equals unk_token or token
  // inserted before unk_token
  unk_index_ = stoi_.find(unk_token_)->second;
}

std::string Vocab::lookup_token(const int64_t &index) {
  if (index < 0 || index > static_cast<int64_t>(itos_.size())) {
#ifdef _MSC_VER
    std::cerr << "[RuntimeError] Specified index " << index
              << " is out of bounds of the size of itos dictionary: "
              << stoi_.size() << std::endl;
#endif
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

std::unordered_map<std::string, int64_t> Vocab::get_stoi() const {
  std::unordered_map<std::string, int64_t> stoi;
  stoi.reserve(stoi_.size());

  // construct tokens and index list
  for (const auto &item : stoi_) {
    stoi[item.first] = item.second;
  }
  return stoi;
}

StringList Vocab::get_itos() const { return itos_; }

int64_t _infer_lines(const std::string &file_path) {
  int64_t num_lines = 0;
  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  while (fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n')) {
    num_lines++;
  }
  return num_lines;
}

void parse_vocab_chunk(const std::string &file_path, size_t offset,
                       const int64_t start_line, const int64_t end_line,
                       std::shared_ptr<StringList> tokens) {
  std::ifstream fin;
  fin.open(file_path, std::ios::in);
  fin.seekg(offset);

  for (int64_t i = start_line; i < end_line; i++) {
    std::string token;
    fin >> token;
    fin >> std::ws;

    tokens->push_back(token);
  }
}

std::tuple<IndexDict, StringList>
_concat_tokens(std::vector<std::shared_ptr<StringList>> chunk_tokens,
               const std::string &unk_token, const int64_t min_freq,
               const int64_t num_lines) {
  TORCH_CHECK(chunk_tokens.size() > 0,
              "There must be at least 1 chunk to concatenate!");

  std::unordered_map<std::string, int64_t> tokens_freq;
  IndexDict stoindex;
  StringList tokens;
  stoindex.reserve(num_lines);
  tokens.reserve(num_lines);

  // create tokens frequency map
  for (size_t i = 0; i < chunk_tokens.size(); i++) {
    auto &subset_tokens = *chunk_tokens[i];
    for (size_t j = 0; j < subset_tokens.size(); j++) {
      // const auto &item = tokens_freq.find(subset_tokens[j]);
      if (tokens_freq.find(subset_tokens[j]) != tokens_freq.end()) {
        tokens_freq[subset_tokens[j]]++;
      } else {
        tokens_freq[subset_tokens[j]] = 1;
      }
    }
  }

  int64_t index = 0;
  // insert unk_token if not present
  if (tokens_freq.find(unk_token) == tokens_freq.end()) {
    std::cerr << "The `unk_token` " << unk_token
              << " wasn't found in the `ordered_dict`. Adding the `unk_token` "
                 "to the beginning of the Vocab."
              << std::endl;

    tokens.emplace_back(unk_token);
    stoindex[unk_token] = index;
  }

  // create tokens list and stoindex map
  for (size_t i = 0; i < chunk_tokens.size(); i++) {
    auto &subset_tokens = *chunk_tokens[i];
    for (size_t j = 0; j < subset_tokens.size(); j++) {
      if (tokens_freq[subset_tokens[j]] >= min_freq &&
          stoindex.find(subset_tokens[j]) == stoindex.end()) {
        tokens.emplace_back(subset_tokens[j]);
        stoindex[subset_tokens[j]] = index;
        index++;
      }
    }
  }
  return std::make_tuple(std::move(stoindex), std::move(tokens));
}

constexpr int64_t GRAIN_SIZE = 13107;
Vocab _load_vocab_from_file(const std::string &file_path,
                            const std::string &unk_token,
                            const int64_t min_freq, const int64_t num_cpus) {

  std::cerr << "[INFO] Reading file " << file_path << std::endl;

  int64_t num_lines = _infer_lines(file_path);
  int64_t chunk_size = impl::divup(num_lines, num_cpus);
  // Launching a thread on less lines than this likely has too much overhead.
  // TODO: Add explicit test beyond grain size to cover multithreading
  chunk_size = std::max(chunk_size, GRAIN_SIZE);

  std::vector<size_t> offsets;
  impl::infer_offsets(file_path, num_lines, chunk_size, offsets);

  std::vector<std::shared_ptr<StringList>> chunk_tokens;

  std::mutex m;
  std::condition_variable cv;
  std::atomic<int> counter(0);

  // create threads
  int64_t j = 0;
  for (int64_t i = 0; i < num_lines; i += chunk_size) {
    auto tokens_ptr = std::make_shared<StringList>();

    counter++;
    at::launch([&, file_path, num_lines, chunk_size, j, i, tokens_ptr]() {
      parse_vocab_chunk(file_path, offsets[j], i,
                        std::min(num_lines, i + chunk_size), tokens_ptr);
      std::lock_guard<std::mutex> lk(m);
      counter--;
      cv.notify_all();
    });
    chunk_tokens.push_back(tokens_ptr);
    j++;
  }

  // block until all threads finish execution
  std::unique_lock<std::mutex> lock(m);
  cv.wait(lock, [&counter] { return counter == 0; });

  IndexDict stoindex;
  StringList tokens;
  std::tie(stoindex, tokens) =
      _concat_tokens(chunk_tokens, unk_token, min_freq, num_lines);

  int64_t unk_index = stoindex.find(unk_token)->second;

  return Vocab(std::move(tokens), std::move(stoindex), unk_token, unk_index);
}

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
        "Expected deserialized Vocab to have 4 states but found " +
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
