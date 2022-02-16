#include <ATen/Parallel.h> // @manual
#include <common.h>
#include <torch/torch.h> // @manual
#include <vocab.h> // @manual

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

int64_t _infer_lines(const std::string& file_path) {
  int64_t num_lines = 0;
  std::ifstream fin;
  fin.open(file_path, std::ios::in);
  TORCH_CHECK(fin.is_open(), "Cannot open input file " + file_path);

  while (fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n')) {
    num_lines++;
  }
  return num_lines;
}

void parse_vocab_file_chunk(
    const std::string& file_path,
    size_t offset,
    const int64_t start_line,
    const int64_t end_line,
    const std::shared_ptr<IndexDict>& counter) {
  std::ifstream fin(file_path, std::ios::in);
  TORCH_CHECK(fin.is_open(), "Cannot open input file " + file_path);

  fin.seekg(offset);

  for (int64_t i = start_line; i < end_line; i++) {
    std::string token;
    fin >> token;
    fin >> std::ws;

    if ((*counter).find(token) == (*counter).end()) {
      (*counter)[token] = 1;
    } else {
      (*counter)[token] += 1;
    }
  }
}

void parse_raw_text_file_chunk(
    const std::string& file_path,
    size_t offset,
    const int64_t start_line,
    const int64_t end_line,
    const std::shared_ptr<IndexDict>& counter,
    torch::jit::script::Module& module) {
  std::ifstream fin(file_path, std::ios::in);
  TORCH_CHECK(fin.is_open(), "Cannot open input file " + file_path);

  fin.seekg(offset);

  std::string line;
  for (int64_t i = start_line; i < end_line; i++) {
    std::getline(fin, line);
    auto token_list =
        module.forward(std::vector<c10::IValue>({c10::IValue(line)})).toList();

    for (size_t i = 0; i < token_list.size(); i++) {
      c10::IValue token_ref = token_list.get(i);
      std::string token = token_ref.toStringRef();

      if ((*counter).find(token) == (*counter).end()) {
        (*counter)[token] = 1;
      } else {
        (*counter)[token] += 1;
      }
    }
  }
}

StringList _concat_tokens(
    std::vector<std::shared_ptr<IndexDict>> chunk_counters,
    const int64_t min_freq,
    const int64_t num_lines,
    const bool sort_tokens) {
  TORCH_CHECK(
      chunk_counters.size() > 0,
      "There must be at least 1 chunk to concatenate!");

  IndexDict tokens_freq;
  StringList unique_tokens;
  unique_tokens.reserve(num_lines);

  // concatenate all counters
  for (size_t i = 0; i < chunk_counters.size(); i++) {
    auto& cur_counter = *chunk_counters[i];
    for (const auto& item : cur_counter) {
      int64_t cur_token_freq = item.second;
      if (tokens_freq.find(item.first) != tokens_freq.end()) {
        tokens_freq[item.first] += cur_token_freq;
      } else {
        tokens_freq[item.first] = cur_token_freq;
      }

      // add to tokens list only if all of the conditions are met:
      // 1. token is not empty
      // 2. we exceed min_freq for the first time
      if (item.first.length() &&
          tokens_freq[item.first] - cur_token_freq < min_freq &&
          tokens_freq[item.first] >= min_freq) {
        unique_tokens.push_back(item.first);
      }
    }
  }

  // create token freq pairs
  std::vector<std::pair<std::string, int64_t>> token_freq_pairs;

  for (std::string& token : unique_tokens) {
    auto token_freq = tokens_freq[token];
    token_freq_pairs.emplace_back(std::move(token), token_freq);
  }
  unique_tokens.clear();

  // sort tokens by freq
  if (sort_tokens) {
    CompareTokens compare_tokens;
    std::sort(token_freq_pairs.begin(), token_freq_pairs.end(), compare_tokens);
  }

  // update unique tokens with correct order
  for (auto& token_freq_pair : token_freq_pairs) {
    unique_tokens.emplace_back(std::move(token_freq_pair.first));
  }

  return unique_tokens;
}

constexpr int64_t GRAIN_SIZE = 13107;
Vocab _load_vocab_from_file(
    const std::string& file_path,
    const int64_t min_freq,
    const int64_t num_cpus) {
  int64_t num_lines = _infer_lines(file_path);
  int64_t chunk_size = impl::divup(num_lines, num_cpus);
  // Launching a thread on less lines than this likely has too much overhead.
  // TODO: Add explicit test beyond grain size to cover multithreading
  chunk_size = std::max(chunk_size, GRAIN_SIZE);

  std::vector<size_t> offsets;
  impl::infer_offsets(file_path, num_lines, chunk_size, offsets);

  std::vector<std::shared_ptr<IndexDict>> chunk_counters;

  std::mutex m;
  std::condition_variable cv;
  std::atomic<int> thread_count(0);

  // create threads
  int64_t j = 0;
  for (int64_t i = 0; i < num_lines; i += chunk_size) {
    auto counter_ptr = std::make_shared<IndexDict>();

    thread_count++;
    at::launch([&, file_path, num_lines, chunk_size, j, i, counter_ptr]() {
      parse_vocab_file_chunk(
          file_path,
          offsets[j],
          i,
          std::min(num_lines, i + chunk_size),
          counter_ptr);
      std::lock_guard<std::mutex> lk(m);
      thread_count--;
      cv.notify_all();
    });
    chunk_counters.push_back(counter_ptr);
    j++;
  }

  // block until all threads finish execution
  std::unique_lock<std::mutex> lock(m);
  cv.wait(lock, [&thread_count] { return thread_count == 0; });

  StringList tokens =
      _concat_tokens(chunk_counters, min_freq, num_lines, false);

  return Vocab(std::move(tokens));
}

Vocab _build_vocab_from_text_file(
    const std::string& file_path,
    const int64_t min_freq,
    const int64_t num_cpus,
    torch::jit::script::Module tokenizer) {
  int64_t num_lines = _infer_lines(file_path);
  int64_t chunk_size = impl::divup(num_lines, num_cpus);
  // Launching a thread on less lines than this likely has too much overhead.
  chunk_size = std::max(chunk_size, GRAIN_SIZE);

  std::vector<size_t> offsets;
  impl::infer_offsets(file_path, num_lines, chunk_size, offsets);

  std::vector<std::shared_ptr<IndexDict>> chunk_counters;

  std::mutex m;
  std::condition_variable cv;
  std::atomic<int> thread_count(0);

  // create threads
  int64_t j = 0;
  for (int64_t i = 0; i < num_lines; i += chunk_size) {
    auto counter_ptr = std::make_shared<IndexDict>();
    thread_count++;
    at::launch([&, file_path, num_lines, chunk_size, j, i, counter_ptr]() {
      parse_raw_text_file_chunk(
          file_path,
          offsets[j],
          i,
          std::min(num_lines, i + chunk_size),
          counter_ptr,
          tokenizer);
      std::lock_guard<std::mutex> lk(m);
      thread_count--;
      cv.notify_all();
    });
    chunk_counters.push_back(counter_ptr);
    j++;
  }

  // block until all threads finish execution
  std::unique_lock<std::mutex> lock(m);
  cv.wait(lock, [&thread_count] { return thread_count == 0; });

  StringList tokens = _concat_tokens(chunk_counters, min_freq, num_lines, true);

  return Vocab(std::move(tokens));
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
