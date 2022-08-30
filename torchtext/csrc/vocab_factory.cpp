#include <ATen/Parallel.h> // @manual
#include <pybind11/stl.h>
#include <torch/torch.h> // @manual
#include <torchtext/csrc/common.h>
#include <torchtext/csrc/vocab.h> // @manual
#include <torchtext/csrc/vocab_factory.h> // @manual

#include <fstream>
#include <stdexcept>
#include <string>

namespace torchtext {

Vocab _build_vocab_from_text_file_using_python_tokenizer(
    const std::string& file_path,
    const int64_t min_freq,
    py::object tokenizer) {
  // find number of lines
  int64_t num_lines = _infer_lines(file_path);
  // Read text from file and add tokens
  std::ifstream fin(file_path, std::ios::in);
  TORCH_CHECK(fin.is_open(), "Cannot open input file " + file_path);

  IndexDict counter;
  std::string line;
  for (int64_t i = 0; i < num_lines; i++) {
    std::getline(fin, line);
    std::vector<std::string> token_list =
        tokenizer(line).cast<std::vector<std::string>>();

    for (size_t i = 0; i < token_list.size(); i++) {
      std::string token = token_list[i];

      if (counter.find(token) == counter.end()) {
        counter[token] = 1;
      } else {
        counter[token] += 1;
      }
    }
  }

  // create tokens-frequency pairs
  std::vector<std::pair<std::string, int64_t>> token_freq_pairs;
  for (const auto& item : counter) {
    if (item.second >= min_freq) {
      token_freq_pairs.push_back(item);
    }
  }

  // sort tokens by frequency
  CompareTokens compare_tokens;
  std::sort(token_freq_pairs.begin(), token_freq_pairs.end(), compare_tokens);

  // Create final list of tokens
  StringList tokens;
  for (const auto& token_freq_pair : token_freq_pairs) {
    tokens.push_back(token_freq_pair.first);
  }

  return Vocab(std::move(tokens));
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

    for (size_t j = 0; j < token_list.size(); j++) {
      c10::IValue token_ref = token_list.get(j);
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
} // namespace torchtext
