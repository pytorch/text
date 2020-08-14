#include <ATen/Parallel.h>
#include <atomic>
#include <common.h>
#include <condition_variable>
#include <double-conversion/double-conversion.h>
#include <double-conversion/ieee.h>
#include <double-conversion/utils.h>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/script.h>
#include <vectors.h>

namespace torchtext {

Vectors::Vectors(const IndexMap &stoi, const torch::Tensor vectors,
                 const torch::Tensor &unk_tensor)
    : stoi_(stoi), vectors_(vectors), unk_tensor_(unk_tensor) {}

Vectors::Vectors(const std::vector<std::string> &tokens,
                 const torch::Tensor &vectors, const torch::Tensor &unk_tensor)
    : vectors_(std::move(vectors)), unk_tensor_(std::move(unk_tensor)) {
  // guarding against size mismatch of vectors and tokens
  if (static_cast<int>(tokens.size()) != vectors.size(0)) {
    throw std::runtime_error(
        "Mismatching sizes for tokens and vectors. Size of tokens: " +
        std::to_string(tokens.size()) +
        ", size of vectors: " + std::to_string(vectors.size(0)) + ".");
  }

  stoi_.reserve(tokens.size());
  stovec_.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); i++) {
    // tokens should not have any duplicates
    const auto &item_index = stoi_.find(tokens[i]);
    if (item_index != stoi_.end()) {
      throw std::runtime_error("Duplicate token found in tokens list: " +
                               tokens[i]);
    }
    stoi_[std::move(tokens[i])] = i;
  }
}

torch::Tensor Vectors::__getitem__(const std::string &token) {
  const auto &item = stovec_.find(token);
  if (item != stovec_.end()) {
    return item->second;
  }

  const auto &item_index = stoi_.find(token);
  if (item_index != stoi_.end()) {
    auto vector = vectors_[item_index->second];
    stovec_[token] = vector;
    return vector;
  }
  return unk_tensor_;
}

torch::Tensor Vectors::lookup_vectors(const std::vector<std::string> &tokens) {
  std::vector<torch::Tensor> vectors;
  for (const std::string &token : tokens) {
    vectors.push_back(__getitem__(token));
  }

  return torch::stack(vectors, 0);
}

void Vectors::__setitem__(const std::string &token,
                          const torch::Tensor &vector) {
  const auto &item_index = stoi_.find(token);
  if (item_index != stoi_.end()) {
    stovec_[token] = vector;
    vectors_[item_index->second] = vector;
  } else {
    stoi_[token] = vectors_.size(0);
    stovec_[token] = vector;
    // TODO: This could be done lazily during serialization (if necessary).
    // We would cycle through the vectors and concatenate those that aren't
    // views.
    vectors_ = at::cat({vectors_, vector.unsqueeze(0)});
  }
}

int64_t Vectors::__len__() { return stovec_.size(); }

std::tuple<int64_t, int64_t, int64_t> _infer_shape(const std::string &file_path,
                                                   const char delimiter) {

  int64_t num_header_lines = 0, num_lines = 0, vector_dim = -1;
  std::vector<std::string> vec_str;
  std::string line, word;

  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  while (std::getline(fin, line)) {
    vec_str.clear();
    if (vector_dim == -1) {
      std::istringstream s(line);

      // get rid of the token
      std::getline(s, word, delimiter);
      // we assume entries for vector are always seperated by ' '
      while (std::getline(s, word, ' ')) {
        vec_str.push_back(word);
      }

      // assuming word, [vector] format
      // the header present in some(w2v) formats contains two elements
      if (vec_str.size() <= 2) {
        num_header_lines++;
      } else if (vec_str.size() > 2) {
        vector_dim = vec_str.size();
        num_lines++; // first element read
      }
    } else {
      num_lines++;
    }
  }
  return std::make_tuple(num_lines, num_header_lines, vector_dim);
}

void parse_vectors_chunk(const std::string &file_path, size_t offset,
                         const int64_t start_line, const int64_t end_line,
                         const int64_t vector_dim, const char delimiter,
                         std::shared_ptr<StringList> tokens, float *data_ptr) {
  std::ifstream fin;
  fin.open(file_path, std::ios::in);
  fin.seekg(offset);

  int converter_flags = double_conversion::StringToDoubleConverter::NO_FLAGS;
  double_conversion::StringToDoubleConverter converter(
      converter_flags, 0.0f, double_conversion::Single::NaN(), NULL, NULL);

  for (int64_t i = start_line; i < end_line; i++) {
    std::string token;
    // read the token
    std::getline(fin, token, delimiter);
    tokens->push_back(token);

    std::string vec_val;
    int processed_characters_count;
    // read the vector
    for (int64_t j = 0; j < vector_dim; j++) {
      fin >> vec_val;
      const char *tmp_str = vec_val.c_str();
      data_ptr[i * vector_dim + j] = converter.StringToFloat(
          tmp_str, strlen(tmp_str), &processed_characters_count);
      TORCH_CHECK(processed_characters_count == strlen(tmp_str),
                  "Processed characters count didn't match vector string "
                  "length during string to float conversion!");
    }
    fin >> std::ws;
  }
}

std::tuple<IndexMap, StringList>
_concat_vectors(std::vector<std::shared_ptr<StringList>> chunk_tokens,
                const int64_t num_header_lines, const int64_t num_lines) {
  TORCH_CHECK(chunk_tokens.size() > 0,
              "There must be at least 1 chunk to concatenate!");
  IndexMap tokens;
  StringList dup_tokens;
  tokens.reserve(num_lines);

  // concat all loaded tuples
  int64_t count = num_header_lines;
  for (size_t i = 0; i < chunk_tokens.size(); i++) {
    auto &subset_tokens = *chunk_tokens[i];
    for (size_t j = 0; j < subset_tokens.size(); j++) {
      const auto &token_index = tokens.find(subset_tokens[j]);
      if (token_index != tokens.end()) {
        dup_tokens.emplace_back(std::move(subset_tokens[j]));
      } else {
        tokens[std::move(subset_tokens[j])] = count;
      }
      count++;
    }
  }
  return std::make_tuple(std::move(tokens), std::move(dup_tokens));
}

constexpr int64_t GRAIN_SIZE = 131072;
std::tuple<c10::intrusive_ptr<Vectors>, std::vector<std::string>>
_load_token_and_vectors_from_file(const std::string &file_path,
                                  const std::string delimiter_str,
                                  const int64_t num_cpus,
                                  c10::optional<torch::Tensor> opt_unk_tensor) {
  TORCH_CHECK(delimiter_str.size() == 1,
              "Only string delimeters of size 1 are supported.");
  std::cerr << "[INFO] Reading file " << file_path << std::endl;

  const char delimiter = delimiter_str.at(0);
  int64_t num_lines, num_header_lines, vector_dim;
  std::tie(num_lines, num_header_lines, vector_dim) =
      _infer_shape(file_path, delimiter);

  int64_t chunk_size = impl::divup(num_lines, num_cpus);
  // Launching a thread on less lines than this likely has too much overhead.
  // TODO: Add explicit test beyond grain size to cover multithreading
  chunk_size = std::max(chunk_size, GRAIN_SIZE);

  std::vector<size_t> offsets;
  impl::infer_offsets(file_path, num_lines, chunk_size, offsets,
                      num_header_lines);

  torch::Tensor data_tensor = torch::empty({num_lines, vector_dim});
  float *data_ptr = data_tensor.data_ptr<float>();
  std::vector<std::shared_ptr<StringList>> chunk_tokens;

  std::mutex m;
  std::condition_variable cv;
  std::atomic<int> counter(0);

  // create threads
  int64_t j = 0;
  for (int64_t i = num_header_lines; i < num_lines; i += chunk_size) {
    auto tokens_ptr = std::make_shared<StringList>();

    counter++;
    at::launch([&, file_path, num_lines, chunk_size, vector_dim, delimiter, j,
                i, tokens_ptr, data_ptr]() {
      parse_vectors_chunk(file_path, offsets[j], i,
                          std::min(num_lines, i + chunk_size), vector_dim,
                          delimiter, tokens_ptr, data_ptr);
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

  IndexMap stoi;
  StringList dup_tokens;
  std::tie(stoi, dup_tokens) =
      _concat_vectors(chunk_tokens, num_header_lines, num_lines);

  torch::Tensor unk_tensor;
  if (opt_unk_tensor) {
    unk_tensor = *opt_unk_tensor;
  } else {
    unk_tensor = torch::zeros({vector_dim}, torch::kFloat32);
  }
  auto result = std::make_tuple(
      c10::make_intrusive<Vectors>(Vectors(stoi, data_tensor, unk_tensor)),
      dup_tokens);
  return result;
}

VectorsStates _set_vectors_states(const c10::intrusive_ptr<Vectors> &self) {
  std::vector<std::string> tokens;
  std::vector<int64_t> indices;
  tokens.reserve(self->stoi_.size());
  indices.reserve(self->stoi_.size());

  // construct tokens and index list
  // we need to store indices because the `vectors_` tensor may have gaps
  for (const auto &item : self->stoi_) {
    tokens.push_back(item.first);
    indices.push_back(item.second);
  }

  std::vector<int64_t> integers = std::move(indices);
  std::vector<std::string> strings = std::move(tokens);
  std::vector<torch::Tensor> tensors{self->vectors_, self->unk_tensor_};

  VectorsStates states =
      std::make_tuple(self->version_str_, std::move(integers),
                      std::move(strings), std::move(tensors));

  return states;
}

c10::intrusive_ptr<Vectors> _get_vectors_from_states(VectorsStates states) {
  auto state_size = std::tuple_size<decltype(states)>::value;
  if (state_size != 4) {
    throw std::runtime_error(
        "Expected deserialized Vectors to have 4 states but found only " +
        std::to_string(state_size) + " states.");
  }

  auto &version_str = std::get<0>(states);
  auto &integers = std::get<1>(states);
  auto &strings = std::get<2>(states);
  auto &tensors = std::get<3>(states);

  if (version_str.compare("0.0.1") >= 0) {
    // check integers and tokens are same size
    if (integers.size() != strings.size()) {
      throw std::runtime_error(
          "Expected `integers` and `strings` states to be the same size.");
    }

    IndexMap stoi;
    stoi.reserve(integers.size());
    for (size_t i = 0; i < integers.size(); i++) {
      stoi[strings[i]] = integers[i];
    }

    return c10::make_intrusive<Vectors>(std::move(stoi), std::move(tensors[0]),
                                        std::move(tensors[1]));
  }

  throw std::runtime_error(
      "Found unexpected version for serialized Vector: " + version_str + ".");
}
} // namespace torchtext
