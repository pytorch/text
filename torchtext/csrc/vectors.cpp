#include <double-conversion/double-conversion.h>
#include <double-conversion/ieee.h>
#include <double-conversion/utils.h>
#include <future>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <torch/script.h>

// timing
#include <chrono>
#include <ctime>
#include <ratio>

using c10::Dict;

namespace torchtext {
namespace {

// TODO: Instead of using Dict, could just use Vectors class
typedef Dict<std::string, torch::Tensor> VectorsDict;
typedef Dict<std::string, int64_t> IndexDict;
typedef std::vector<std::string> StringList;

struct Vectors : torch::CustomClassHolder {
public:
  IndexDict stoindex_;
  VectorsDict stovec_;
  torch::Tensor vectors_;
  torch::Tensor unk_tensor_;

  explicit Vectors(const IndexDict &stoindex, const torch::Tensor vectors,
                   const torch::Tensor &unk_tensor)
      : stoindex_(stoindex), vectors_(vectors), unk_tensor_(unk_tensor) {}

  explicit Vectors(const std::vector<std::string> &tokens,
                   const torch::Tensor &vectors,
                   const torch::Tensor &unk_tensor)
      : vectors_(vectors), unk_tensor_(std::move(unk_tensor)) {
    // guarding against size mismatch of vectors and tokens
    if (static_cast<int>(tokens.size()) != vectors.size(0)) {
      throw std::runtime_error(
          "Mismatching sizes for tokens and vectors. Size of tokens: " +
          std::to_string(tokens.size()) + ", size of vectors: " +
          std::to_string(vectors.size(0)) + ".");
    }

    stoindex_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stoindex_.contains(tokens[i])) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stoindex_.insert(std::move(tokens[i]), i);
    }
  }

  torch::Tensor __getitem__(const std::string &token) const {
    const auto &item = stovec_.find(token);
    if (item != stovec_.end()) {
      return item->value();
    }
    const auto &item_index = stoindex_.find(token);
    if (item_index != stoindex_.end()) {
      auto vector = vectors_[item_index->value()];
      stovec_.insert(token, vector);
      return vector;
    }
    return unk_tensor_;
  }

  torch::Tensor lookup_vectors(const std::vector<std::string> &tokens) {
    std::vector<torch::Tensor> vectors;
    for (const std::string &token : tokens) {
      vectors.push_back(__getitem__(token));
    }

    return torch::stack(vectors, 0);
  }

  void __setitem__(const std::string &token, const torch::Tensor &vector) {
    const auto &item = stovec_.find(token);
    if (item != stovec_.end()) {
      item->value() = vector;
    } else {
      stoindex_.insert_or_assign(token, vector.size(0));
      stovec_.insert_or_assign(token, vector);
      // TODO: This could be done lazily during serialization (if necessary).
      // We would cycle through the vectors and concatenate those that aren't
      // views.
      vectors_ = at::cat({vectors_, vector.unsqueeze(0)});
    }
  }

  int64_t __len__() { return stovec_.size(); }
};

inline int64_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }

void _infer_shape(const std::string &file_path, const int64_t delimiter_ascii,
                  std::tuple<int64_t, int64_t, int64_t> *out_tuple) {

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
      std::getline(s, word, static_cast<char>(delimiter_ascii));
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
  *out_tuple = std::make_tuple(num_lines, num_header_lines, vector_dim);
}

void _infer_offsets(const std::string &file_path, int64_t num_lines,
                    int64_t chunk_size, int64_t num_header_lines,
                    std::vector<size_t> *offsets) {

  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  while (num_header_lines > 0) {
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    num_header_lines--;
  }
  (*offsets).push_back(fin.tellg());
  size_t offset = 0;
  while (fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n')) {
    offset++;
    if (offset % chunk_size == 0) {
      (*offsets).push_back(fin.tellg());
    }
  }
}

void parse_chunk(const std::string &file_path, size_t offset,
                 const int64_t start_line, const int64_t end_line,
                 const int64_t vector_dim, const int64_t delimiter_ascii,
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
    std::getline(fin, token, static_cast<char>(delimiter_ascii));
    tokens->push_back(token);

    std::string vec_val;
    // read the vector
    for (int64_t j = 0; j < vector_dim; j++) {
      fin >> vec_val;
      const char *tmp_str = vec_val.c_str();
      int processed_characters_count;
      data_ptr[i * vector_dim + j] = converter.StringToFloat(
          tmp_str, strlen(tmp_str), &processed_characters_count);
    }
    fin >> std::ws;
  }
}

std::tuple<IndexDict, StringList>
concat_vectors(std::vector<std::shared_ptr<StringList>> chunk_tokens,
               torch::Tensor data_tensor, int64_t num_header_lines,
               int64_t num_lines) {
  TORCH_CHECK(chunk_tokens.size() > 0,
              "There must be at least 1 chunk to concatenate!");
  IndexDict tokens;
  StringList dup_tokens;
  tokens.reserve(num_lines);

  // concat all loaded tuples
  int64_t count = num_header_lines;
  for (size_t i = 0; i < chunk_tokens.size(); i++) {
    auto &subset_tokens = *chunk_tokens[i];
    for (size_t j = 0; j < subset_tokens.size(); j++) {
      if (tokens.contains(subset_tokens[j])) {
        dup_tokens.emplace_back(std::move(subset_tokens[j]));
      } else {
        tokens.insert(std::move(subset_tokens[j]), count);
      }
      count++;
    }
  }
  return std::make_tuple(tokens, dup_tokens);
}

constexpr int64_t GRAIN_SIZE = 131072;
std::tuple<c10::intrusive_ptr<Vectors>, std::vector<std::string>>
_load_token_and_vectors_from_file(const std::string &file_path,
                                  const int64_t delimiter_ascii,
                                  int64_t num_cpus,
                                  c10::optional<torch::Tensor> opt_unk_tensor) {
  std::cerr << "[INFO] Reading file " << file_path << std::endl;

  auto start = std::chrono::steady_clock::now();

  std::tuple<int64_t, int64_t, int64_t> out_tuple;
  int64_t num_lines, num_header_lines, vector_dim;

  _infer_shape(file_path, delimiter_ascii, &out_tuple);
  std::tie(num_lines, num_header_lines, vector_dim) = out_tuple;

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "_infer_shape elapsed time: " << elapsed_seconds.count()
            << "s\n";

  int64_t chunk_size = divup(num_lines, num_cpus);
  // Launching a thread on less lines than this likely has too much overhead.
  // TODO: Add explicit test beyond grain size to cover multithreading
  chunk_size = std::max(chunk_size, GRAIN_SIZE);

  start = std::chrono::steady_clock::now();

  std::vector<size_t> offsets;
  _infer_offsets(file_path, num_lines, chunk_size, num_header_lines, &offsets);

  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "_infer_offsets elapsed time: " << elapsed_seconds.count()
            << "s\n";

  start = std::chrono::steady_clock::now();

  torch::Tensor data_tensor = torch::empty({num_lines, vector_dim});
  float *data_ptr = data_tensor.data_ptr<float>();
  std::vector<std::thread> threads;
  std::vector<std::shared_ptr<StringList>> chunk_tokens;

  // create threads
  int64_t j = 0;
  for (int64_t i = num_header_lines; i < num_lines; i += chunk_size) {
    auto tokens_ptr = std::make_shared<StringList>();
    // TODO: Replace this with at::launch (this may require the use of mutexes)
    threads.push_back(std::thread(parse_chunk, file_path, offsets[j], i,
                                  std::min(num_lines, i + chunk_size),
                                  vector_dim, delimiter_ascii, tokens_ptr,
                                  data_ptr));
    chunk_tokens.push_back(tokens_ptr);
    j++;
  }

  // join threads
  for (auto &thread : threads) {
    thread.join();
  }
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "threads elapsed time: " << elapsed_seconds.count() << "s\n";

  start = std::chrono::steady_clock::now();
  IndexDict dict;
  StringList dup_tokens;
  std::tie(dict, dup_tokens) =
      concat_vectors(chunk_tokens, data_tensor, num_header_lines, num_lines);
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "concat_vectors elapsed time: " << elapsed_seconds.count()
            << "s\n";

  start = std::chrono::steady_clock::now();
  torch::Tensor unk_tensor;
  if (opt_unk_tensor) {
    unk_tensor = *opt_unk_tensor;
  } else {
    unk_tensor = torch::zeros({vector_dim});
  }
  auto result = std::make_tuple(
      c10::make_intrusive<Vectors>(Vectors(dict, data_tensor, unk_tensor)),
      dup_tokens);
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "result elapsed time: " << elapsed_seconds.count() << "s\n";
  return result;
}

// Registers our custom class with torch.
static auto vectors =
    torch::class_<Vectors>("torchtext", "Vectors")
        .def(torch::init<std::vector<std::string>, torch::Tensor,
                         torch::Tensor>())
        .def("__getitem__", &Vectors::__getitem__)
        .def("lookup_vectors", &Vectors::lookup_vectors)
        .def("__setitem__", &Vectors::__setitem__)
        .def("__len__", &Vectors::__len__)
        .def_pickle(
            // __setstate__
            [](const c10::intrusive_ptr<Vectors> &self) -> std::tuple<
                Dict<std::string, int64_t>, torch::Tensor, torch::Tensor> {
              return std::make_tuple(self->stoindex_, self->vectors_,
                                     self->unk_tensor_);
            },
            // __getstate__
            [](std::tuple<Dict<std::string, int64_t>, torch::Tensor,
                          torch::Tensor>
                   states) -> c10::intrusive_ptr<Vectors> {
              return c10::make_intrusive<Vectors>(
                  std::move(std::get<0>(states)),
                  std::move(std::get<1>(states)),
                  std::move(std::get<2>(states)));
            });

// Registers our custom op with torch.
TORCH_LIBRARY(torchtext, m) {
  m.def("_load_token_and_vectors_from_file",
        &_load_token_and_vectors_from_file);
}

} // namespace
} // namespace torchtext