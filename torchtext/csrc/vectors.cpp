#include <future>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <torch/script.h>

using c10::Dict;

namespace torchtext {
namespace {

typedef std::tuple<std::vector<std::string>, torch::Tensor,
                   std::vector<std::string>>
    LoadedVectorsTuple;

struct Vectors : torch::CustomClassHolder {
public:
  Dict<std::string, torch::Tensor> stovec_;
  std::vector<std::string> tokens_;
  torch::Tensor vectors_;
  torch::Tensor unk_tensor_;

  explicit Vectors(const std::vector<std::string> &tokens,
                   const torch::Tensor &vectors,
                   const torch::Tensor &unk_tensor)
      : tokens_(std::move(tokens)), vectors_(std::move(vectors)),
        unk_tensor_(std::move(unk_tensor)) {
    // guarding against size mismatch of vectors and tokens
    if (static_cast<int>(tokens.size()) != vectors.size(0)) {
      throw std::runtime_error(
          "Mismatching sizes for tokens and vectors. Size of tokens: " +
          std::to_string(tokens.size()) + ", size of vectors: " +
          std::to_string(vectors.size(0)) + ".");
    }

    stovec_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stovec_.find(tokens[i]) != stovec_.end()) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stovec_.insert(std::move(tokens[i]), vectors_.select(0, i));
    }
  }

  torch::Tensor __getitem__(const std::string &token) const {
    const auto &item = stovec_.find(token);
    if (item != stovec_.end()) {
      return item->value();
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
      tokens_.push_back(token);
      vectors_ = torch::cat({vectors_, torch::unsqueeze(vector, /*dim=*/0)},
                            /*dim=*/0);
      stovec_.insert_or_assign(token, vectors_.select(0, stovec_.size()));
    }
  }

  int64_t __len__() { return stovec_.size(); }
};

// trim str from both ends (in place)
static inline void _trim(std::string &s) {
  // trim start
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int ch) { return !std::isspace(ch); }));
  // trim end
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

std::pair<int64_t, int64_t> _infer_shape(std::ifstream &fin,
                                         const int64_t delimiter_ascii) {

  int64_t num_lines = 0, vector_dim = -1;
  std::vector<std::string> vec_str;
  std::string line, word;

  while (std::getline(fin, line)) {
    vec_str.clear();
    if (vector_dim == -1) {
      _trim(line);
      std::stringstream s(line);

      // get rid of the token
      std::getline(s, word, static_cast<char>(delimiter_ascii));

      // we assume entries for vector are always seperated by ' '
      while (std::getline(s, word, ' ')) {
        vec_str.push_back(word);
      }

      // assuming word, [vector] format
      if (vec_str.size() > 2) {
        // the header present in some(w2v) formats contains two elements
        vector_dim = vec_str.size();
        num_lines++; // first element read
      }
    } else {
      num_lines++;
    }
  }
  fin.clear();
  fin.seekg(0, std::ios::beg);

  return std::make_pair(num_lines, vector_dim);
}

void _load_tokens_from_file_chunk(const std::string &file_path,
                                  const int64_t start_line,
                                  const int64_t num_lines,
                                  const int64_t vector_dim,
                                  const int64_t delimiter_ascii,
                                  std::promise<LoadedVectorsTuple> &&promise) {
  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  std::vector<std::string> tokens;
  tokens.reserve(num_lines);
  torch::Tensor vectors = torch::zeros({num_lines, vector_dim});
  std::vector<float> vec_float;
  std::vector<std::string> dup_tokens;
  std::unordered_set<std::string> tokens_set;

  std::string line, token, vec_val;
  int64_t num_vecs_loaded = 0;

  // get to line we care about
  for (int64_t i = 0; i < start_line; i++) {
    std::getline(fin, line);
  }

  for (int64_t i = start_line; i < start_line + num_lines; i++) {
    vec_float.clear();

    std::getline(fin, line);
    _trim(line);
    std::stringstream s(line);

    // read the token
    std::getline(s, token, static_cast<char>(delimiter_ascii));

    // read every value of the vector and
    // store it in a string variable, 'vec_val'
    while (std::getline(s, vec_val, ' ')) {
      vec_float.push_back(std::stof(vec_val));
    }

    if (vec_float.size() == 1) {
      std::cout << "Skipping token " << token
                << " with 1-dimensional vector ; likely a header" << std::endl;
      continue;
    } else if (vector_dim != static_cast<int64_t>(vec_float.size())) {
      throw std::runtime_error(
          "Vector for token " + token + " has " +
          std::to_string(vec_float.size()) +
          " but previously read vectors have " + std::to_string(vector_dim) +
          " dimensions. All vectors must have the same number of dimensions.");
    }

    if (tokens_set.find(token) != tokens_set.end()) {
      dup_tokens.push_back(token);
      continue;
    }

    tokens_set.insert(token);
    tokens.push_back(token);
    vectors[num_vecs_loaded] = torch::tensor(vec_float);
    num_vecs_loaded++;
  }

  promise.set_value(std::make_tuple(
      tokens, vectors.narrow(0, 0, num_vecs_loaded), dup_tokens));
}

void concat_loaded_vectors_tuples(std::vector<LoadedVectorsTuple> &tuples,
                                  const int64_t num_lines,
                                  const int64_t vector_dim,
                                  LoadedVectorsTuple *out_tuple) {
  std::vector<std::string> tokens;
  torch::Tensor vectors = torch::zeros({num_lines, vector_dim});
  std::vector<std::string> dup_tokens;
  std::unordered_set<std::string> tokens_set;
  int64_t num_vecs_loaded = 0;

  tokens.reserve(num_lines);

  // concat all loaded tuples
  for (size_t i = 0; i < tuples.size(); i++) {
    auto &&subset_tokens = std::move(std::get<0>(tuples[i]));
    auto &&subset_vectors = std::move(std::get<1>(tuples[i]));
    auto &&subset_dup_tokens = std::move(std::get<2>(tuples[i]));

    dup_tokens.insert(dup_tokens.end(), subset_dup_tokens.begin(),
                      subset_dup_tokens.end());

    // process tokens from each tuple
    for (size_t j = 0; j < subset_tokens.size(); j++) {
      if (tokens_set.find(subset_tokens[j]) != tokens_set.end()) {
        dup_tokens.push_back(subset_tokens[j]);
        continue;
      }

      tokens_set.insert(subset_tokens[j]);
      tokens.push_back(subset_tokens[j]);
      vectors[num_vecs_loaded] = subset_vectors[j];
      num_vecs_loaded++;
    }
  }
  *out_tuple = std::make_tuple(tokens, vectors.narrow(0, 0, num_vecs_loaded),
                               dup_tokens);
}

LoadedVectorsTuple
_load_token_and_vectors_from_file(const std::string &file_path,
                                  const int64_t delimiter_ascii = 32,
                                  const int64_t num_cpus = 10) {
  std::cout << "Reading file " << file_path << std::endl;

  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  std::pair<int64_t, int64_t> num_lines_and_vector_dim_pair =
      _infer_shape(fin, delimiter_ascii);
  int64_t num_lines = num_lines_and_vector_dim_pair.first;
  int64_t vector_dim = num_lines_and_vector_dim_pair.second;

  // need chunk size large enough to read entire file
  int64_t chunk_size = num_lines / num_cpus + 1;

  std::vector<std::future<LoadedVectorsTuple>> futures;
  std::vector<std::thread> threads;
  std::vector<LoadedVectorsTuple> tuples;

  // create threads
  for (int64_t i = 0; i < num_cpus; i++) {
    std::promise<LoadedVectorsTuple> p;
    std::future<LoadedVectorsTuple> f = p.get_future();
    futures.push_back(std::move(f));
    threads.push_back(
        std::thread(_load_tokens_from_file_chunk, file_path, i * chunk_size,
                    std::min(chunk_size, num_lines - (i * chunk_size)),
                    vector_dim, delimiter_ascii, std::move(p)));
  }

  // join threads
  for (int64_t i = 0; i < num_cpus; i++) {
    threads[i].join();
  }

  // get all loaded tuples
  for (int64_t i = 0; i < num_cpus; i++) {
    tuples.push_back(std::move(futures[i].get()));
  }

  LoadedVectorsTuple out_tuple;
  concat_loaded_vectors_tuples(tuples, num_lines, vector_dim, &out_tuple);
  return out_tuple;
}

// Registers our custom op with torch.
TORCH_LIBRARY(torchtext, m) {
  m.def("_load_token_and_vectors_from_file",
        &_load_token_and_vectors_from_file);
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
                std::vector<std::string>, torch::Tensor, torch::Tensor> {
              std::tuple<std::vector<std::string>, torch::Tensor, torch::Tensor>
                  states(self->tokens_, self->vectors_, self->unk_tensor_);
              return states;
            },
            // __getstate__
            [](std::tuple<std::vector<std::string>, torch::Tensor,
                          torch::Tensor>
                   states) -> c10::intrusive_ptr<Vectors> {
              return c10::make_intrusive<Vectors>(
                  std::move(std::get<0>(states)),
                  std::move(std::get<1>(states)),
                  std::move(std::get<2>(states)));
            });

} // namespace
} // namespace torchtext
