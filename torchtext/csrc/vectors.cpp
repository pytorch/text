#include <fstream>
// #include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/script.h>

using c10::Dict;

namespace torchtext {
namespace {

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

// trim str from start (in place)
static inline void _ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int ch) { return !std::isspace(ch); }));
}

// trim str from end (in place)
static inline void _rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// trim str from both ends (in place)
static inline void _trim(std::string &s) {
  _ltrim(s);
  _rtrim(s);
}

std::pair<int64_t, int64_t> _infer_shape(std::ifstream &fin,
                                         const int64_t delimiter_ascii = 32) {

  int64_t num_lines = 0, vector_dim = -1;
  std::vector<std::string> vec_str;
  std::string line, word;

  while (std::getline(fin, line)) {
    vec_str.clear();
    if (vector_dim == -1) {
      _trim(line);
      std::stringstream s(line);

      // std::cout << "[INFER LINE] " << line << std::endl;

      // get rid of the token
      std::getline(s, word, static_cast<char>(delimiter_ascii));
      // std::cout << "[INFER_WORD] " << word << std::endl;

      // we assume entries for vector are always seperated by ' '
      while (std::getline(s, word, ' ')) {
        // _trim(word);
        // std::cout << "[INFER_VAL] `" << word << "`" << std::endl;
        vec_str.push_back(word);
        // std::cout << "[VEC_SIZE] " << std::to_string(vec_str.size())
        //           << std::endl;
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
  std::cout << "[LINES, DIM] " << num_lines << ", " << vector_dim << std::endl;
  return std::make_pair(num_lines, vector_dim);
}

std::tuple<std::vector<std::string>, torch::Tensor, std::vector<std::string>>
_load_token_and_vectors_from_file(const std::string &file_path,
                                  const int64_t delimiter_ascii = 32) {
  std::cout << "Reading file " << file_path << std::endl;
  // std::cout << "[FILE_PATH] " << file_path << std::endl;
  // std::cout << "[DELIMITER] "
  //           << "`" << static_cast<char>(delimiter_ascii) << "`" << std::endl;

  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  std::pair<int64_t, int64_t> num_lines_and_vector_dim_pair =
      _infer_shape(fin, delimiter_ascii);
  int64_t num_lines = num_lines_and_vector_dim_pair.first;
  int64_t vector_dim = num_lines_and_vector_dim_pair.second;

  std::vector<std::string> tokens;
  torch::Tensor vectors = torch::zeros({num_lines, vector_dim});
  std::vector<float> vec_float;
  // std::vector<std::tuple<std::string, int64_t>> dup_tokens;
  std::vector<std::string> dup_tokens;
  std::unordered_set<std::string> tokens_set;

  tokens.reserve(num_lines);

  std::string line, token, vec_val;
  int64_t num_vecs_loaded = 0;

  while (std::getline(fin, line)) {
    // std::cout << "[CUR_LINE] " << line << std::endl;

    vec_float.clear();

    _trim(line);
    std::stringstream s(line);

    // read the token
    std::getline(s, token, static_cast<char>(delimiter_ascii));
    // std::cout << "[CUR_TOKEN] " << token << std::endl;

    // read every value of the vector and
    // store it in a string variable, 'vec_val'
    while (std::getline(s, vec_val, ' ')) {
      // std::cout << "[CUR_VAL] " << vec_val << std::endl;

      vec_float.push_back(std::stof(vec_val));
    }

    if (vector_dim == -1 && vec_float.size() > 1) {
      vector_dim = vec_float.size();
    } else if (vec_float.size() == 1) {
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
      // dup_tokens.push_back(std::make_tuple(token, vec_float.size() + 1));
      dup_tokens.push_back(token);
    }

    tokens_set.insert(token);
    tokens.push_back(token);
    vectors[num_vecs_loaded] = torch::tensor(vec_float);
    num_vecs_loaded++;
  }
  std::cout << "Done reading file." << std::endl;

  std::tuple<std::vector<std::string>, torch::Tensor, std::vector<std::string>>
      out_tuple(tokens, vectors.narrow(0, 0, num_vecs_loaded), dup_tokens);
  return out_tuple;
}

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
