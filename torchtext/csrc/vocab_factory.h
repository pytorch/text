#include <pybind11/pybind11.h>
#include <vocab.h> // @manual

namespace py = pybind11;

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
} // namespace torchtext
