#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <stdint.h>
#include <string>
#include <vector>

namespace torchtext {
namespace core {
class Vocabulary {
  static const size_t MAX_VOCAB_SIZE = 30000000;
  static const size_t negative = static_cast<size_t>(-1);

  struct Item {
    std::string word;
    size_t count = 1;
    size_t index = negative;
  };

  size_t size;
  std::vector<Item> words;

  size_t word_hash(const std::string& word);
  size_t find(const std::string& word);

 public:
  inline Vocabulary() : size(0), words(MAX_VOCAB_SIZE) {}

  size_t add_word(const std::string& word);
  size_t get_index(const std::string& word);
};

} // namespace core
} // namespace torchtext

#endif
