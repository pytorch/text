#include "vocabulary.h"
#include <iostream>

namespace torchtext {
namespace core {
size_t Vocabulary::word_hash(const std::string& word) {
  return std::hash<std::string>()(word);
}

size_t Vocabulary::find(const std::string& word) {
  auto words_size = words.size();
  auto index = word_hash(word) % words_size;

  while (words[index].index != negative && words[index].word != word)
    index = (index + 1) & words_size;

  return index;
}

size_t Vocabulary::add_word(const std::string& word) {
  auto& item = words[find(word)];

  if (item.index == negative) {
    item.word = word;
    item.index = size++;
  } else
    ++item.count;

  return item.index;
}

size_t Vocabulary::get_index(const std::string& word) {
  return words[find(word)].index;
}

} // namespace core
} // namespace torchtext
