#include "dictionary.h"
#include <iostream>

namespace torch {
namespace text {
namespace core {
size_t Dictionary::word_hash(const std::string& word) {
  size_t hash = 2166136261;
  for (auto C : word) {
    hash = hash ^ uint32_t(uint8_t(C));
    hash = hash * 16777619;
  }

  return hash;
}

size_t Dictionary::find(const std::string& word) {
  auto words_size = words.size();
  auto index = word_hash(word) % words_size;

  while (words[index].index != negative && words[index].word != word)
    index = (index + 1) & words_size;

  return index;
}

size_t Dictionary::add_word(const std::string& word) {
  auto& item = words[find(word)];

  if (item.index == negative) {
    item.word = word;
    item.index = size++;
  } else
    ++item.count;

  return item.index;
}

size_t Dictionary::get_index(const std::string& word) {
  return words[find(word)].index;
}

} // namespace core
} // namespace text
} // namespace torch
