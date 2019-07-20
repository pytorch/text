#ifndef VOCABTEST_H
#define VOCABTEST_H

#include <gtest/gtest.h>
#include <torchtext/text.h>

using namespace torchtext::core;

TEST(Vocabulary, AddAndGet) {
  Vocabulary vocab;
  std::vector<std::string> words = {
      "hi", "hello", "hi", "bye", "how", "where", "hi", "how"};
  std::vector<int32_t> indexes = {0, 1, 0, 2, 3, 4, 0, 3};

  for (size_t i = 0; i < words.size(); ++i) {
    auto index = vocab.add_word(words[i]);
    EXPECT_EQ(index, indexes[i]);
  }

  for (size_t i = 0; i < words.size(); ++i) {
    auto index = vocab.get_index(words[i]);
    EXPECT_EQ(index, indexes[i]);
  }
}

#endif
