#ifndef DICTTEST_H
#define DICTTEST_H

#include <gtest/gtest.h>
#include <torchtext/text.h>

using namespace torch::text::core;

TEST(Dictionary, AddAndGet) {
  Dictionary dict;
  std::vector<std::string> words = {
      "hi", "hello", "hi", "bye", "how", "where", "hi", "how"};
  std::vector<int32_t> indexes = {0, 1, 0, 2, 3, 4, 0, 3};

  for (size_t i = 0; i < words.size(); ++i) {
    auto index = dict.add_word(words[i]);
    EXPECT_EQ(index, indexes[i]);
  }

  for (size_t i = 0; i < words.size(); ++i) {
    auto index = dict.get_index(words[i]);
    EXPECT_EQ(index, indexes[i]);
  }
}

#endif
