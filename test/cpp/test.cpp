#include <gtest/gtest.h>
#include <torchtext/text.h>
#include <iostream>
#include "vocabtest.h"

using namespace std;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
