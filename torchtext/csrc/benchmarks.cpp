#include <benchmark/benchmark.h>
#include <fstream>
#include "text.h"

static void benchmark_basic_english_normalize(benchmark::State& state) {
  std::ifstream stream("benchmark-article.txt");
  auto it = std::istreambuf_iterator<std::string::value_type>(stream);
  auto lines = torch::text::core::impl::split(std::string(it, {}), '\n');

  for (auto _ : state) {
    std::vector<std::string> words;
    for (const auto& line : lines) {
      auto list = torch::text::core::impl::basic_english_normalize(line);
      words.insert(words.end(), list.begin(), list.end());
    }
  }
}

BENCHMARK(benchmark_basic_english_normalize);

BENCHMARK_MAIN();
