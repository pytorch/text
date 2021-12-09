#include <string>
#include <vector>

namespace torchtext {
    // Applies regex based pre-tokenization step for GPT-2 BPE tokenizer
    // and returns a list of tokens.
    std::vector<std::string> gpt2_bpe_pre_tokenizer(std::string input);
}   // namespace torchtext
