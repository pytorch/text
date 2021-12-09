#include <algorithm>
#include <gpt2_bpe_tokenizer.h>
#include <regex.h> // @manual

namespace torchtext {
    const Regex kGPT2Regex(
        "(\\'s|\\'t|\\'re|\\'ve|\\'m|\\'ll|\\'d| ?\\pL+|"
        " ?\\pN+| ?[^\\s\\v\\pL\\pN]+|[\\s\\v]+)"
    );

    bool is_whitespace(const std::string &input) {
        for (std::string::const_iterator it = input.begin(); it != input.end(); ++it) {
            if (!isspace(*it)) {
                return false;
            }
        }
        return true;
    }

    std::vector<std::string> gpt2_bpe_tokenizer(std::string input) {
        std::string token;
        std::vector<std::string> tokens;
        re2::StringPiece inp(input);
        bool append_space = false;
        while (kGPT2Regex.FindAndConsume(&inp, &token)) {
            // tokens.push_back(token);
            // Check if whitespace
            if (is_whitespace(token)) {
                append_space = false;
                if (inp.empty()) {
                    tokens.push_back(token);
                } else {
                    if (token.length() > 1) {
                        tokens.push_back(token.substr(0, token.length() - 1));
                    }
                    if (token[token.length() - 1] == ' ') {
                        append_space = true;
                    } else {
                        tokens.push_back(token.substr(token.length() - 1));
                    }
                }
            } else if (append_space) {
                tokens.push_back(" " + token);
                append_space = false;
            } else {
                tokens.push_back(token);
            }
        }
        return tokens;
    }
}   // namespace torchtext