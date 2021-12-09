#include <algorithm>
#include <gpt2_bpe_tokenizer.h>
#include <regex.h> // @manual

namespace torchtext {
    const Regex kGPT2Regex(
        "(\\'s|\\'t|\\'re|\\'ve|\\'m|\\'ll|\\'d| ?\\pL+|"
        " ?\\pN+| ?[^\\s\\v\\pL\\pN]+|[\\s\\v]+)"
    );

    bool is_whitespace(const std::string &input) {
        for (const char& c : input) {
            if (!isspace(c)) {
                return false;
            }
        }
        return true;
    }

    std::vector<std::string> gpt2_bpe_pre_tokenizer(std::string input) {
        // Python implementation: https://github.com/pytorch/fairseq/blob/main/fairseq/data/encoders/gpt2_bpe_utils.py#L69
        // Original regex contains a negative lookahead pattern, which is not 
        // supported in re2. This implementation modifies the original regex in
        // the following two ways:
        // 1. Removes negative lookahead and adds a post-processing step instead.
        // 2. Replace all [\s] occurences with [\s\v] because re2 does not include
        //    vertical tab (\v) in whitespace. PCRE and Python re include \v in \s.
        //
        // Pseudocode of post-processing step:
        // - Loop over all tokens
        // - IF token is all whitespace:
        //   - set prepend_space to False
        //   - IF token is last token, add it to return vector
        //   - ELSE
        //     - If token length is >1, add token[0:len(token) - 1] to return list
        //     - IF token[-1] is space (ascii 32), then carry it over for next token, set append_space = True
        //     - ELSE make token[-1] its own token and add to return list
        // - ELSE IF prepend_space == True, prepend a space to the token and add to return list
        // - ELSE, add token to return list
        std::string token;
        std::vector<std::string> tokens;
        re2::StringPiece inp(input);
        bool prepend_space = false;
        while (kGPT2Regex.FindAndConsume(&inp, &token)) {
            if (is_whitespace(token)) {
                prepend_space = false;
                if (inp.empty()) { // token is last token
                    tokens.push_back(token);
                } else {
                    if (token.length() > 1) {
                        tokens.push_back(token.substr(0, token.length() - 1));
                    }
                    if (token[token.length() - 1] == ' ') { // last char is space
                        prepend_space = true;
                    } else { // push last whitespace char as a token if it is not a space
                        tokens.push_back(token.substr(token.length() - 1));
                    }
                }
            } else if (prepend_space) {
                tokens.push_back(" " + token);
                prepend_space = false;
            } else {
                tokens.push_back(token);
            }
        }
        return tokens;
    }
}   // namespace torchtext
