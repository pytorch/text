#include <torchtext/csrc/gpt2_bpe_tokenizer.h>
#include <torchtext/csrc/regex.h> // @manual

#include <algorithm>
#include <codecvt>
#include <locale>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

namespace torchtext {
const Regex kGPT2Regex(
    "(\\'s|\\'t|\\'re|\\'ve|\\'m|\\'ll|\\'d| ?\\pL+|"
    " ?\\pN+| ?[^\\s\\v\\pL\\pN]+|[\\s\\v]+)");

bool is_whitespace(const std::string& input) {
  for (const char& c : input) {
    if (!isspace(c)) {
      return false;
    }
  }
  return true;
}

template <class Key_, class Value_>
c10::Dict<Key_, Value_> _map_to_c10_dict(std::unordered_map<Key_, Value_> m) {
  c10::Dict<Key_, Value_> d;
  for (const auto& item : m)
    d.insert(item.first, item.second);
  return d;
}

template <class Key_, class Value_>
std::unordered_map<Key_, Value_> _c10_dict_to_map(c10::Dict<Key_, Value_> d) {
  std::unordered_map<Key_, Value_> m;
  for (const auto& item : d)
    m[item.key()] = item.value();
  return m;
}

std::vector<std::string> gpt2_bpe_pre_tokenizer(std::string input) {
  // Python implementation:
  // https://github.com/pytorch/fairseq/blob/main/fairseq/data/encoders/gpt2_bpe_utils.py#L69
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
  //     - IF token[-1] is space (ascii 32), then carry it over for next token,
  //     set append_space = True
  //     - ELSE make token[-1] its own token and add to return list
  // - ELSE IF prepend_space == True, prepend a space to the token and add to
  // return list
  // - ELSE, add token to return list
  std::vector<std::string> tokens;
  bool prepend_space = false;
  std::vector<std::string> index_matches;

  /* Notes on handling Special Tokens:
  We use regex pattern to first identify the special tokens in the input text.
  Other 'non-special' tokens go through pre-tokenization as usual, but special
  tokens skip those steps.

  Steps:
  * Loop over the set containing user-supplied strings that are to be treated as
  special tokens. This set gets created through the calls to
  `add_special_tokens` API.
    - form a regex pattern that helps in extracting special tokens from the
  input text.
  * Create a vector that contains chunks of input text, such that each chunk is
  either a sequence of non-special token or a single special token. For example,
  assuming <|special_tok|> and [SEP] are special tokens, the following text
      "This is an example with <|special_tok|> and [SEP] and [SPAM]."
  will get converted to a vector of strings:
      ["This is an example with", "<|special_tok|>", "and", "[SEP]", "and
  [SPAM]."]
    - if the input does not contain any special tokens, the vector will just
  contain a single token that is the whole original input text.
  * For all of the tokens in the above vector, we proceed with BPE tokenization
  as usual while skipping over certain steps as appropriate for special tokens.
  */

  if (bpe_never_split_set_.size() > 0) {
    std::string pattern = "";
    // Escape regex characters for matching special tokens. This is done to
    // ensure that characters like '|' in certain special tokens such as
    // <|endoftext|> don't get special regex treatment.
    for (std::string token : bpe_never_split_set_) {
      std::string::size_type pos = 0;
      while ((pos = token.find_first_of("|[]", pos)) != std::string::npos) {
        switch (token[pos]) {
          case '|':
            token.replace(pos, 1, "\\|");
            pos += 2;
            break;
          case '[':
            token.replace(pos, 1, "\\[");
            pos += 2;
            break;
          case ']':
            token.replace(pos, 1, "\\]");
            pos += 2;
            break;
        }
      }
      if (pattern.length() != 0) {
        pattern += "|";
      }
      pattern += token;
    }

    // break input into non-special and special parts
    const Regex specialTokenRegex("(" + pattern + ")");
    re2::StringPiece input_strp(input);
    std::string match;
    int64_t last_idx = 0;
    while (specialTokenRegex.FindAndConsume(&input_strp, &match)) {
      int64_t start_idx = input.size() - input_strp.size() - match.size();
      if (start_idx > last_idx) {
        if (isspace(input[start_idx - 1])) {
          // strip space on the left of the special token
          index_matches.push_back(
              input.substr(last_idx, start_idx - last_idx - 1));
        } else {
          index_matches.push_back(input.substr(last_idx, start_idx - last_idx));
        }
      }
      index_matches.push_back(input.substr(start_idx, match.size()));
      last_idx = start_idx + match.size();
      if (isspace(input[last_idx])) {
        // strip space on the right of the special token
        last_idx++;
      }
    }
    if (last_idx <= input.length() - 1) {
      index_matches.push_back(
          input.substr(last_idx, input.length() - last_idx));
    }
  } else {
    // input does not have any special tokens
    index_matches.push_back(input);
  }

  for (std::string index_token : index_matches) {
    bool is_never_split_token =
        bpe_never_split_set_.find(index_token) != bpe_never_split_set_.end();
    if (is_never_split_token) {
      // skip the rest of pre-tokenization work for special tokens
      tokens.push_back(index_token);
      continue;
    }
    re2::StringPiece inp(index_token);
    std::string token;
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
  }
  return tokens;
}

std::pair<std::string, std::string> split_tokens(
    std::string s,
    std::string delimiter) {
  auto pos = s.find(delimiter);
  TORCH_CHECK(pos != std::string::npos, "Expected `s`to contain `delimiter`");
  return std::make_pair(s.substr(0, pos), s.substr(pos + delimiter.length()));
}

int list_str_index(
    std::vector<std::string> list,
    std::string element,
    int start) {
  // Equivalent to: list.index(element, start)
  for (std::size_t i = start; i < list.size(); ++i) {
    if (list[i] == element) {
      return i;
    }
  }
  return -1;
}

std::string concatenate_strings(const std::vector<std::string>& list) {
  std::string ret = "";
  for (auto s : list)
    ret += s;
  return ret;
}

std::vector<std::string> get_pairs(
    std::vector<std::string> token_list,
    const std::string& seperator) {
  // For example: ["he", "l", "l", "o"]
  //    ==> ["he\u0001l", "l\u0001l", "l\u0001o"]
  std::unordered_set<std::string> pairs;
  std::vector<std::string> pairs_vec;

  if (token_list.empty())
    return pairs_vec;

  std::string prev_token = token_list[0];
  for (std::size_t i = 1; i < token_list.size(); ++i) {
    pairs.insert(prev_token + seperator + token_list[i]);
    prev_token = token_list[i];
  }
  pairs_vec.insert(pairs_vec.end(), pairs.begin(), pairs.end());
  return pairs_vec;
}

GPT2BPEEncoder::GPT2BPEEncoder(
    const c10::Dict<std::string, int64_t>& bpe_encoder,
    const c10::Dict<std::string, int64_t>& bpe_merge_ranks,
    const std::string& seperator,
    const c10::Dict<int64_t, std::string>& byte_encoder,
    bool caching_enabled)
    : inf_(bpe_merge_ranks.size() + 1),
      bpe_encoder_(std::move(bpe_encoder)),
      bpe_merge_ranks_(std::move(bpe_merge_ranks)),
      byte_encoder_(std::move(byte_encoder)),
      seperator_(std::move(seperator)),
      caching_enabled_(caching_enabled) {
  for (auto const& x : bpe_encoder_) {
    bpe_decoder_.insert(x.value(), x.key());
  }

  for (auto const& x : byte_encoder_) {
    byte_decoder_.insert(x.value(), x.key());
  }
}

GPT2BPEEncoder::GPT2BPEEncoder(
    const std::unordered_map<std::string, int64_t>& bpe_encoder,
    const std::unordered_map<std::string, int64_t>& bpe_merge_ranks,
    const std::string& seperator,
    const std::unordered_map<int64_t, std::string>& byte_encoder,
    bool caching_enabled)
    : GPT2BPEEncoder(
          _map_to_c10_dict<std::string, int64_t>(bpe_encoder),
          _map_to_c10_dict<std::string, int64_t>(bpe_merge_ranks),
          seperator,
          _map_to_c10_dict<int64_t, std::string>(byte_encoder),
          caching_enabled) {}

std::vector<std::string> GPT2BPEEncoder::ByteEncode_(
    std::string token,
    bool is_never_split_token) {
  // Equivalent to: (self.byte_encoder[b] for b in token.encode('utf-8')
  std::vector<std::string> encoded;
  if (is_never_split_token) {
    encoded.push_back(token);
  } else {
    for (auto& ch : token) {
      encoded.push_back(byte_encoder_.at((unsigned char)ch));
    }
  }
  return encoded;
}

int64_t GPT2BPEEncoder::GetBPEMergeRank_(std::string pair) {
  if (bpe_merge_ranks_.contains(pair)) {
    return bpe_merge_ranks_.at(pair);
  }
  return inf_;
}

std::string GPT2BPEEncoder::FindBestPair_(std::vector<std::string> pairs) {
  // Equivalent to:
  //    min(pairs, key = lambda pair: self.bpe_merge_ranks.get(pair,
  //    float('inf')))
  auto best_pair_idx = 0;
  auto best_rank = GetBPEMergeRank_(pairs[best_pair_idx]);

  for (std::size_t i = 1; i < pairs.size(); ++i) {
    auto rank = GetBPEMergeRank_(pairs[i]);
    if (rank < best_rank) {
      best_pair_idx = i;
      best_rank = rank;
    }
  }
  return pairs[best_pair_idx];
}

std::vector<std::string> GPT2BPEEncoder::BPE_(
    const std::vector<std::string>& token_list) {
  // Given a list of input tokens, keep finding the best bpe merge and
  // generate a new list of tokens until
  //  1) token list size reduced to 1
  //      OR
  //  2) can't find bpe merge
  auto concatenated = concatenate_strings(token_list);
  if (caching_enabled_ && cache_.contains(concatenated)) {
    return cache_.at(concatenated);
  }

  std::vector<std::string> tok_list = token_list;
  auto pairs = get_pairs(tok_list, seperator_);
  if (pairs.empty()) {
    return {concatenated};
  }
  while (true) {
    auto bigram = FindBestPair_(pairs);
    if (!bpe_merge_ranks_.contains(bigram))
      break;

    // Finding all indexes that token_list[i] == first and token_list[i+1] ==
    // second. After the loop, new token list will be
    //  1) first + second pair
    //  2) all the other tokens in the original token list
    //
    // For example: first="a" second="w" and token_list =
    // ["a", "w", "some", "a", "w", "e"]
    // Result: new_token_list = ["aw", "some", "aw", "e"]

    auto parts = split_tokens(bigram, seperator_);
    std::vector<std::string> new_token_list;
    std::size_t i = 0;
    while (i < tok_list.size()) {
      auto j = list_str_index(tok_list, parts.first, i);
      if (j != -1) {
        for (int k = i; k < j; k++)
          new_token_list.push_back(tok_list[k]);
        i = j;
      } else {
        for (std::size_t k = i; k < tok_list.size(); k++)
          new_token_list.push_back(tok_list[k]);
        break;
      }

      if (tok_list[i] == parts.first && i < (tok_list.size() - 1) &&
          tok_list[i + 1] == parts.second) {
        new_token_list.push_back(parts.first + parts.second);
        i += 2;
      } else {
        new_token_list.push_back(tok_list[i]);
        i += 1;
      }
    }

    tok_list = new_token_list;
    if (tok_list.size() == 1) {
      break;
    } else {
      pairs = get_pairs(tok_list, seperator_);
    }
  }

  if (caching_enabled_)
    cache_.insert(concatenated, tok_list);
  return tok_list;
}

std::vector<std::string> GPT2BPEEncoder::PreTokenize_(std::string input) {
  return gpt2_bpe_pre_tokenizer(input);
}

std::vector<int64_t> GPT2BPEEncoder::Encode(const std::string& text) {
  std::vector<int64_t> bpe_token_ids;
  for (const auto& token : PreTokenize_(text)) {
    if (added_tokens_encoder_.contains(token)) {
      bpe_token_ids.push_back(added_tokens_encoder_.at(token));
      continue;
    }
    bool is_never_split_token =
        bpe_never_split_set_.find(token) != bpe_never_split_set_.end();
    auto byte_encoded_token = ByteEncode_(token, is_never_split_token);
    for (const auto& bpe_token : BPE_(byte_encoded_token)) {
      bpe_token_ids.push_back(bpe_encoder_.at(bpe_token));
    }
  }
  return bpe_token_ids;
}

std::string GPT2BPEEncoder::Decode(const std::vector<int64_t>& tokens) {
  std::string text;
  bool is_prev_special = false;
  bool is_current_special = false;
  // setup converter for converting wide chars to/from chars
  using convert_type = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_type, wchar_t> converter;

  for (int tok_idx = 0; tok_idx < tokens.size(); tok_idx++) {
    const auto token = tokens[tok_idx];
    std::string decoded_token;

    if (added_tokens_decoder_.contains(token)) {
      // string is a special token from extended vocab
      decoded_token = added_tokens_decoder_.at(token);
      is_current_special = true;
    } else {
      const std::string str = bpe_decoder_.at(token);
      if (bpe_never_split_set_.find(str) != bpe_never_split_set_.end()) {
        // string is a special token from known vocab
        decoded_token = str;
        is_current_special = true;
      } else {
        // string is a regular token from known vocab
        is_current_special = false;
        const std::wstring ws = converter.from_bytes(str);
        for (wchar_t wchr : ws) {
          // get output character from byte decoder for each wide character
          unsigned char uchr = byte_decoder_.at(converter.to_bytes(wchr));
          decoded_token.push_back(uchr);
        }
      }
    }

    /* Fixing leading/trailing space(s)

    We need to ensure spaces before and after special tokens are removed
    appropirately. Assuming <|endoftext|> and HELLO are special tokens:
    string input: "<|endoftext|> <|endoftext|> and HELLO world !"
      is to be tokenized as:
    ['<|endoftext|>', '<|endoftext|>', 'and', 'HELLO', 'world', 'Ġ!']
      whereas an input like:
    "<|endoftext|> and anything else!", gets tokenized as:
    ['<|endoftext|>', 'and', 'Ġanything', 'Ġelse', '!']

    Hence while decoding the corresponding string tokens back to
    the original string text, we will have to insert those spaces back again.
    - Add empty space before a special token if it is not at the begining of the
      sentence and if it is not following another special token.
    - Add empty space after a special token if it is not at the end of the
    sentence.
    */

    // fix left space(s) for special tokens
    if (is_current_special && (tok_idx > 0 && !is_prev_special)) {
      text.push_back(' ');
    }
    text.append(decoded_token);
    // fix right space(s) for special tokens
    if (is_current_special && tok_idx != tokens.size() - 1) {
      text.push_back(' ');
    }
    is_prev_special = is_current_special;
  }
  return text;
}

std::vector<std::string> GPT2BPEEncoder::Tokenize(const std::string& text) {
  std::vector<std::string> bpe_tokens;
  for (const auto& token : PreTokenize_(text)) {
    bool is_never_split_token =
        bpe_never_split_set_.find(token) != bpe_never_split_set_.end();
    auto byte_encoded_token = ByteEncode_(token, is_never_split_token);
    for (const auto& bpe_token : BPE_(byte_encoded_token)) {
      bpe_tokens.push_back(bpe_token);
    }
  }
  return bpe_tokens;
}

int64_t GPT2BPEEncoder::AddSpecialTokens(
    const c10::Dict<std::string, std::string>& standard_special_tokens_dict,
    const std::vector<std::string>& additional_special_tokens) {
  int64_t newly_added = 0;

  /* All special tokens get added to `bpe_never_split_set_` set to avoid being
   * split during tokenization. Tokens are added to `added_tokens_encoder_` only
   * if they are not already known (i.e. not already present in `bpe_encoder_`).
   */

  // Loop for standard tokens such as "bos_token", "eos_token", etc.
  for (auto const& token : standard_special_tokens_dict) {
    if (added_tokens_encoder_.contains(token.value())) {
      continue;
    }
    bpe_never_split_set_.insert(token.value());
    if (!bpe_encoder_.contains(token.value())) {
      added_tokens_encoder_.insert(
          token.value(), bpe_encoder_.size() + added_tokens_encoder_.size());
      added_tokens_decoder_.insert(
          bpe_decoder_.size() + added_tokens_decoder_.size(), token.value());
      newly_added++;
    }
  }

  // Loop for any additional tokens
  for (auto const& token : additional_special_tokens) {
    if (added_tokens_encoder_.contains(token))
      continue;
    bpe_never_split_set_.insert(token);
    if (!bpe_encoder_.contains(token)) {
      added_tokens_encoder_.insert(
          token, bpe_encoder_.size() + added_tokens_encoder_.size());
      added_tokens_decoder_.insert(
          bpe_decoder_.size() + added_tokens_decoder_.size(), token);
      newly_added++;
    }
  }

  return newly_added;
}

std::unordered_map<std::string, int64_t> GPT2BPEEncoder::GetBPEEncoder() const {
  return _c10_dict_to_map(bpe_encoder_);
}

std::unordered_map<std::string, int64_t> GPT2BPEEncoder::GetBPEMergeRanks()
    const {
  return _c10_dict_to_map(bpe_merge_ranks_);
}

std::unordered_map<int64_t, std::string> GPT2BPEEncoder::GetByteEncoder()
    const {
  return _c10_dict_to_map(byte_encoder_);
}

GPT2BPEEncoderStatesPybind _serialize_gpt2_bpe_encoder_pybind(
    const c10::intrusive_ptr<GPT2BPEEncoder>& self) {
  return std::make_tuple(
      self->GetBPEEncoder(),
      self->GetBPEMergeRanks(),
      self->seperator_,
      self->GetByteEncoder(),
      self->caching_enabled_);
}

GPT2BPEEncoderStatesTorchbind _serialize_gpt2_bpe_encoder_torchbind(
    const c10::intrusive_ptr<GPT2BPEEncoder>& self) {
  return std::make_tuple(
      self->bpe_encoder_,
      self->bpe_merge_ranks_,
      self->seperator_,
      self->byte_encoder_,
      self->caching_enabled_);
}

c10::intrusive_ptr<GPT2BPEEncoder> _deserialize_gpt2_bpe_encoder_pybind(
    GPT2BPEEncoderStatesPybind states) {
  auto state_size = std::tuple_size<decltype(states)>::value;
  TORCH_CHECK(
      state_size == 5,
      "Expected deserialized GPT2BPEEncoder to have 5 states but found " +
          std::to_string(state_size) + " states");
  return c10::make_intrusive<GPT2BPEEncoder>(
      std::move(std::get<0>(states)),
      std::move(std::get<1>(states)),
      std::get<2>(states),
      std::move(std::get<3>(states)),
      std::get<4>(states));
}

c10::intrusive_ptr<GPT2BPEEncoder> _deserialize_gpt2_bpe_encoder_torchbind(
    GPT2BPEEncoderStatesTorchbind states) {
  auto state_size = std::tuple_size<decltype(states)>::value;
  TORCH_CHECK(
      state_size == 5,
      "Expected deserialized GPT2BPEEncoder to have 5 states but found " +
          std::to_string(state_size) + " states");
  return c10::make_intrusive<GPT2BPEEncoder>(
      std::move(std::get<0>(states)),
      std::move(std::get<1>(states)),
      std::get<2>(states),
      std::move(std::get<3>(states)),
      std::get<4>(states));
}

} // namespace torchtext
