#include <clip_tokenizer.h>
#include <regex.h> // @manual

#include <unordered_set>

namespace torchtext {
const Regex kCLIPRegex(
    "(?i)(<\\|startoftext\\|>|<\\|endoftext\\|>|\\'s|\\'t|\\'re|\\'ve|"
    "\\'m|\\'ll|\\'d|[\\pL]+|[\\pN]|[^\\s\\pL\\pN]+)");
const std::string kWhitespaceString("</w>");
const std::unordered_set<std::string> kSpecialTokens{
    "<|startoftext|>",
    "<|endoftext|>"};

std::vector<std::string> clip_pre_tokenizer(std::string input) {
  std::string token;
  std::vector<std::string> tokens;
  re2::StringPiece inp(input);
  while (kCLIPRegex.FindAndConsume(&inp, &token)) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string> CLIPEncoder::BPE_(
    const std::vector<std::string>& token_list) {
  // Given a list of input tokens, keep finding the best bpe merge and
  // generate a new list of tokens until
  //  1) token list size reduced to 1
  //      OR
  //  2) can't find bpe merge
  auto concatenated = concatenate_strings(token_list);
  if (caching_enabled_ && cache_.contains(concatenated)) {
    return cache_.at(concatenated);
  } else if (kSpecialTokens.find(concatenated) != kSpecialTokens.end()) {
    return {concatenated};
  }

  std::vector<std::string> tok_list(token_list.begin(), token_list.end() - 1);
  tok_list.push_back(token_list[token_list.size() - 1] + kWhitespaceString);
  auto pairs = get_pairs(tok_list, seperator_);
  if (pairs.empty()) {
    return {concatenated + kWhitespaceString};
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

std::vector<std::string> CLIPEncoder::PreTokenize_(std::string input) {
  return clip_pre_tokenizer(input);
}

std::vector<int64_t> CLIPEncoder::Encode(const std::string& text) {
  return GPT2BPEEncoder::Encode(text);
}

CLIPEncoderStatesPybind _serialize_clip_encoder_pybind(
    const c10::intrusive_ptr<CLIPEncoder>& self) {
  return std::make_tuple(
      self->GetBPEEncoder(),
      self->GetBPEMergeRanks(),
      self->seperator_,
      self->GetByteEncoder(),
      self->caching_enabled_);
}

CLIPEncoderStatesTorchbind _serialize_clip_encoder_torchbind(
    const c10::intrusive_ptr<CLIPEncoder>& self) {
  return std::make_tuple(
      self->bpe_encoder_,
      self->bpe_merge_ranks_,
      self->seperator_,
      self->byte_encoder_,
      self->caching_enabled_);
}

c10::intrusive_ptr<CLIPEncoder> _deserialize_clip_encoder_pybind(
    CLIPEncoderStatesPybind states) {
  auto state_size = std::tuple_size<decltype(states)>::value;
  TORCH_CHECK(
      state_size == 5,
      "Expected deserialized CLIPEncoder to have 5 states but found " +
          std::to_string(state_size) + " states");
  return c10::make_intrusive<CLIPEncoder>(
      std::move(std::get<0>(states)),
      std::move(std::get<1>(states)),
      std::get<2>(states),
      std::move(std::get<3>(states)),
      std::get<4>(states));
}

c10::intrusive_ptr<CLIPEncoder> _deserialize_clip_encoder_torchbind(
    CLIPEncoderStatesTorchbind states) {
  auto state_size = std::tuple_size<decltype(states)>::value;
  TORCH_CHECK(
      state_size == 5,
      "Expected deserialized CLIPEncoder to have 5 states but found " +
          std::to_string(state_size) + " states");
  return c10::make_intrusive<CLIPEncoder>(
      std::move(std::get<0>(states)),
      std::move(std::get<1>(states)),
      std::get<2>(states),
      std::move(std::get<3>(states)),
      std::get<4>(states));
}

}; // namespace torchtext
