#include <torchtext/csrc/bert_tokenizer.h>
#include <utf8proc.h>

namespace torchtext {

std::string BERTEncoder::kUnkToken = "[UNK]";

int kMaxCharsPerWords = 100;

static bool _is_whitespace(uint32_t c) {
  if (c == '\t' || c == '\n' || c == '\r' || c == ' ') {
    return true;
  }
  return (UTF8PROC_CATEGORY_ZS == utf8proc_category(c));
}

static bool _is_control(uint32_t c) {
  if (c == '\t' || c == '\n' || c == '\r') {
    return false;
  }
  utf8proc_category_t cat = utf8proc_category(c);

  // Fixed: HF referece: All categories starting with 'C'
  return (
      cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF ||
      cat == UTF8PROC_CATEGORY_CN || cat == UTF8PROC_CATEGORY_CS ||
      cat == UTF8PROC_CATEGORY_CO);
}

static bool _is_chinese_char(uint32_t cp) {
  if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F)) {
    return true;
  }
  return false;
}

static bool _is_punct_char(uint32_t cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
    return true;
  }

  if (cp == ' ') {
    return false;
  }

  int cate = static_cast<int>(utf8proc_category(cp));
  return (cate >= 12 && cate <= 18);
}

static UString _convert_to_unicode(const std::string& text) {
  size_t i = 0;
  UString ret;
  while (i < text.size()) {
    uint32_t codepoint;
    utf8proc_ssize_t forward = utf8proc_iterate(
        (utf8proc_uint8_t*)&text[i],
        text.size() - i,
        (utf8proc_int32_t*)&codepoint);
    if (forward < 0)
      return UString();
    ret.append(1, codepoint);
    i += forward;
  }
  return ret;
}

static std::string _convert_from_unicode(const UString& text) {
  char dst[64];
  std::string ret;
  for (auto ch : text) {
    utf8proc_ssize_t num = utf8proc_encode_char(ch, (utf8proc_uint8_t*)dst);
    if (num <= 0)
      return "";
    ret += std::string(dst, dst + num);
  }
  return ret;
}

static void _to_lower(UString& text) {
  for (size_t i = 0; i < text.size(); i++) {
    text[i] = utf8proc_tolower(text[i]);
  }
}

BERTEncoder::BERTEncoder(const std::string& vocab_file, bool to_lower)
    : vocab_{_load_vocab_from_file(vocab_file, 1, 1)} {
  to_lower_ = to_lower;
}

BERTEncoder::BERTEncoder(Vocab vocab, bool to_lower) : vocab_{vocab} {
  to_lower_ = to_lower;
}

UString BERTEncoder::_clean(UString text) {
  /* This function combines:
      * cleaning
      * strip accents
    If we later want to add option for optional accents stripping, this require
    refactoring
  */
  size_t len = text.size();
  UString ret;
  for (size_t i = 0; i < len; i++) {
    uint32_t c = text[i];
    if (c == 0 || c == 0xFFFD || _is_control(c) ||
        utf8proc_category(c) == UTF8PROC_CATEGORY_MN) {
      continue;
    }
    if (_is_whitespace(c)) {
      ret.append(1, ' ');
    } else {
      ret.append(1, c);
    }
  }
  return ret;
}

void BERTEncoder::split_(
    std::string& str,
    std::vector<std::string>& tokens,
    const char& delimiter) {
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }
}

void BERTEncoder::_max_seg(std::string s, std::vector<std::string>& results) {
  int end = s.size();
  int start = 0;
  std::vector<std::string> sub_tokens;
  while (start < end) {
    std::string test(s.c_str() + start, end - start);

    if (start > 0) {
      test = std::string("##") + test;
    }

    if (vocab_.__contains__(test)) {
      sub_tokens.push_back(test);
      start = end;
      end = s.size();
    } else {
      end -= 1;
      if (start == end) {
        results.push_back(kUnkToken);
        return;
      }
    }
  }

  for (auto& token : sub_tokens) {
    results.push_back(token);
  }
}

UString BERTEncoder::_basic_tokenize(UString text) {
  /*
  This function enables white space based tokenization for following:
    * chinese character
    * punctuation
  */

  UString ret;
  size_t len = text.size();
  for (size_t i = 0; i < len; i++) {
    uint32_t c = text[i];
    if (_is_chinese_char(c) || _is_punct_char(c)) {
      if (!ret.empty() && ret.back() != ' ') {
        ret.append(1, ' ');
      }
      ret.append(1, c);
      ret.append(1, ' ');
    } else if (c == ' ') {
      if (!ret.empty() && ret.back() != ' ') {
        ret.append(1, c);
      }
    } else {
      ret.append(1, c);
    }
  }
  if (!ret.empty() && ret.back() == ' ') {
    ret.erase(ret.end() - 1);
  }
  return ret;
}

std::vector<std::string> BERTEncoder::Tokenize(std::string text) {
  std::vector<std::string> results;

  // normalize
  char* nfkcstr = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(text.c_str())));
  if (nfkcstr == nullptr) {
    return {};
  }

  text.assign(nfkcstr, strlen(nfkcstr));

  free(nfkcstr);

  // convert to unicode codepoints
  UString unicodes = _convert_to_unicode(text);

  // clean -> invalid character removal, whitespce cleanup, strip accents
  unicodes = _clean(unicodes);

  // Add whitespace in front/back of tokens to enable splitting based on
  // white-space Enables tokenization on chinese characters, Punctuations
  unicodes = _basic_tokenize(unicodes);

  // Convert text to lower-case
  if (to_lower_)
    _to_lower(unicodes);

  // Convert back to string from code-points
  std::string newtext = _convert_from_unicode(unicodes);

  std::vector<std::string> tokens;

  // split based on whitespace
  split_(newtext, tokens);

  // Perform WORDPIECE tokenization
  for (auto s : tokens) {
    if (s.size() > kMaxCharsPerWords) {
      results.push_back(kUnkToken);
    } else {
      _max_seg(s, results);
    }
  }
  return results;
}

std::vector<int64_t> BERTEncoder::Encode(std::string text) {
  std::vector<std::string> tokens = Tokenize(text);
  std::vector<int64_t> indices(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    indices[i] = vocab_.__getitem__(c10::string_view{tokens[i]});
  }
  return indices;
}

BERTEncoderStates _serialize_bert_encoder(
    const c10::intrusive_ptr<BERTEncoder>& self) {
  return std::make_tuple(true, self->vocab_.itos_);
}

c10::intrusive_ptr<BERTEncoder> _deserialize_bert_encoder(
    BERTEncoderStates states) {
  auto state_size = std::tuple_size<decltype(states)>::value;
  TORCH_CHECK(
      state_size == 2,
      "Expected deserialized BERTEncoder to have 2 states but found " +
          std::to_string(state_size) + " states");

  auto to_lower = std::get<0>(states);
  auto strings = std::get<1>(states);
  return c10::make_intrusive<BERTEncoder>(Vocab(std::move(strings)), to_lower);
}

} // namespace torchtext
