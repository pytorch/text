#include <re2/re2.h>
#include <re2/stringpiece.h>
#include <torch/script.h>
#include <torchtext/csrc/export.h>
#include <string>

namespace torchtext {
struct Regex : torch::CustomClassHolder {
 private:
  RE2* compiled_pattern_;

 public:
  std::string re_str_;

  TORCHTEXT_API Regex(const std::string& re_str);
  TORCHTEXT_API ~Regex();
  TORCHTEXT_API std::string Sub(std::string str, const std::string& repl) const;
  TORCHTEXT_API bool FindAndConsume(re2::StringPiece* input, std::string* text)
      const;
};

TORCHTEXT_API std::string _serialize_regex(
    const c10::intrusive_ptr<Regex>& self);
TORCHTEXT_API c10::intrusive_ptr<Regex> _deserialize_regex(std::string&& state);

} // namespace torchtext
