#include <re2/re2.h>
#include <re2/stringpiece.h>
#include <torch/script.h>
#include <string>

namespace torchtext {
struct Regex : torch::CustomClassHolder {
 private:
  RE2* compiled_pattern_;

 public:
  std::string re_str_;

  Regex(const std::string& re_str);
  std::string Sub(std::string str, const std::string& repl) const;
  bool FindAndConsume(re2::StringPiece* input, std::string* text) const;
};

std::string _serialize_regex(const c10::intrusive_ptr<Regex>& self);
c10::intrusive_ptr<Regex> _deserialize_regex(std::string&& state);

} // namespace torchtext
