#include <string>
#include <torch/script.h>

namespace torchtext {
struct Regex : torch::CustomClassHolder {
public:
  // re_str_ holds the serialized regex string passed at the initialization.
  // We need this because we need to be able to serialize the model so that we
  // can save the scripted object. pickle will get the
  // serialized model from this re_str_ member, thus it needs to be public.
  std::string re_str_;

  Regex(const std::string &re_str);
  std::string Sub(const std::string &str, const std::string &repl) const;
};
}