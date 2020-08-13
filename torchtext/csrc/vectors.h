#include <torch/script.h>

namespace torchtext {

typedef std::vector<std::string> StringList;

// order_preserving_flat_hash_map is buggy on Windows
#ifdef _MSC_VER
typedef std::unordered_map<std::string, torch::Tensor> VectorsMap;
typedef std::unordered_map<std::string, int64_t> IndexMap;
#else
typedef ska_ordered::order_preserving_flat_hash_map<std::string, torch::Tensor>
    VectorsMap;
typedef ska_ordered::order_preserving_flat_hash_map<std::string, int64_t>
    IndexMap;
#endif

typedef std::tuple<std::string, std::vector<int64_t>, std::vector<std::string>,
                   std::vector<torch::Tensor>>
    VectorsStates;

struct Vectors : torch::CustomClassHolder {
public:
  const std::string version_str_ = "0.0.1";
  IndexMap stoi_;
  VectorsMap stovec_;
  torch::Tensor vectors_;
  torch::Tensor unk_tensor_;

  explicit Vectors(const IndexMap &stoi, const torch::Tensor vectors,
                   const torch::Tensor &unk_tensor);
  explicit Vectors(const std::vector<std::string> &tokens,
                   const std::vector<std::int64_t> &indices,
                   const torch::Tensor &vectors,
                   const torch::Tensor &unk_tensor);
  std::unordered_map<std::string, int64_t> get_stoi();
  torch::Tensor __getitem__(const std::string &token);
  torch::Tensor lookup_vectors(const std::vector<std::string> &tokens);
  void __setitem__(const std::string &token, const torch::Tensor &vector);
  int64_t __len__();
};

c10::intrusive_ptr<Vectors> _get_vectors_from_states(VectorsStates states);
VectorsStates _set_vectors_states(const c10::intrusive_ptr<Vectors> &self);

std::tuple<Vectors, std::vector<std::string>> _load_token_and_vectors_from_file(
    const std::string &file_path, const std::string delimiter_str,
    const int64_t num_cpus, c10::optional<torch::Tensor> opt_unk_tensor);

} // namespace torchtext
