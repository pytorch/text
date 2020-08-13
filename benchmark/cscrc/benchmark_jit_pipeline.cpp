#include <torch/script.h> // One-stop header.

#include <iostream>
#include <istream>
#include <memory>
#include <string>
#include <vector>

enum class CSVState { UnquotedField, QuotedField, QuotedQuote };

std::vector<std::string> readCSVRow(const std::string &row) {
  CSVState state = CSVState::UnquotedField;
  std::vector<std::string> fields{""};
  size_t field_index = 0; // index of the current field

  for (char c : row) {
    switch (state) {
    case CSVState::UnquotedField:
      switch (c) {
      case ',': // end of field
        fields.push_back("");
        field_index++;
        break;
      case '"':
        state = CSVState::QuotedField;
        break;
      default:
        fields[field_index].push_back(c);
        break;
      }
      break;
    case CSVState::QuotedField:
      switch (c) {
      case '"':
        state = CSVState::QuotedQuote;
        break;
      default:
        fields[field_index].push_back(c);
        break;
      }
      break;
    case CSVState::QuotedQuote:
      switch (c) {
      case ',': // , after closing quote
        fields.push_back("");
        field_index++;
        state = CSVState::UnquotedField;
        break;
      case '"': // "" -> "
        fields[field_index].push_back('"');
        state = CSVState::QuotedField;
        break;
      default: // end of quote
        state = CSVState::UnquotedField;
        break;
      }
      break;
    }
  }
  return fields;
}

/// Read CSV file, Excel dialect. Accept "quoted fields ""with quotes"""
std::vector<std::vector<std::string>>
readCSV(std::string file_path,
        std::vector<std::vector<std::string>> &all_lines) {
  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  // std::vector<std::vector<std::string>> all_lines;
  std::string row;
  while (!fin.eof()) {
    std::getline(fin, row);
    if (fin.bad() || fin.fail()) {
      break;
    }
    auto fields = readCSVRow(row);
    all_lines.push_back(fields);
  }
  return all_lines;
}

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> "
                 "<path-to-train-dataset-csv>\n";
    return -1;

    // // Create a vector of inputs.
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 224, 224}));

    // // Execute the model and turn its output into a tensor.
    // at::Tensor output = module.forward(inputs).toTensor();
    // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  }

  torch::jit::script::Module module;
  std::string script_module_path = argv[1];
  std::string file_path = argv[2];

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(script_module_path);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // std::vector<std::vector<std::string>> all_lines;
  // readCSV(file_path, all_lines);

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back("test string random");

  std::cout << "[TEST 1]\n";

  // std::vector<std::string> split_tokens;
  auto split_tokens = module.forward(inputs).toListRef();

  std::cout << "[TEST 2]\n";

  for (int i = 0; i < split_tokens.size(); i++) {
    // std::cout << "[TEST 3]\n";

    // std::cout << split_tokens[i] << std::endl;
    // std::string string_token = split_tokens[i].toStringRef();
    c10::IValue ival_token = split_tokens[i];

    std::cout << "[isString] " << ival_token.isString() << std::endl;
    std::cout << "[isInt] " << ival_token.isInt() << std::endl;

    std::string str_token = ival_token.toString()->string();
    std::cout << "[print] " << str_token << std::endl;

    // std::cout << string_token << std::endl;
  }

  // for (auto ivalue_token : split_tokens) {
  //   std::cout << "[TEST 3]\n";

  //   std::cout << *ivalue_token << std::endl;
  //   std::string string_token = (*ivalue_token).toStringRef();

  //   std::cout << "[TEST 4]\n";

  //   std::cout << string_token << std::endl;
  // }

  // for (int i = 0; i < 5; i++) {
  //   for (auto it = all_lines[i].begin() + 1; it != all_lines[i].end(); it++)
  //   {
  //     std::cout << *it << std::endl;
  //     split_tokens = module.forward(*it);

  //     for (auto &token : split_tokens) {
  //       std::cout << token << std::endl;
  //     }
  //   }
  // }

  std::cout << "[DONE]\n";
}