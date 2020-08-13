#include <torch/script.h> // One-stop header.

#include <chrono>
#include <ctime>
#include <iostream>
#include <istream>
#include <memory>
#include <ratio>
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

void pipeline_benchmark(
    torch::jit::script::Module &module,
    std::vector<std::vector<torch::jit::IValue>> all_lines_ivalue) {

  for (auto &line : all_lines_ivalue) {
    auto token_list = module.forward(line).toList();

    for (size_t i = 0; i < token_list.size(); i++) {
      c10::IValue token_ref = token_list.get(i);
      std::string token = token_ref.toStringRef();
      // std::cout << token << "\n";
    }
  }
}

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> "
                 "<path-to-train-dataset-csv>\n";
    return -1;
  }

  torch::jit::script::Module module, vocab;
  std::string tokenizer_path = argv[1];
  // std::string vocab_path = argv[2];
  std::string file_path = argv[2];

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(tokenizer_path);
    // vocab = torch::jit::load(tokenizer_path);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<std::vector<std::string>> all_lines;
  readCSV(file_path, all_lines);

  // Create a vector of inputs.
  std::vector<std::vector<torch::jit::IValue>> all_lines_ivalue;
  // for (int i = 0; i < 1; i++) {
  //   auto &lines = all_lines[i];
  for (auto &lines : all_lines) {

    std::string concat_lines = "";
    // skip first item since its the label
    for (auto line_it = lines.begin() + 1; line_it != lines.end(); line_it++) {
      concat_lines += " " + *line_it;
    }
    all_lines_ivalue.push_back(
        std::vector<torch::jit::IValue>{c10::IValue(concat_lines)});
  }

  std::cout << "benchmarking jit pipeline\n";

  auto start = std::chrono::steady_clock::now();

  pipeline_benchmark(module, all_lines_ivalue);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "pipeline_benchmark elapsed time: " << elapsed_seconds.count()
            << "s\n";

  std::cout << "[DONE]\n";
}