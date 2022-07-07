
#include <torch/script.h>
#include <torch/nn/functional/activation.h>

#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, const char* argv[]) {
    std::cout << "Loading model...\n";

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        return -1;
    }

    torch::NoGradGuard no_grad; // ensures that autograd is off
    torch::jit::IValue tokens_ivalue = module.forward(std::vector<c10::IValue>(1,"The green grasshopper jumped over the fence"));
    std::cout << "Result: " << tokens_ivalue << std::endl;

    return 0;
}
