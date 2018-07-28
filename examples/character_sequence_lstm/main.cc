// Author: Karl Stratos (me@karlstratos.com)

#include <ctime>
#include <limits>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../../core/neural.h"

int main (int argc, char* argv[]) {
  size_t seed = 42;
  size_t dim = 50;
  bool tie = false;
  double step_size = 0.01;
  size_t num_epochs = 200;
  std::string characters = "abcdefghijklmnopqrstuvwxyz *";

  bool display_options_and_quit = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--tie") {
      tie = true;
    } else if (arg == "--dim") {
      dim = std::stoi(argv[++i]);
    } else if (arg == "--epochs") {
      num_epochs = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h"){
      display_options_and_quit = true;
    } else {
      std::cerr << "Invalid argument \"" << arg << "\": run the command with "
                << "-h or --help to see possible arguments." << std::endl;
      exit(-1);
    }
  }
  if (display_options_and_quit) {
    std::cout << "--tie    \t"
              << "tie lookup matrix with softmax weights?" << std::endl;
    std::cout << "--dim [" << dim << "]: \t"
              << "dimension of character/state vectors" << std::endl;
    std::cout << "--epochs [" << num_epochs << "]:\t"
              << "number of epochs" << std::endl;
    std::cout << "--help, -h: \t"
              << "show options and quit?" << std::endl;
    exit(0);
  }
  std::srand(seed);

  std::string sentence;
  std::cout << "type character sequence (default: \"a quick brown fox jumped "
            << "over the lazy dog\")" << std::endl;
  std::getline(std::cin, sentence);
  if (sentence.size() == 0) {
    sentence = "a quick brown fox jumped over the lazy dog";
  }

  neural::Model model;
  std::unordered_map<char, size_t> c2i;
  for (size_t i = 0; i < characters.size(); ++i) { c2i[characters[i]] = i; }
  auto i2c = util_misc::invert(c2i);

  size_t i_C = model.AddWeight(dim, c2i.size(), "unit-variance");
  neural::LSTM lstm(1, dim, dim, &model);
  size_t i_W = (tie) ? 0 : model.AddWeight(c2i.size(), dim, "unit-variance");

  auto get_loss = [&]() {
    auto U = (tie) ? transpose(model.MakeInput(i_C)) : model.MakeInput(i_W);
    auto h = lstm.ComputeNewStateStack(model.MakeInputColumn(i_C, c2i['*']));
    std::vector<std::shared_ptr<neural::Variable>> ls;
    for (char c : sentence + '*') {
      auto l = cross_entropy(U * h.back()[0], {c2i[c]});
      ls.push_back(l);
      h = lstm.ComputeNewStateStack(model.MakeInputColumn(i_C, c2i[c]), h);
    }
    return sum(ls);

  };

  std::mt19937 gen(seed);

  auto generate = [&]() {
    auto U = (tie) ? transpose(model.MakeInput(i_C)) : model.MakeInput(i_W);
    auto h = lstm.ComputeNewStateStack(model.MakeInputColumn(i_C, c2i['*']));
    std::string output;
    while (true) {
      auto p = softmax(U * h.back()[0])->Forward();
      std::vector<double> probabilities(p.data(), p.data() + p.rows());
      char new_c = i2c[util_misc::sample(probabilities, &gen)];
      if (new_c == '*') { break; }
      output.push_back(new_c);
      h = lstm.ComputeNewStateStack(model.MakeInputColumn(i_C, c2i[new_c]), h);
    }
    model.ClearComputation();
    return output;
  };

  neural::Adam gd(&model, step_size);
  for (size_t i = 1; i < num_epochs; ++i) {
    auto l = get_loss();
    double loss = l->ForwardBackward();
    gd.UpdateWeights();
    if (i % 5 == 0) {
      std::cout << util_string::printf_format("%.2f", loss) << "\t"
                << generate() << std::endl;
    }
  }
}
