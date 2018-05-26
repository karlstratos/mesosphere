// Author: Karl Stratos (me@karlstratos.com)

#include <ctime>
#include <limits>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../../core/autodiff.h"

int main (int argc, char* argv[]) {
  size_t random_seed = std::time(0);
  std::string updater = "sgd";
  bool use_sqerr = false;
  size_t hdim = 8;
  size_t num_epochs = 2000;
  bool frozen = true;  // More difficult to learn if frozen.
  double step_size = 0.1;

  // Parse command line arguments.
  bool display_options_and_quit = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--seed") {
      random_seed = std::stoi(argv[++i]);
    } else if (arg == "--updater") {
      updater = argv[++i];
    } else if (arg == "--sqerr") {
      use_sqerr = true;
    } else if (arg == "--hdim") {
      hdim = std::stoi(argv[++i]);
    } else if (arg == "--epochs") {
      num_epochs = std::stoi(argv[++i]);
    } else if (arg == "--update") {
      frozen = false;
    } else if (arg == "--step") {
      step_size = std::stod(argv[++i]);
    } else if (arg == "--help" || arg == "-h"){
      display_options_and_quit = true;
    } else {
      std::cerr << "Invalid argument \"" << arg << "\": run the command with "
           << "-h or --help to see possible arguments." << std::endl;
      exit(-1);
    }
  }
  if (display_options_and_quit) {
    std::cout << "--seed [" << random_seed << "]:        \t"
         << "random seed" << std::endl;
    std::cout << "--updater [" << updater << "]:   \t"
         << "choice of updater" << std::endl;
    std::cout << "--sqerr [" << use_sqerr << "]:        \t"
         << "use squared error instead of cross entropy?" << std::endl;
    std::cout << "--hdim [" << hdim << "]:        \t"
         << "dimension of feedforward output vector" << std::endl;
    std::cout << "--update:         \t"
         << "update the input representation?" << std::endl;
    std::cout << "--epochs [" << num_epochs << "]:\t"
         << "number of epochs" << std::endl;
    std::cout << "--step [" << step_size << "]:        \t"
         << "step size for gradient descent" << std::endl;
    std::cout << "--help, -h:           \t"
         << "show options and quit?" << std::endl;
    exit(0);
  }

  std::srand(random_seed);

  autodiff::InputList inputs;
  auto x11 = inputs.Add("x11", {{1}, {1}}, frozen);
  auto x10 = inputs.Add("x10", {{1}, {0}}, frozen);
  auto x01 = inputs.Add("x01", {{0}, {1}}, frozen);
  auto x00 = inputs.Add("x00", {{0}, {0}}, frozen);
  auto Y = inputs.Add("Y", {{0, 1, 1, 0}}, true);  // Frozen label for l2

  auto W1 = inputs.Add("W1", hdim, 2, "unit-variance");
  auto b1 = inputs.Add("b1", hdim, 1, "unit-variance");
  auto W2 = inputs.Add("W2", 1, hdim, "unit-variance");
  auto b2 = inputs.Add("b2", 1, 1, "unit-variance");

  std::unique_ptr<autodiff::Updater> gd;
  if (updater == "sgd") {
    gd = cc14::make_unique<autodiff::SimpleGradientDescent>(&inputs, step_size);
  } else if (updater == "adam") {
    gd = cc14::make_unique<autodiff::Adam>(&inputs, step_size);
  } else {
    ASSERT(false, "Unknown updater " << updater);
  }

  double loss = -std::numeric_limits<double>::infinity();
  for (size_t epoch_num = 1; epoch_num <= num_epochs; ++epoch_num) {
    auto H = W2 * tanh(W1 * (x11 ^ x10 ^ x01 ^ x00) + b1) + b2;
    auto l = (use_sqerr) ?
             average(squared_norm(H - Y)) :
             average(binary_cross_entropy(H, {false, true, true, false}));
    double new_loss = l->ForwardBackward();
    std::cout << "epoch: " << epoch_num << "     "
              << "step size: " << gd->step_size() << "     "
              << "loss: " << new_loss << std::endl;
    gd->UpdateValuesAndResetGradients();
    loss = new_loss;
  }
  std::cout << std::endl;

  auto H = W2 * tanh(W1 * (x11 ^ x10 ^ x01 ^ x00) + b1) + b2;
  auto P = (use_sqerr) ? H : logistic(H);
  Eigen::MatrixXd P_value = P->Forward();

  std::cout << "x11 value: " << x11->value()->transpose()
            << " -> " << P_value(0) << std::endl;
  std::cout << "x10 value: " << x10->value()->transpose()
            << " -> " << P_value(1) << std::endl;
  std::cout << "x01 value: " << x01->value()->transpose()
            << " -> " << P_value(2) << std::endl;
  std::cout << "x00 value: " << x00->value()->transpose()
            << " -> " << P_value(3) << std::endl;
  if (use_sqerr) { std::cout << "Y value: " << *Y->value() << std::endl; }
}
