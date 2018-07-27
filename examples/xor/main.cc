// Author: Karl Stratos (me@karlstratos.com)

#include <ctime>
#include <limits>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../../core/neural.h"

int main (int argc, char* argv[]) {
  size_t random_seed = std::time(0);
  std::string updater = "adam";
  bool use_sqerr = false;
  size_t hdim = 8;
  size_t num_epochs = 2000;
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
    std::cout << "--epochs [" << num_epochs << "]:\t"
              << "number of epochs" << std::endl;
    std::cout << "--step [" << step_size << "]:        \t"
              << "step size for gradient descent" << std::endl;
    std::cout << "--help, -h:           \t"
              << "show options and quit?" << std::endl;
    exit(0);
  }

  std::srand(random_seed);

  // Model parameters
  neural::Model model;
  auto i_W1 = model.AddWeight(hdim, 2, "unit-variance");
  auto i_b1 = model.AddWeight(hdim, 1, "unit-variance");
  auto i_W2 = model.AddWeight(1, hdim, "unit-variance");
  auto i_b2 = model.AddWeight(1, 1, "unit-variance");

  std::unique_ptr<neural::Updater> gd;
  if (updater == "sgd") {
    gd = cc14::make_unique<neural::SimpleGradientDescent>(&model, step_size);
  } else if (updater == "adam") {
    gd = cc14::make_unique<neural::Adam>(&model, step_size);
  } else {
    ASSERT(false, "Unknown updater " << updater);
  }

  auto draw_XY = [&]() {
    size_t i_X = model.AddTemporaryWeight({{1, 1, 0, 0}, {1, 0, 1, 0}});
    size_t i_Y = model.AddTemporaryWeight({{0, 1, 1, 0}});
    auto X = model.MakeTemporaryInput(i_X);
    auto Y = model.MakeTemporaryInput(i_Y);
    return std::make_pair(X, Y);
  };

  double loss = -std::numeric_limits<double>::infinity();
  for (size_t epoch_num = 1; epoch_num <= num_epochs; ++epoch_num) {
    auto training_data = draw_XY();
    auto X = training_data.first;
    auto Y = training_data.second;

    // Compute loss with current model.
    auto W1 = model.MakeInput(i_W1);
    auto b1 = model.MakeInput(i_b1);
    auto W2 = model.MakeInput(i_W2);
    auto b2 = model.MakeInput(i_b2);
    auto H = W2 * tanh(W1 * X + b1) + b2;
    auto l = (use_sqerr) ?
             average(squared_norm(H - Y)) :
             average(binary_cross_entropy(H, {false, true, true, false}));
    double new_loss = l->ForwardBackward();
    std::cout << "epoch: " << epoch_num << "     "
              << "step size: " << gd->step_size() << "     "
              << "loss: " << new_loss << std::endl;
    gd->UpdateWeights();
    loss = new_loss;
  }
  std::cout << std::endl;

  auto training_data = draw_XY();
  auto X = training_data.first;
  auto W1 = model.MakeInput(i_W1);
  auto b1 = model.MakeInput(i_b1);
  auto W2 = model.MakeInput(i_W2);
  auto b2 = model.MakeInput(i_b2);
  auto H = W2 * tanh(W1 * X + b1) + b2;
  auto P = (use_sqerr) ? H : logistic(H);
  Eigen::MatrixXd P_value = P->Forward();

  std::cout << "(1, 1) -> " << P_value(0) << std::endl;
  std::cout << "(1, 0) -> " << P_value(1) << std::endl;
  std::cout << "(0, 1) -> " << P_value(2) << std::endl;
  std::cout << "(0, 0) -> " << P_value(3) << std::endl;
}
