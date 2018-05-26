// Author: Karl Stratos (me@karlstratos.com)

#include <ctime>
#include <limits>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../../core/autodiff.h"

int main (int argc, char* argv[]) {
  size_t random_seed = std::time(0);
  std::string updater = "sgd";
  size_t N = 10000;
  double step_size = 0.01;

  bool display_options_and_quit = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--seed") {
      random_seed = std::stoi(argv[++i]);
    } else if (arg == "--updater") {
      updater = argv[++i];
    } else if (arg == "--N") {
      N = std::stoi(argv[++i]);
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
    std::cout << "--N [" << N << "]:\t"
         << "number of samples" << std::endl;
    std::cout << "--step [" << step_size << "]:        \t"
         << "step size for gradient descent" << std::endl;
    std::cout << "--help, -h:           \t"
         << "show options and quit?" << std::endl;
    exit(0);
  }

  std::srand(random_seed);

  // Draw x uniformly at random from an interval.
  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  // Target linear function
  double true_slope = dis(gen);
  double true_bias = dis(gen);
  auto f = [&](double x) { return true_slope * x + true_bias; };

  // Model parameters
  autodiff::InputList inputs;
  auto w = inputs.Add("w", 1, 1, "unit-variance");
  auto b = inputs.Add("b", 1, 1, "unit-variance");

  std::unique_ptr<autodiff::Updater> gd;
  if (updater == "sgd") {
    gd = cc14::make_unique<autodiff::SimpleGradientDescent>(&inputs, step_size);
  } else if (updater == "adam") {
    gd = cc14::make_unique<autodiff::Adam>(&inputs, step_size);
  } else {
    ASSERT(false, "Unknown updater " << updater);
  }

  double loss = -std::numeric_limits<double>::infinity();
  for (size_t sample_num = 1; sample_num <= N; ++sample_num) {
    double x_value = dis(gen);
    auto y = autodiff::MakeInput({{f(x_value)}});
    auto x = autodiff::MakeInput({{x_value}});
    auto y_pred = w * x + b;
    auto l = 0.5 * squared_norm(y_pred - y);
    double new_loss = l->ForwardBackward();

    // y_pred      = wx + b
    // l(w, b)     = (1/2) (y - y_pred)^2
    // dl(w, b)/db = y_pred - y
    // dl(w, b)/dw = (y_pred - y) x
    double b_grad = (*y_pred->value())(0) - f(x_value);
    double w_grad = ((*y_pred->value())(0) - f(x_value)) * x_value;

    std::string msg = util_string::buffer_string(std::to_string(sample_num),
                                                 10, ' ', "left");
    msg += "x: " + util_string::buffer_string(
        util_string::to_string_with_precision(x_value, 2), 10, ' ', "left");
    msg += "y: " + util_string::buffer_string(
        util_string::to_string_with_precision(f(x_value), 2), 10, ' ', "left");
    msg += "y_pred: " + util_string::buffer_string(
        util_string::to_string_with_precision((*y_pred->value())(0), 2), 10,
        ' ', "left");
    msg += "w: " + util_string::buffer_string(
        util_string::to_string_with_precision((*w->value())(0), 2), 10,
        ' ', "left");
    msg += "w grad: " + util_string::buffer_string(
        util_string::to_string_with_precision(w_grad, 2)
        + " (" + util_string::to_string_with_precision(w_grad, 2) + ")", 25,
        ' ', "left");
    msg += "b: " + util_string::buffer_string(
        util_string::to_string_with_precision((*b->value())(0), 2), 10,
        ' ', "left");
    msg += "b grad: " + util_string::buffer_string(
        util_string::to_string_with_precision(b_grad, 2)
        + " (" + util_string::to_string_with_precision(b_grad, 2) + ")", 25,
        ' ', "left");
    msg += "loss: " + util_string::buffer_string(
        util_string::to_string_with_precision(new_loss, 2), 10, ' ', "left");
    std::cout << msg << std::endl;

    gd->UpdateValuesAndResetGradients();
    loss = new_loss;
  }
  std::cout << std::endl;

  std::cout << "w: " << *w->value()
            << " (vs true " << true_slope << ")" << std::endl;
  std::cout << "b: " << *b->value()
            << " (vs true " << true_bias << ")" << std::endl;
}
