// Author: Karl Stratos (me@karlstratos.com)
//
// ./main 42 30 0 10

#include <limits>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../../core/neural.h"

int main (int argc, char* argv[]) {
  size_t random_seed = std::stoi(argv[1]);
  size_t num_epochs = std::stoi(argv[2]);
  bool frozen = std::stoi(argv[3]);  // If frozen, difficult to learn.
  double step_size = std::stod(argv[4]);

  std::srand(random_seed);

  Eigen::MatrixXd X_value(2, 4);
  X_value <<
      1, 1, 0, 0,
      1, 0, 1, 0;
  std::vector<autodiff::Input *> inputs;
  autodiff::Input *X = new autodiff::Input("X", &X_value, &inputs, frozen);
  neural::Feedforward ff(2, 2, "tanh", &inputs);
  neural::Feedforward ff2(2, 2, "identity", &inputs);
  std::cout << "Initial X value" << std::endl;
  std::cout << X_value << std::endl << std::endl;
  std::cout << "Initial W1 value" << std::endl;
  std::cout << *ff.W()->value() << std::endl << std::endl;
  std::cout << "Initial b1 value" << std::endl;
  std::cout << *ff.b()->value() << std::endl << std::endl;
  std::cout << "Initial W2 value" << std::endl;
  std::cout << *ff2.W()->value() << std::endl << std::endl;
  std::cout << "Initial b2 value" << std::endl;
  std::cout << *ff2.b()->value() << std::endl << std::endl;

  autodiff::SimpleUpdater gd(inputs, step_size);

  double loss = -std::numeric_limits<double>::infinity();
  for (size_t epoch_num = 1; epoch_num <= num_epochs; ++epoch_num) {
    autodiff::Variable *Z = ff.Transform(X);
    autodiff::Variable *H = ff2.Transform(Z);
    autodiff::Variable *l =
        neural::average_negative_log_likelihood(H, {0, 1, 1, 0});
    l->ForwardBackward();
    double new_loss = (*l->value())(0);
    std::cout << "epoch: " << epoch_num << "     "
              << "step size: " << gd.step_size() << "     "
              << "loss: " << new_loss << std::endl;
    gd.Update();
    loss = new_loss;
    l->DeleteThisUniqueSinkExceptRoots();
  }
  std::cout << std::endl;

  std::cout << "X value" << std::endl;
  std::cout << X_value << std::endl << std::endl;
  std::cout << "W1 value" << std::endl;
  std::cout << *ff.W()->value() << std::endl << std::endl;
  std::cout << "b1 value" << std::endl;
  std::cout << *ff.b()->value() << std::endl << std::endl;
  std::cout << "W2 value" << std::endl;
  std::cout << *ff2.W()->value() << std::endl << std::endl;
  std::cout << "b2 value" << std::endl;
  std::cout << *ff2.b()->value() << std::endl << std::endl;

  autodiff::Variable *Z = ff.Transform(X);
  autodiff::Variable *H = ff2.Transform(Z);
  autodiff::Variable *P = new autodiff::Softmax(H);
  Eigen::MatrixXd Y_pred = P->Forward();
  std::cout << "estimated distributions" << std::endl;
  std::cout << Y_pred << std::endl << std::endl;
  std::cout << "true distributions" << std::endl;
  Eigen::MatrixXd Y_value(2, 4);
  Y_value <<
      1, 0, 0, 1,
      0, 1, 1, 0;
  std::cout << Y_value << std::endl << std::endl;

  autodiff::Variable *sink = new autodiff::ReduceSum(P);
  sink->DeleteThisUniqueSink();
}
