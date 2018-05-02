// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../neural.h"

TEST(Feedforward, Test) {

  //TODO: gradient check on each model.
  neural::Feedforward ff(2, 2, "tanh");

  Eigen::MatrixXd X_value(2, 4);
  X_value << 1, 1, 0, 0, 1, 0, 1, 0;
  autodiff::Input *X = new autodiff::Input("X", &X_value);
  autodiff::Variable *H = ff.Transform(X);
  autodiff::Variable *l = neural::average_negative_log_likelihood(H,
                                                                  {0, 1, 1, 0});
  l->ForwardBackward();

  std::cout << "W" << std::endl;
  std::cout << *ff.W() << std::endl << std::endl;
  std::cout << "b" << std::endl;
  std::cout << *ff.b() << std::endl << std::endl;
  std::cout << "X" << std::endl;
  std::cout << X->Shape() << std::endl;
  std::cout << *X->value() << std::endl << std::endl;
  std::cout << "tanh(WX + b)" << std::endl;
  std::cout << H->Shape() << std::endl;
  std::cout << *H->value() << std::endl << std::endl;

  l->DeleteThisUniqueSink();
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
