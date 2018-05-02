// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../neural.h"

TEST(Feedforward, Test) {
  double epsilon = 1e-4;
  size_t num_examples = 5;
  size_t dim_input = 10;
  size_t dim_output = 3;

  Eigen::MatrixXd X_value = Eigen::MatrixXd::Random(dim_input,
                                                    num_examples);
  std::vector<size_t> labels;
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dis(0, dim_output - 1);
  for (size_t i = 0; i < num_examples; ++i) { labels.push_back(dis(gen)); }

  std::vector<autodiff::Input *> inputs;
  autodiff::Input *X = new autodiff::Input("X", &X_value, &inputs);
  neural::Feedforward ff1(dim_output, dim_input, "tanh", &inputs);
  neural::Feedforward ff2(dim_output, dim_output, "relu", &inputs);
  neural::Feedforward ff3(dim_output, dim_output, "identity", &inputs);
  neural::Feedforward ff4(dim_output, dim_output, "logistic", &inputs);

  autodiff::Variable *H1 = ff1.Transform(X);
  autodiff::Variable *H2 = ff2.Transform(H1);
  autodiff::Variable *H3 = ff3.Transform(H2);
  autodiff::Variable *H4 = ff4.Transform(H3);
  autodiff::Variable *l =
      neural::average_negative_log_likelihood(H4, labels);
  l->ForwardBackward();

  double l_original = (*l->value())(0);
  Eigen::MatrixXd X_grad = *X->gradient();


  for (size_t i = 0; i < X_grad.rows(); ++i) {
    for (size_t j = 0; j < X_grad.cols(); ++j) {
      l->DeleteThisUniqueSinkExceptRoots();
      for (autodiff::Input *input: inputs) { input->reset_gradient(); }

      X_value(i, j) += epsilon;

      H1 = ff1.Transform(X);
      H2 = ff2.Transform(H1);
      H3 = ff3.Transform(H2);
      H4 = ff4.Transform(H3);
      l = neural::average_negative_log_likelihood(H4, labels);
      l->ForwardBackward();
      EXPECT_NEAR(((*l->value())(0) - l_original) / epsilon,
                  X_grad(i, j), 1e-5);

      X_value(i, j) -= epsilon;
    }
  }

  l->DeleteThisUniqueSink();
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
