// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../autodiff.h"

TEST(Add, Test0) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1}});
  auto y = inputs.Add("y", {{2}});
  auto z = x + y;
  double result = z->ForwardBackward();

  EXPECT_EQ(3, result);
  EXPECT_EQ(1, (*x->gradient())(0));
  EXPECT_EQ(1, (*y->gradient())(0));
}

TEST(Add, Test1) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1}});
  auto y = inputs.Add("y", {{2}});
  auto z = x + y;
  auto q = z + x;
  auto l = q + q;
  auto o = l + (y + (y + y));
  double result = o->ForwardBackward();

  EXPECT_EQ(14, result);
  EXPECT_EQ(4, (*x->gradient())(0));
  EXPECT_EQ(5, (*y->gradient())(0));

  EXPECT_EQ(2, inputs.Size());
  EXPECT_EQ(4, (*inputs(0)->gradient())(0));
  EXPECT_EQ(5, (*inputs(1)->gradient())(0));
}

TEST(Add, Test2) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1}});
  auto u3 = (x + (x + x)) + x;
  auto y = u3 + (x + (x + x));
  auto t = y + u3;
  auto z = t + y + u3;
  double result = z->ForwardBackward();
  EXPECT_EQ(22, result);
  EXPECT_EQ(22, (*x->gradient())(0));
}

TEST(Add, MatrixVector) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1, 2, 3}});
  auto y = inputs.Add("y", {{-1}});
  auto z = x + y;
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_EQ(3, result);
  for (size_t i = 0; i < x->NumColumns(); ++i) {
    EXPECT_EQ(1, (*x->gradient())(i));
  }
  EXPECT_EQ(3, (*y->gradient())(0));
}

TEST(ScalarVariableAddSubtractMultiplyMix, Test) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1}});
  auto y = x + 1;             //     y = x + 1  = 2
  auto z = 1 + 2 * x - 2;     //     z = 2x - 1 = 1
  auto q = y - z;             //     q = -x + 2 = 1
  auto t = z - y;             //     t = x - 2  = -1
  auto l = q * t;             //     l = -x^2 + 4x -4 = -1
  double result = l->ForwardBackward();

  EXPECT_EQ(-1, result);
  EXPECT_EQ(2, (*x->gradient())(0));  // -2x + 4
}

TEST(ReduceSum, Test) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1, 2}, {3, 4}, {5, 6}});
  auto z = sum(x);
  double result = z->ForwardBackward();

  EXPECT_EQ(21, result);
  for (size_t i = 0; i < x->NumRows(); ++i) {
    for (size_t j = 0; j < x->NumColumns(); ++j) {
      EXPECT_EQ(1, (*x->gradient())(i, j));
    }
  }
}

TEST(Multiply, Test) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{2}});
  auto y = inputs.Add("y", {{3}});
  auto l = x * (y * x) * y;
  auto q = x * l * l * x;  // x^6 y^4
  double result = q->ForwardBackward();

  EXPECT_EQ(5184, result);
  EXPECT_EQ(15552, (*x->gradient())(0));  // dq/dx = 6 x^5 y^4
  EXPECT_EQ(6912, (*y->gradient())(0));  // dq/dy = 4 x^6 y^3
}

TEST(Dot, Test) {
  autodiff::InputList inputs;
  auto X = inputs.Add("X", {{1, 3}, {2, 4}});
  auto Y = inputs.Add("Y", {{5, 7}, {6, 8}});
  auto Z = dot(X, Y);
  auto l = sum(Z);
  l->ForwardBackward();

  EXPECT_EQ(17, (*Z->value())(0));
  EXPECT_EQ(53, (*Z->value())(1));

  EXPECT_EQ(5, (*X->gradient())(0, 0));
  EXPECT_EQ(6, (*X->gradient())(1, 0));

  EXPECT_EQ(7, (*X->gradient())(0, 1));
  EXPECT_EQ(8, (*X->gradient())(1, 1));

  EXPECT_EQ(1, (*Y->gradient())(0, 0));
  EXPECT_EQ(2, (*Y->gradient())(1, 0));

  EXPECT_EQ(3, (*Y->gradient())(0, 1));
  EXPECT_EQ(4, (*Y->gradient())(1, 1));
}

TEST(AddMultiplyDotFlipSign, Test) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1}, {2}});
  auto y = inputs.Add("y", {{3}, {4}});
  auto p = -dot(x + y, 2 * y);  // -2(x'y - y'y)
  double result = p->ForwardBackward();

  EXPECT_EQ(-72, result);

  // Dx = 2y = [6; 8]
  EXPECT_EQ(-6, (*x->gradient())(0));
  EXPECT_EQ(-8, (*x->gradient())(1));

  // Dy = 2x + 4y = [14; 20]
  EXPECT_EQ(-14, (*y->gradient())(0));
  EXPECT_EQ(-20, (*y->gradient())(1));
}

TEST(Logistic, Test) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1}});
  auto z = logistic(x / 0.5);
  double result = z->ForwardBackward();

  EXPECT_NEAR(0.8808, result, 1e-4);
  EXPECT_NEAR(0.2100, (*x->gradient())(0), 1e-4);
}

TEST(Tanh, Test) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1}});
  auto z = tanh(2 * x);
  double result = z->ForwardBackward();

  EXPECT_NEAR(0.9640, result, 1e-4);
  EXPECT_NEAR(0.1413, (*x->gradient())(0), 1e-4);
}

TEST(SoftmaxPick, Test) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{1, 1}, {2, 2}, {3, 3}});
  auto y = softmax(x);
  auto z = pick(y, {1, 2});
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_NEAR(0.9100, result, 1e-4);
  EXPECT_NEAR(0.2447, (*z->value())(0), 1e-4);
  EXPECT_NEAR(0.6652, (*z->value())(1), 1e-4);
  EXPECT_NEAR(0.0900, (*y->value())(0), 1e-4);
  EXPECT_NEAR(0.2447, (*y->value())(1), 1e-4);
  EXPECT_NEAR(0.6652, (*y->value())(2), 1e-4);
  EXPECT_NEAR(-0.0220, (*x->gradient())(0, 0), 1e-4);
  EXPECT_NEAR(0.1848, (*x->gradient())(1, 0), 1e-4);
  EXPECT_NEAR(-0.1628, (*x->gradient())(2, 0), 1e-4);
  EXPECT_NEAR(-0.0599, (*x->gradient())(0, 1), 1e-4);
  EXPECT_NEAR(-0.1628, (*x->gradient())(1, 1), 1e-4);
  EXPECT_NEAR(0.2227, (*x->gradient())(2, 1), 1e-4);
}

TEST(PickNegativeLogSoftmax, Test) {
  autodiff::InputList inputs;
  auto x1 = inputs.Add("x1", {{1}});
  auto x2 = inputs.Add("x2", {{2}});
  auto x3 = inputs.Add("x3", {{3}});
  auto x4 = (x1 & x2 & x1) ^ (x1 & x2 & x1);
  auto x5 = (x1 & x1 & x3) ^ (x1 & x1 & x3);
  auto x = x4 % x5;  // x = [1 1; 2 2; 3 3]
  auto l = sum(cross_entropy(x, {1, 2}));
  l->ForwardBackward();

  EXPECT_NEAR(0.0900, (*x->gradient())(0, 0), 1e-4);  // p(0)
  EXPECT_NEAR(-0.7553, (*x->gradient())(1, 0), 1e-4); // p(1) - 1
  EXPECT_NEAR(0.6652, (*x->gradient())(2, 0), 1e-4);  // p(2)

  EXPECT_NEAR(0.0900, (*x->gradient())(0, 1), 1e-4);  // p(0)
  EXPECT_NEAR(0.2447, (*x->gradient())(1, 1), 1e-4);  // p(1)
  EXPECT_NEAR(-0.3348, (*x->gradient())(2, 1), 1e-4); // p(2) - 1
}

TEST(FlagNegativeLogistic, Test) {
  autodiff::InputList inputs;
  auto x = inputs.Add("x", {{-1, 2}});
  auto z = binary_cross_entropy(x, {false, true});
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_NEAR(0.4402, result, 1e-4);  // -log p(F1) - log p(T2)
  EXPECT_NEAR(0.3133, (*z->value())(0), 1e-4);  // -log p(F1)
  EXPECT_NEAR(0.1269, (*z->value())(1), 1e-4);  // -log p(T2)
  EXPECT_NEAR(0.2689, (*x->gradient())(0), 1e-4);  //  p(T1)
  EXPECT_NEAR(-0.1192, (*x->gradient())(1), 1e-4); // -p(F2)
}

TEST(GradientCheck, Test) {
  std::srand(std::time(0));
  double epsilon = 1e-4;
  size_t num_examples = 5;
  size_t dim_input = 10;
  size_t num_labels = 3;

  std::vector<size_t> labels;
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dis(0, num_labels - 1);
  for (size_t i = 0; i < num_examples; ++i) { labels.push_back(dis(gen)); }

  autodiff::InputList inputs;
  auto X = inputs.Add("X", dim_input, num_examples, "unit-variance");

  auto W1 = inputs.Add("W1", num_labels, dim_input, "unit-variance");
  auto b1 = inputs.Add("b1", num_labels, 1, "unit-variance");
  auto W2 = inputs.Add("W2", num_labels, num_labels, "unit-variance");
  auto b2 = inputs.Add("b2", num_labels, 1, "unit-variance");
  auto W3 = inputs.Add("W3", num_labels, num_labels, "unit-variance");
  auto b3 = inputs.Add("b3", num_labels, 1, "unit-variance");
  auto W4 = inputs.Add("W4", num_labels, num_labels, "unit-variance");
  auto b4 = inputs.Add("b4", num_labels, 1, "unit-variance");
  auto u = inputs.Add("u", num_labels, num_examples, "unit-variance");

  auto H = logistic(W4 * (W3 * relu(W2 * tanh(W1 * X + b1) + b2) + b3)
                    + b4) % u / 10.0;
  auto l = average(cross_entropy(H, labels));
  double l_original = l->ForwardBackward();
  Eigen::MatrixXd X_grad = *X->gradient();

  for (size_t i = 0; i < X_grad.rows(); ++i) {
    for (size_t j = 0; j < X_grad.cols(); ++j) {
      inputs.ResetGradient();
      (*X->value())(i, j) += epsilon;

      auto H = logistic(W4 * (W3 * relu(W2 * tanh(W1 * X + b1) + b2) + b3)
                        + b4) % u / 10.0;
      auto l = average(cross_entropy(H, labels));
      double l_value = l->ForwardBackward();
      EXPECT_NEAR((l_value - l_original) / epsilon, X_grad(i, j), 1e-5);
      (*X->value())(i, j) -= epsilon;
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
