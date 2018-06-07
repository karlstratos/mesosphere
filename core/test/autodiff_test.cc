// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../autodiff.h"

TEST(Add, Test0) {
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto z = x + y;
  double result = z->ForwardBackward();

  EXPECT_EQ(3, result);
  EXPECT_EQ(1, (*x->gradient())(0));
  EXPECT_EQ(1, (*y->gradient())(0));
}

TEST(Add, Test1) {
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto z = x + y;
  auto q = z + x;
  auto l = q + q;
  auto o = l + (y + (y + y));
  double result = o->ForwardBackward();

  EXPECT_EQ(14, result);
  EXPECT_EQ(4, (*x->gradient())(0));
  EXPECT_EQ(5, (*y->gradient())(0));

  EXPECT_EQ(2, model.NumWeights());
  EXPECT_EQ(4, (*model.gradient(0))(0));
  EXPECT_EQ(5, (*model.gradient(1))(0));
}

TEST(Add, Test2) {
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1}});

  auto x = model.MakeInput(i_x);
  auto u3 = (x + (x + x)) + x;
  auto y = u3 + (x + (x + x));
  auto t = y + u3;
  auto z = t + y + u3;
  double result = z->ForwardBackward();
  EXPECT_EQ(22, result);
  EXPECT_EQ(22, (*x->gradient())(0));
}

TEST(Add, MatrixVector) {
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1, 2, 3}});
  size_t i_y = model.AddWeight({{-1}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
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
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1}});
  auto x = model.MakeInput(i_x);
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
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1, 2}, {3, 4}, {5, 6}});
  auto x = model.MakeInput(i_x);
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
  autodiff::Model model;
  size_t i_x = model.AddWeight({{2}});
  size_t i_y = model.AddWeight({{3}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto l = x * (y * x) * y;
  auto q = x * l * l * x;  // x^6 y^4
  double result = q->ForwardBackward();

  EXPECT_EQ(5184, result);
  EXPECT_EQ(15552, (*x->gradient())(0));  // dq/dx = 6 x^5 y^4
  EXPECT_EQ(6912, (*y->gradient())(0));  // dq/dy = 4 x^6 y^3
}

TEST(Dot, Test) {
  autodiff::Model model;
  size_t i_X = model.AddWeight({{1, 3}, {2, 4}});
  size_t i_Y = model.AddWeight({{5, 7}, {6, 8}});

  auto X = model.MakeInput(i_X);
  auto Y = model.MakeInput(i_Y);
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
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1}, {2}});
  size_t i_y = model.AddWeight({{3}, {4}});
  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
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
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1}});
  auto x = model.MakeInput(i_x);
  auto z = logistic(x / 0.5);
  double result = z->ForwardBackward();

  EXPECT_NEAR(0.8808, result, 1e-4);
  EXPECT_NEAR(0.2100, (*x->gradient())(0), 1e-4);
}

TEST(Tanh, Test) {
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1}});
  auto x = model.MakeInput(i_x);
  auto z = tanh(2 * x);
  double result = z->ForwardBackward();

  EXPECT_NEAR(0.9640, result, 1e-4);
  EXPECT_NEAR(0.1413, (*x->gradient())(0), 1e-4);
}

TEST(SoftmaxPick, Test) {
  autodiff::Model model;
  size_t i_x = model.AddWeight({{1, 1}, {2, 2}, {3, 3}});
  auto x = model.MakeInput(i_x);
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
  autodiff::Model model;
  size_t i_x1 = model.AddWeight({{1}});
  size_t i_x2 = model.AddWeight({{2}});
  size_t i_x3 = model.AddWeight({{3}});
  auto x1 = model.MakeInput(i_x1);
  auto x2 = model.MakeInput(i_x2);
  auto x3 = model.MakeInput(i_x3);
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
  autodiff::Model model;
  size_t i_x = model.AddWeight({{-1, 2}});

  auto x = model.MakeInput(i_x);
  auto z = binary_cross_entropy(x, {false, true});
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_NEAR(0.4402, result, 1e-4);  // -log p(F1) - log p(T2)
  EXPECT_NEAR(0.3133, (*z->value())(0), 1e-4);  // -log p(F1)
  EXPECT_NEAR(0.1269, (*z->value())(1), 1e-4);  // -log p(T2)
  EXPECT_NEAR(0.2689, (*x->gradient())(0), 1e-4);  //  p(T1)
  EXPECT_NEAR(-0.1192, (*x->gradient())(1), 1e-4); // -p(F2)
}

TEST(Feedforward, GradientCheck) {
  std::srand(std::time(0));
  double epsilon = 1e-4;
  size_t num_examples = 5;
  size_t dim_input = 10;
  size_t num_labels = 3;

  std::vector<size_t> labels;
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dis(0, num_labels - 1);
  for (size_t i = 0; i < num_examples; ++i) { labels.push_back(dis(gen)); }

  autodiff::Model model;
  size_t i_X = model.AddWeight(dim_input, num_examples, "unit-variance");
  size_t i_W1 = model.AddWeight(num_labels, dim_input, "unit-variance");
  size_t i_b1 = model.AddWeight(num_labels, 1, "unit-variance");
  size_t i_W2 = model.AddWeight(num_labels, num_labels, "unit-variance");
  size_t i_b2 = model.AddWeight(num_labels, 1, "unit-variance");
  size_t i_W3 = model.AddWeight(num_labels, num_labels, "unit-variance");
  size_t i_b3 = model.AddWeight(num_labels, 1, "unit-variance");
  size_t i_W4 = model.AddWeight(num_labels, num_labels, "unit-variance");
  size_t i_b4 = model.AddWeight(num_labels, 1, "unit-variance");
  size_t i_u = model.AddWeight(num_labels, num_examples, "unit-variance");

  auto compute_output = [&](Eigen::MatrixXd *X_grad) {
    auto X = model.MakeInput(i_X);
    auto W1 = model.MakeInput(i_W1);
    auto b1 = model.MakeInput(i_b1);
    auto W2 = model.MakeInput(i_W2);
    auto b2 = model.MakeInput(i_b2);
    auto W3 = model.MakeInput(i_W3);
    auto b3 = model.MakeInput(i_b3);
    auto W4 = model.MakeInput(i_W4);
    auto b4 = model.MakeInput(i_b4);
    auto u = model.MakeInput(i_u);
    auto H = logistic(W4 * (W3 * relu(W2 * tanh(W1 * X + b1) + b2) + b3)
                      + b4) % u / 10.0;
    auto l = average(cross_entropy(H, labels));
    double l_value = l->ForwardBackward();
    if (X_grad != nullptr) { *X_grad = *X->gradient(); }
    return l_value;
  };

  Eigen::MatrixXd X_grad;
  double l0 = compute_output(&X_grad);

  for (size_t i = 0; i < X_grad.rows(); ++i) {
    for (size_t j = 0; j < X_grad.cols(); ++j) {
      (*model.weight(i_X))(i, j) += epsilon;
      double l1 = compute_output(nullptr);
      EXPECT_NEAR((l1 - l0) / epsilon, X_grad(i, j), 1e-5);
      (*model.weight(i_X))(i, j) -= epsilon;
    }
  }
}

TEST(Adam, Test) {
  autodiff::Model model;
  size_t i_X = model.AddWeight({{0}});
  auto X = model.MakeInput(i_X);
  (*X->gradient()).resize(1, 1);
  (*X->gradient())(0) = 10;

  double step_size = 0.5;
  double b1 = 0.6;
  double b2 = 0.3;
  double ep = 0.1;
  autodiff::Adam gd(&model, step_size, b1, b2, ep);
  gd.UpdateWeights();

  EXPECT_NEAR((*X->value())(0), -0.4941, 1e-4);
}

TEST(SimpleRNN, Test) {
  autodiff::Model model;
  size_t i_X1 = model.AddWeight({{1, 3}, {2, 4}});
  size_t i_X2 = model.AddWeight({{0, 0}, {1, -1}});
  autodiff::SimpleRNN srnn(2, 2, 1, &model);
  Eigen::MatrixXd U1(1, 2);
  U1 << 1, 1;
  Eigen::MatrixXd U2(1, 1);
  U2 << 2;
  Eigen::MatrixXd V1(1, 1);
  V1 << 3;
  Eigen::MatrixXd V2(1, 1);
  V2 << 2;
  Eigen::MatrixXd b1(1, 1);
  b1 << 1;
  Eigen::MatrixXd b2(1, 1);
  b2 << -1;
  srnn.SetWeights(U1, V1, b1, 0);
  srnn.SetWeights(U2, V2, b2, 1);

  auto X1 = model.MakeInput(i_X1);
  auto X2 = model.MakeInput(i_X2);
  auto HH = srnn.Transduce({X1, X2})[0];
  Eigen::MatrixXd upper_right_H = HH.back().back()->Forward();

  EXPECT_NEAR(upper_right_H(0, 0), 0.9872, 1e-4);
  EXPECT_NEAR(upper_right_H(0, 1), 0.9870, 1e-4);
}

TEST(LSTM, Test) {
  autodiff::Model model;
  size_t i_X1 = model.AddWeight({{1}});
  size_t i_X2 = model.AddWeight({{-1}});
  autodiff::LSTM lstm(1, 1, 1, &model);
  Eigen::MatrixXd raw_U(1, 1);
  raw_U << 0.5;
  Eigen::MatrixXd raw_V(1, 1);
  raw_V << 0.5;
  Eigen::MatrixXd raw_b(1, 1);
  raw_b << 0.5;
  Eigen::MatrixXd input_U(1, 1);
  input_U << 1;
  Eigen::MatrixXd input_V(1, 1);
  input_V << 1;
  Eigen::MatrixXd input_b(1, 1);
  input_b << 1;
  Eigen::MatrixXd forget_U(1, 1);
  forget_U << 0;
  Eigen::MatrixXd forget_V(1, 1);
  forget_V << 0;
  Eigen::MatrixXd forget_b(1, 1);
  forget_b << 0;
  Eigen::MatrixXd output_U(1, 1);
  output_U << 2;
  Eigen::MatrixXd output_V(1, 1);
  output_V << 2;
  Eigen::MatrixXd output_b(1, 1);
  output_b << 2;
  lstm.SetWeights(raw_U, raw_V, raw_b, input_U, input_V, input_b,
                  forget_U, forget_V, forget_b, output_U, output_V, output_b,
                  0);

  auto X1 = model.MakeInput(i_X1);
  auto X2 = model.MakeInput(i_X2);
  auto HH = lstm.Transduce({X1, X2})[0];
  Eigen::MatrixXd upper_right_H = HH.back().back()->Forward();

  EXPECT_NEAR(upper_right_H(0, 0), 0.3596, 1e-4);
}

TEST(LSTM, GradientCheck) {
  std::srand(std::time(0));
  double epsilon = 1e-4;
  size_t num_labels = 3;
  size_t batch_size = 4;
  size_t dim_observation = 2;
  size_t dim_state = 3;
  size_t num_layers = 3;

  std::vector<size_t> labels;
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dis(0, num_labels - 1);
  for (size_t i = 0; i < batch_size; ++i) { labels.push_back(dis(gen)); }

  autodiff::Model model;
  size_t i_X1 = model.AddWeight(dim_observation, batch_size, "unit-variance");
  size_t i_X2 = model.AddWeight(dim_observation, batch_size, "unit-variance");
  size_t i_X3 = model.AddWeight(dim_observation, batch_size, "unit-variance");
  autodiff::LSTM lstm(num_layers, dim_observation, dim_state, &model);
  size_t i_W = model.AddWeight(num_labels, dim_state, "unit-variance");

  auto compute_output = [&](Eigen::MatrixXd *X1_grad) {
    auto X1 = model.MakeInput(i_X1);
    auto X2 = model.MakeInput(i_X2);
    auto X3 = model.MakeInput(i_X3);
    auto H = lstm.Transduce({X1, X2, X3})[0].back().back();
    auto W = model.MakeInput(i_W);
    auto l = average(cross_entropy(W * H, labels));
    double l_value = l->ForwardBackward();
    if (X1_grad != nullptr) { *X1_grad = *X1->gradient(); }
    return l_value;
  };

  Eigen::MatrixXd X1_grad;
  double l0 = compute_output(&X1_grad);

  for (size_t i = 0; i < X1_grad.rows(); ++i) {
    for (size_t j = 0; j < X1_grad.cols(); ++j) {
      (*model.weight(i_X1))(i, j) += epsilon;
      double l1 = compute_output(nullptr);
      EXPECT_NEAR((l1 - l0) / epsilon, X1_grad(i, j), 1e-5);
      (*model.weight(i_X1))(i, j) -= epsilon;
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
