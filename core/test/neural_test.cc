// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../neural.h"

TEST(Add, Test0) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto z = x + y;
  double result = z->ForwardBackward();

  EXPECT_EQ(3, result);
  EXPECT_EQ(1, x->get_gradient(0));
  EXPECT_EQ(1, y->get_gradient(0));
}

TEST(Add, Test1) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto z = x + y;  // 1 + 2 = 3
  auto q = z + x;  // 3 + 1 = 4
  auto l = q + q;  // 4 + 4 = 8
  auto o = l + (y + (y + y));  // 4 + (2 + (2 + 2)) = 14
  double result = o->ForwardBackward();

  EXPECT_EQ(14, result);
  EXPECT_EQ(4, x->get_gradient(0));
  EXPECT_EQ(5, y->get_gradient(0));

  EXPECT_EQ(2, model.NumWeights());
  EXPECT_EQ(4, (*model.gradient(0))(0));
  EXPECT_EQ(5, (*model.gradient(1))(0));
}

TEST(Add, Test2) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});

  auto x = model.MakeInput(i_x);
  auto u3 = (x + (x + x)) + x;
  auto y = u3 + (x + (x + x));
  auto t = y + u3;
  auto z = t + y + u3;
  double result = z->ForwardBackward();
  EXPECT_EQ(22, result);
  EXPECT_EQ(22, x->get_gradient(0));
}

TEST(Add, MatrixVector) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1, 2, 3}});
  size_t i_y = model.AddWeight({{-1}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto z = x + y;
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_EQ(3, result);
  for (size_t i = 0; i < x->NumColumns(); ++i) {
    EXPECT_EQ(1, x->get_gradient(i));
  }
  EXPECT_EQ(3, y->get_gradient(0));
}

TEST(ScalarVariableAddSubtractMultiplyMix, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  auto x = model.MakeInput(i_x);
  auto y = x + 1;             //     y = x + 1  = 2
  auto z = 1 + 2 * x - 2;     //     z = 2x - 1 = 1
  auto q = y - z;             //     q = -x + 2 = 1
  auto t = z - y;             //     t = x - 2  = -1
  auto l = q * t;             //     l = -x^2 + 4x -4 = -1
  double result = l->ForwardBackward();

  EXPECT_EQ(-1, result);
  EXPECT_EQ(2, x->get_gradient(0));  // -2x + 4
}

TEST(ReduceSum, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1, 2}, {3, 4}, {5, 6}});
  auto x = model.MakeInput(i_x);
  auto z = sum(x);
  double result = z->ForwardBackward();

  EXPECT_EQ(21, result);
  for (size_t i = 0; i < x->NumRows(); ++i) {
    for (size_t j = 0; j < x->NumColumns(); ++j) {
      EXPECT_EQ(1, x->get_gradient(i, j));
    }
  }
}

TEST(Multiply, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{2}});
  size_t i_y = model.AddWeight({{3}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto l = x * (y * x) * y;
  auto q = x * l * l * x;  // x^6 y^4
  double result = q->ForwardBackward();

  EXPECT_EQ(5184, result);
  EXPECT_EQ(15552, x->get_gradient(0));  // dq/dx = 6 x^5 y^4
  EXPECT_EQ(6912, y->get_gradient(0));  // dq/dy = 4 x^6 y^3
}

TEST(Dot, Test) {
  neural::Model model;
  size_t i_X = model.AddWeight({{1, 3}, {2, 4}});
  size_t i_Y = model.AddWeight({{5, 7}, {6, 8}});

  auto X = model.MakeInput(i_X);
  auto Y = model.MakeInput(i_Y);
  auto Z = dot(X, Y);
  auto l = sum(Z);
  l->ForwardBackward();

  EXPECT_EQ(17, Z->get_value(0, 0));
  EXPECT_EQ(53, Z->get_value(0, 1));

  EXPECT_EQ(5, X->get_gradient(0, 0));
  EXPECT_EQ(6, X->get_gradient(1, 0));

  EXPECT_EQ(7, X->get_gradient(0, 1));
  EXPECT_EQ(8, X->get_gradient(1, 1));

  EXPECT_EQ(1, Y->get_gradient(0, 0));
  EXPECT_EQ(2, Y->get_gradient(1, 0));

  EXPECT_EQ(3, Y->get_gradient(0, 1));
  EXPECT_EQ(4, Y->get_gradient(1, 1));
}

TEST(AddMultiplyDotFlipSign, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}, {2}});
  size_t i_y = model.AddWeight({{3}, {4}});
  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto p = -dot(x + y, 2 * y);  // -2(x'y - y'y)
  double result = p->ForwardBackward();

  EXPECT_EQ(-72, result);

  // Dx = 2y = [6; 8]
  EXPECT_EQ(-6, x->get_gradient(0));
  EXPECT_EQ(-8, x->get_gradient(1));

  // Dy = 2x + 4y = [14; 20]
  EXPECT_EQ(-14, y->get_gradient(0));
  EXPECT_EQ(-20, y->get_gradient(1));
}


TEST(Logistic, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  auto x = model.MakeInput(i_x);
  auto z = logistic(x / 0.5);
  double result = z->ForwardBackward();

  EXPECT_NEAR(0.8808, result, 1e-4);
  EXPECT_NEAR(0.2100, x->get_gradient(0), 1e-4);
}

TEST(Tanh, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  auto x = model.MakeInput(i_x);
  auto z = tanh(2 * x);
  double result = z->ForwardBackward();

  EXPECT_NEAR(0.9640, result, 1e-4);
  EXPECT_NEAR(0.1413, x->get_gradient(0), 1e-4);
}

TEST(SoftmaxPick, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1, 1}, {2, 2}, {3, 5}});
  auto x = model.MakeInput(i_x);

  // y = [0.0900   0.0171
  //      0.2447   0.0466
  //      0.6652   0.9362]
  auto y = softmax(x);

  // z = [0.2447   0.9362]
  auto z = pick(y, {1, 2});

  // l = 1.1809
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_NEAR(1.1809, result, 1e-4);
  EXPECT_NEAR(0.2447, z->get_value(0), 1e-4);
  EXPECT_NEAR(0.9362, z->get_value(1), 1e-4);

  EXPECT_NEAR(0.0900, y->get_value(0, 0), 1e-4);
  EXPECT_NEAR(0.2447, y->get_value(1, 0), 1e-4);
  EXPECT_NEAR(0.6652, y->get_value(2, 0), 1e-4);
  EXPECT_NEAR(0.0171, y->get_value(0, 1), 1e-4);
  EXPECT_NEAR(0.0466, y->get_value(1, 1), 1e-4);
  EXPECT_NEAR(0.9362, y->get_value(2, 1), 1e-4);

  EXPECT_NEAR(-0.0220, x->get_gradient(0, 0), 1e-4);
  EXPECT_NEAR(0.1848, x->get_gradient(1, 0), 1e-4);
  EXPECT_NEAR(-0.1628, x->get_gradient(2, 0), 1e-4);
  EXPECT_NEAR(-0.0160, x->get_gradient(0, 1), 1e-4);
  EXPECT_NEAR(-0.0436, x->get_gradient(1, 1), 1e-4);
  EXPECT_NEAR(0.0597, x->get_gradient(2, 1), 1e-4);
}

TEST(PickNegativeLogSoftmax, Test) {
  neural::Model model;
  size_t i_x1 = model.AddWeight({{1}});
  size_t i_x2 = model.AddWeight({{2}});
  size_t i_x3 = model.AddWeight({{3}});
  auto x1 = model.MakeInput(i_x1);
  auto x2 = model.MakeInput(i_x2);
  auto x3 = model.MakeInput(i_x3);
  auto x4 = (x1 & x2 & x1) ^ (x1 & x2 & x1);
  auto x5 = (x1 & x1 & x3) ^ (x1 & x1 & (x3 + 2));
  auto x = x4 % x5;  // x = [1 1; 2 2; 3 5]

  // [0.0900   0.0171
  //  0.2447   0.0466     =>   -log(0.2447..) - log(0.9362..) = 1.4735
  //  0.6652   0.9362]
  auto l = sum(cross_entropy(x, {1, 2}));
  l->ForwardBackward();

  EXPECT_NEAR(1.4735, l->get_value(0), 1e-4);

  EXPECT_NEAR(0.0900, x->get_gradient(0, 0), 1e-4);  // p(0)
  EXPECT_NEAR(-0.7553, x->get_gradient(1, 0), 1e-4); // p(1) - 1
  EXPECT_NEAR(0.6652, x->get_gradient(2, 0), 1e-4);  // p(2)

  EXPECT_NEAR(0.0171, x->get_gradient(0, 1), 1e-4);  // p(0)
  EXPECT_NEAR(0.0466, x->get_gradient(1, 1), 1e-4);  // p(1)
  EXPECT_NEAR(-0.0638, x->get_gradient(2, 1), 1e-4); // p(2) - 1
}

TEST(FlagNegativeLogistic, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{-1, 2}});

  auto x = model.MakeInput(i_x);
  auto z = binary_cross_entropy(x, {false, true});
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_NEAR(0.4402, result, 1e-4);  // -log p(F1) - log p(T2)
  EXPECT_NEAR(0.3133, z->get_value(0), 1e-4);  // -log p(F1)
  EXPECT_NEAR(0.1269, z->get_value(1), 1e-4);  // -log p(T2)
  EXPECT_NEAR(0.2689, x->get_gradient(0), 1e-4);  //  p(T1)
  EXPECT_NEAR(-0.1192, x->get_gradient(1), 1e-4); // -p(F2)
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

  neural::Model model;
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
    if (X_grad != nullptr) { *X_grad = X->ref_gradient(); }
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
  neural::Model model;
  size_t i_X = model.AddWeight({{0}});
  auto X = model.MakeInput(i_X);
  X->ref_gradient()(0, 0) = 10;  // X already has gradient of shape (1, 1).

  double step_size = 0.5;
  double b1 = 0.6;
  double b2 = 0.3;
  double ep = 0.1;
  neural::Adam gd(&model, step_size, b1, b2, ep);
  gd.UpdateWeights();

  EXPECT_NEAR(X->get_value(0), -0.4941, 1e-4);
}

TEST(SimpleRNN, Test) {
  neural::Model model;
  size_t i_X1 = model.AddWeight({{1, 3}, {2, 4}});
  size_t i_X2 = model.AddWeight({{0, 0}, {1, -1}});
  neural::SimpleRNN srnn(2, 2, 1, &model);
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
  auto HHs = srnn.Transduce({X1, X2});
  Eigen::MatrixXd upper_right_H = HHs.back().back()[0]->Forward();

  EXPECT_NEAR(upper_right_H(0, 0), 0.9872, 1e-4);
  EXPECT_NEAR(upper_right_H(0, 1), 0.9870, 1e-4);
}

TEST(LSTM, Test) {
  neural::Model model;
  size_t i_X1 = model.AddWeight({{1}});
  size_t i_X2 = model.AddWeight({{-1}});
  neural::LSTM lstm(1, 1, 1, &model);
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
  auto HHs = lstm.Transduce({X1, X2});
  Eigen::MatrixXd upper_right_H = HHs.back().back()[0]->Forward();

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

  neural::Model model;
  size_t i_X1 = model.AddWeight(dim_observation, batch_size, "unit-variance");
  size_t i_X2 = model.AddWeight(dim_observation, batch_size, "unit-variance");
  size_t i_X3 = model.AddWeight(dim_observation, batch_size, "unit-variance");
  neural::LSTM lstm(num_layers, dim_observation, dim_state, &model);
  size_t i_W = model.AddWeight(num_labels, dim_state, "unit-variance");

  auto compute_output = [&](Eigen::MatrixXd *X1_grad) {
    auto X1 = model.MakeInput(i_X1);
    auto X2 = model.MakeInput(i_X2);
    auto X3 = model.MakeInput(i_X3);
    auto H = lstm.Transduce({X1, X2, X3}).back().back()[0];
    auto W = model.MakeInput(i_W);
    auto l = average(cross_entropy(W * H, labels));
    double l_value = l->ForwardBackward();
    if (X1_grad != nullptr) { *X1_grad = X1->ref_gradient(); }
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

TEST(LSTM, DropoutDoesNotCrash) {
  neural::Model model;
  size_t i_X1 = model.AddWeight(5, 10, "unit-variance");
  size_t i_X2 = model.AddWeight(5, 10, "unit-variance");
  neural::LSTM lstm(2, 5, 20, &model);
  lstm.UseDropout(0.5, 42);

  auto X1 = model.MakeInput(i_X1);
  auto X2 = model.MakeInput(i_X2);
  auto HHs = lstm.Transduce({X1, X2});
  sum(HHs.back().back()[0])->ForwardBackward();
}

TEST(OverwriteSharedPointers, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  const auto &x = model.MakeInput(i_x);
  const auto &y = model.MakeInput(i_y);
  auto z = x + y;  // 1 + 2
  z = z + y;  // 3 + 2
  z = z + x;  // 5 + 1
  z = z * y;  // 6 * 2
  double result = z->ForwardBackward();

  EXPECT_EQ(12, result);  // 2xy + 2y^2
  EXPECT_EQ(4, x->get_gradient(0));  // 2y
  EXPECT_EQ(10, y->get_gradient(0));  // 2x + 4y
}

TEST(IntermediateForwardCalls, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  const auto &x = model.MakeInput(i_x);
  const auto &y = model.MakeInput(i_y);
  auto z = x + y;  // 1 + 2
  EXPECT_EQ(3, z->Forward()(0, 0));

  z = z + y;  // 3 + 2
  EXPECT_EQ(5, z->Forward()(0, 0));

  z = z + x;  // 5 + 1
  EXPECT_EQ(6, z->Forward()(0, 0));

  z = z * y;  // 6 * 2
  EXPECT_EQ(12, z->Forward()(0, 0));

  double result = z->ForwardBackward();
  EXPECT_EQ(12, result);  // 2xy + 2y^2
  EXPECT_EQ(4, x->get_gradient(0));  // 2y
  EXPECT_EQ(10, y->get_gradient(0));  // 2x + 4y
}

TEST(InputColumn, OnlyIndividualColumnUpdates) {
  neural::Model model;
  //   1    2    3    4
  //   1    2    3    4
  //   1    2    3    4
  size_t i_X = model.AddWeight({{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}});
  neural::SimpleGradientDescent gd(&model, 0.01);

  auto x2 = model.MakeInputColumn(i_X, 1);  // (2 2 2)
  auto x3 = model.MakeInputColumn(i_X, 2);  // (3 3 3)
  EXPECT_EQ(3, x2->NumRows());
  EXPECT_EQ(1, x2->NumColumns());

  auto y = x2 % x3;  // (6 6 6)
  y = sum(y);  // 18
  double result = y->ForwardBackward();

  gd.UpdateWeights();

  //   1    1.97    2.98    4
  //   1    1.97    2.98    4
  //   1    1.97    2.98    4
  EXPECT_EQ(18, result);
  EXPECT_EQ(1.97, (*model.weight(i_X))(0, 1));
  EXPECT_EQ(1.97, (*model.weight(i_X))(1, 1));
  EXPECT_EQ(1.97, (*model.weight(i_X))(2, 1));
  EXPECT_EQ(2.98, (*model.weight(i_X))(0, 2));
  EXPECT_EQ(2.98, (*model.weight(i_X))(1, 2));
  EXPECT_EQ(2.98, (*model.weight(i_X))(2, 2));

  EXPECT_EQ(0, gd.num_updates(i_X));
  EXPECT_EQ(0, gd.num_column_updates(i_X, 0));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 1));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 2));
  EXPECT_EQ(0, gd.num_column_updates(i_X, 3));

  auto x4 = model.MakeInputColumn(i_X, 3);  // (4 4 4)
  x2 = model.MakeInputColumn(i_X, 1);  // (1.97 1.97 1.97)
  y = x4 + x2;  // (5.97 5.97 5.97)
  y = sum(y);  // 17.9
  result = y->ForwardBackward();
  gd.UpdateWeights();

  //   1    1.96    2.98    3.99
  //   1    1.96    2.98    3.99
  //   1    1.96    2.98    3.99
  EXPECT_EQ(17.91, result);
  EXPECT_EQ(1.96, (*model.weight(i_X))(0, 1));
  EXPECT_EQ(1.96, (*model.weight(i_X))(1, 1));
  EXPECT_EQ(1.96, (*model.weight(i_X))(2, 1));
  EXPECT_EQ(3.99, (*model.weight(i_X))(0, 3));
  EXPECT_EQ(3.99, (*model.weight(i_X))(1, 3));
  EXPECT_EQ(3.99, (*model.weight(i_X))(2, 3));

  EXPECT_EQ(0, gd.num_updates(i_X));
  EXPECT_EQ(0, gd.num_column_updates(i_X, 0));
  EXPECT_EQ(2, gd.num_column_updates(i_X, 1));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 2));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 3));
}

TEST(InputColumn, MixedUpdates) {
  neural::Model model;
  //   1    2    3
  size_t i_X = model.AddWeight({{1, 2, 3}});
  neural::SimpleGradientDescent gd(&model, 0.01);

  auto x1 = model.MakeInputColumn(i_X, 0);  // 1
  auto x3 = model.MakeInputColumn(i_X, 2);  // 3
  auto X = model.MakeInput(i_X);            // [1 2 3]
  auto y = sum((x1 % x3) * X);              // sum(3 * [1 2 3]) = 18
  double result = y->ForwardBackward();

  // x1^2 x3 + x_1 x2 x3 + x1 x3^2
  // d(x1) = 2 x1 x3 + x2 x3 + x3^2 = 6 + 6 + 9 = 21
  // d(x2) = x1 x3 = 3
  // d(x3) = x1^2 + x1 x2 + 2 x1 x3 = 1 + 2 + 6 = 9
  EXPECT_EQ(18, result);
  EXPECT_EQ(21, (*model.gradient(i_X))(0));
  EXPECT_EQ(3, (*model.gradient(i_X))(1));
  EXPECT_EQ(9, (*model.gradient(i_X))(2));

  gd.UpdateWeights();
  EXPECT_EQ(0.79, (*model.weight(i_X))(0));
  EXPECT_EQ(1.97, (*model.weight(i_X))(1));
  EXPECT_EQ(2.91, (*model.weight(i_X))(2));

  EXPECT_EQ(1, gd.num_updates(i_X));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 0));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 1));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 2));

  x1 = model.MakeInputColumn(i_X, 0);
  result = x1->ForwardBackward();
  EXPECT_EQ(0.79, result);
  EXPECT_EQ(1, (*model.gradient(i_X))(0));

  gd.UpdateWeights();
  EXPECT_EQ(0.78, (*model.weight(i_X))(0));
  EXPECT_EQ(1.97, (*model.weight(i_X))(1));
  EXPECT_EQ(2.91, (*model.weight(i_X))(2));

  EXPECT_EQ(1, gd.num_updates(i_X));
  EXPECT_EQ(2, gd.num_column_updates(i_X, 0));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 1));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 2));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
