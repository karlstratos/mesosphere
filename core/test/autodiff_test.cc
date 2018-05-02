// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../autodiff.h"

TEST(Add, Test1) {
  std::vector<autodiff::Input *> inputs;
  Eigen::MatrixXd x_value(1, 1);
  Eigen::MatrixXd y_value(1, 1);
  x_value << 1.0;
  y_value << 2.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value, &inputs);
  autodiff::Input *y = new autodiff::Input("y", &y_value, &inputs);
  autodiff::Add *z = new autodiff::Add(x, y);
  autodiff::Add *q = new autodiff::Add(z, x);
  autodiff::Add *l = new autodiff::Add(q, q);
  autodiff::Add *m = new autodiff::Add(y, y);
  autodiff::Add *n = new autodiff::Add(m, y);
  autodiff::Add *o = new autodiff::Add(l, n);

  // x=1_______     __
  //    \      \   /  \
  //     z=3____q=4____l=8____o=14
  //    /                    /
  // y=2______m=4_____n=6___/
  //    \____/_______/
  std::vector<autodiff::Variable *> topological_ordering = o->ForwardBackward();

  EXPECT_EQ(8, topological_ordering.size());
  EXPECT_EQ(14.0, (*o->value())(0));
  EXPECT_EQ(4.0, (*x->gradient())(0));
  EXPECT_EQ(5.0, (*y->gradient())(0));

  EXPECT_EQ(2, inputs.size());
  EXPECT_EQ(4, (*inputs[0]->gradient())(0));
  EXPECT_EQ(5, (*inputs[1]->gradient())(0));

  o->DeleteThisUniqueSink();
}

TEST(Add, Test2) {
  Eigen::MatrixXd x_value(1, 1);
  x_value << 1.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Add *u1 = new autodiff::Add(x, x);
  autodiff::Add *u2 = new autodiff::Add(x, u1);
  autodiff::Add *u3 = new autodiff::Add(u2, x);
  autodiff::Add *d1 = new autodiff::Add(x, x);
  autodiff::Add *d2 = new autodiff::Add(x, d1);
  autodiff::Add *y = new autodiff::Add(u3, d2);
  autodiff::Add *t = new autodiff::Add(u3, y);
  autodiff::Add *q = new autodiff::Add(y, u3);
  autodiff::Add *z = new autodiff::Add(t, q);

  //                                 z=22
  //                                |  \
  //                             ___|___q=11
  //                             |  |    |
  //                            /__t=11  /
  //    ___u1=2____u2=3____u3=4/  /     /
  //   /   /       /       /  \   |    /
  // x=1__/_______/_______/    \  |___/
  //  |____                     y=7
  //  /\   \                    /
  //  \/    \                  /
  //  d1=2___d2=3_____________/
  std::vector<autodiff::Variable *> topological_ordering = z->ForwardBackward();

  EXPECT_EQ(10, topological_ordering.size());
  EXPECT_EQ(22.0, (*z->value())(0));
  EXPECT_EQ(22.0, (*x->gradient())(0));

  z->DeleteThisUniqueSink();
}

TEST(Add, MatrixVector) {
  Eigen::MatrixXd x_value(1, 3);
  x_value << 1, 2, 3;
  Eigen::MatrixXd y_value(1, 1);
  y_value << -1;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Add *z = new autodiff::Add(x, y);  // [1 2 3] + [-1] = [0 1 2]
  autodiff::ReduceSum *l = new autodiff::ReduceSum(z);

  l->ForwardBackward();

  EXPECT_EQ(3, (*l->value())(0));
  for (size_t i = 0; i < x->NumColumns(); ++i) {
    EXPECT_EQ(1, (*x->gradient())(i));
  }
  EXPECT_EQ(3, (*y->gradient())(0));

  l->DeleteThisUniqueSink();
}

TEST(ReduceSum, Test) {
  Eigen::MatrixXd x_value(2, 3);
  x_value << 1, 2, 3, 4, 5, 6;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::ReduceSum *z = new autodiff::ReduceSum(x);

  z->ForwardBackward();

  EXPECT_EQ(21, (*z->value())(0));
  for (size_t i = 0; i < x->NumRows(); ++i) {
    for (size_t j = 0; j < x->NumColumns(); ++j) {
      EXPECT_EQ(1, (*x->gradient())(i, j));
    }
  }
  z->DeleteThisUniqueSink();
}

TEST(Multiply, Test) {
  Eigen::MatrixXd x_value(1, 1);
  Eigen::MatrixXd y_value(1, 1);
  x_value << 2.0;
  y_value << 3.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Multiply *z = new autodiff::Multiply(x, y);
  autodiff::Multiply *l = new autodiff::Multiply(z, z);
  autodiff::Multiply *t = new autodiff::Multiply(l, x);
  autodiff::Multiply *r = new autodiff::Multiply(x, l);
  autodiff::Multiply *q = new autodiff::Multiply(r, t);  // x^6 y^4

  std::vector<autodiff::Variable *> topological_ordering = q->ForwardBackward();

  EXPECT_EQ(7, topological_ordering.size());
  EXPECT_EQ(5184.0, (*q->value())(0));
  EXPECT_EQ(15552.0, (*x->gradient())(0));  // dq/dx = 6 x^5 y^4
  EXPECT_EQ(6912.0, (*y->gradient())(0));  // dq/dy = 4 x^6 y^3

  q->DeleteThisUniqueSink();
}

TEST(Dot, ColumnColumn) {
  Eigen::MatrixXd x_value(2, 1);
  Eigen::MatrixXd y_value(2, 1);
  x_value << 1, 1;
  y_value << 2, 3;

  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Dot *z = new autodiff::Dot(x, y);

  z->ForwardBackward();

  EXPECT_EQ(5, (*z->value())(0));
  EXPECT_EQ(2, (*x->gradient())(0));
  EXPECT_EQ(3, (*x->gradient())(1));
  EXPECT_EQ(1, (*y->gradient())(0));
  EXPECT_EQ(1, (*y->gradient())(1));

  z->DeleteThisUniqueSink();
}

TEST(Dot, ColumnRow) {
  Eigen::MatrixXd x_value(2, 1);
  Eigen::MatrixXd y_value(1, 2);
  x_value << 1, 1;
  y_value << 2, 3;

  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Dot *z = new autodiff::Dot(x, y);

  z->ForwardBackward();

  EXPECT_EQ(5, (*z->value())(0));
  EXPECT_EQ(2, (*x->gradient())(0));
  EXPECT_EQ(3, (*x->gradient())(1));
  EXPECT_EQ(1, (*y->gradient())(0));
  EXPECT_EQ(1, (*y->gradient())(1));

  z->DeleteThisUniqueSink();
}

TEST(Dot, RowColumn) {
  Eigen::MatrixXd x_value(1, 2);
  Eigen::MatrixXd y_value(2, 1);
  x_value << 1, 1;
  y_value << 2, 3;

  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Dot *z = new autodiff::Dot(x, y);

  z->ForwardBackward();

  EXPECT_EQ(5, (*z->value())(0));
  EXPECT_EQ(2, (*x->gradient())(0));
  EXPECT_EQ(3, (*x->gradient())(1));
  EXPECT_EQ(1, (*y->gradient())(0));
  EXPECT_EQ(1, (*y->gradient())(1));

  z->DeleteThisUniqueSink();
}

TEST(Dot, RowRow) {
  Eigen::MatrixXd x_value(1, 2);
  Eigen::MatrixXd y_value(1, 2);
  x_value << 1, 1;
  y_value << 2, 3;

  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Dot *z = new autodiff::Dot(x, y);

  z->ForwardBackward();

  EXPECT_EQ(5, (*z->value())(0));
  EXPECT_EQ(2, (*x->gradient())(0));
  EXPECT_EQ(3, (*x->gradient())(1));
  EXPECT_EQ(1, (*y->gradient())(0));
  EXPECT_EQ(1, (*y->gradient())(1));

  z->DeleteThisUniqueSink();
}

TEST(Dot, ScalarScalar) {
  Eigen::MatrixXd x_value(1, 1);
  Eigen::MatrixXd y_value(1, 1);
  x_value << 1;
  y_value << 2;

  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Dot *z = new autodiff::Dot(x, y);

  z->ForwardBackward();

  EXPECT_EQ(2, (*z->value())(0));
  EXPECT_EQ(2, (*x->gradient())(0));
  EXPECT_EQ(1, (*y->gradient())(0));

  z->DeleteThisUniqueSink();
}

TEST(AddMultiplyDotFlipSign, Test) {
  Eigen::MatrixXd x_value(2, 1);
  Eigen::MatrixXd y_value(2, 1);
  x_value << 1.0, 2.0;
  y_value << 3.0, 4.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Add *z = new autodiff::Add(x, y);  // x + y
  autodiff::Add *t = new autodiff::Add(y, y);

  // l = (x + y)^T (y + y) =  2x^T y + 2y^T y
  autodiff::Dot *l = new autodiff::Dot(z, t);

  // p = - 2x^T y - 2y^T y
  autodiff::FlipSign *p = new autodiff::FlipSign(l);

  p->ForwardBackward();

  EXPECT_EQ(-72.0, (*p->value())(0));

  // dl/dx = 2y = [6; 8]
  EXPECT_EQ(-6.0, (*x->gradient())(0));
  EXPECT_EQ(-8.0, (*x->gradient())(1));

  // dl/dy = 2x + 4y = [14; 20]
  EXPECT_EQ(-14.0, (*y->gradient())(0));
  EXPECT_EQ(-20.0, (*y->gradient())(1));

  p->DeleteThisUniqueSink();
}

TEST(Logistic, Test) {
  Eigen::MatrixXd x_value(1, 1);
  x_value << 1.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Add *y = new autodiff::Add(x, x);
  autodiff::Logistic *z = new autodiff::Logistic(y);

  z->ForwardBackward();

  EXPECT_NEAR(0.8808, (*z->value())(0), 1e-4);  // 2 s(2x)
  EXPECT_NEAR(0.2100, (*x->gradient())(0), 1e-4);  // 2 s(2x) (1-s(2x))

  z->DeleteThisUniqueSink();
}

TEST(Tanh, Test) {
  Eigen::MatrixXd x_value(1, 1);
  x_value << 1.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Add *y = new autodiff::Add(x, x);
  autodiff::Tanh *z = new autodiff::Tanh(y);

  // z = tanh(2x)
  // dz/dx = 2 (1 - tanh(2x)^2)
  z->ForwardBackward();

  EXPECT_NEAR(0.9640, (*z->value())(0), 1e-4);
  EXPECT_NEAR(0.1413, (*x->gradient())(0), 1e-4);

  z->DeleteThisUniqueSink();
}

TEST(SoftmaxPick, Test) {
  Eigen::MatrixXd x_value(3, 2);
  x_value << 1, 1, 2, 2, 3, 3;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Softmax *y = new autodiff::Softmax(x);
  autodiff::Pick *z = new autodiff::Pick(y, {1, 2});
  autodiff::ReduceSum *l = new autodiff::ReduceSum(z);

  l->ForwardBackward();

  EXPECT_NEAR(0.9100, (*l->value())(0), 1e-4);
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

  l->DeleteThisUniqueSink();
}

TEST(PickNegativeLogSoftmax, Test) {
  Eigen::MatrixXd x_value(3, 2);
  x_value << 1, 1, 2, 2, 3, 3;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::PickNegativeLogSoftmax *z
      = new autodiff::PickNegativeLogSoftmax(x, {1, 2});
  autodiff::ReduceSum *l = new autodiff::ReduceSum(z);

  l->ForwardBackward();

  EXPECT_NEAR(0.0900, (*x->gradient())(0, 0), 1e-4);  // p(0)
  EXPECT_NEAR(-0.7553, (*x->gradient())(1, 0), 1e-4); // p(1) - 1
  EXPECT_NEAR(0.6652, (*x->gradient())(2, 0), 1e-4);  // p(2)

  EXPECT_NEAR(0.0900, (*x->gradient())(0, 1), 1e-4);  // p(0)
  EXPECT_NEAR(0.2447, (*x->gradient())(1, 1), 1e-4);  // p(1)
  EXPECT_NEAR(-0.3348, (*x->gradient())(2, 1), 1e-4); // p(2) - 1

  l->DeleteThisUniqueSink();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
