// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../autodiff.h"

TEST(Add, Test1) {
  Eigen::MatrixXd x_value(1, 1);
  Eigen::MatrixXd y_value(1, 1);
  x_value << 1.0;
  y_value << 2.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Add *z = new autodiff::Add("z", x, y);
  autodiff::Add *q = new autodiff::Add("q", z, x);
  autodiff::Add *l = new autodiff::Add("l", q, q);
  autodiff::Add *m = new autodiff::Add("m", y, y);
  autodiff::Add *n = new autodiff::Add("n", m, y);
  autodiff::Add *o = new autodiff::Add("o", l, n);

  // x=1_______     __
  //    \      \   /  \
  //     z=3____q=4____l=8____o=14
  //    /                    /
  // y=2______m=4_____n=6___/
  //    \____/_______/
  std::vector<autodiff::Variable *> topological_ordering = o->ForwardBackward();

  EXPECT_EQ(topological_ordering.size(), 8);
  EXPECT_EQ((*o->value())(0), 14.0);
  EXPECT_EQ((*x->gradient())(0), 4.0);
  EXPECT_EQ((*y->gradient())(0), 5.0);

  o->DeleteThisUniqueSink();
}

TEST(Add, Test2) {
  Eigen::MatrixXd x_value(1, 1);
  x_value << 1.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Add *u1 = new autodiff::Add("u1", x, x);
  autodiff::Add *u2 = new autodiff::Add("u2", x, u1);
  autodiff::Add *u3 = new autodiff::Add("u3", u2, x);
  autodiff::Add *d1 = new autodiff::Add("d1", x, x);
  autodiff::Add *d2 = new autodiff::Add("d2", x, d1);
  autodiff::Add *y = new autodiff::Add("y", u3, d2);
  autodiff::Add *t = new autodiff::Add("t", u3, y);
  autodiff::Add *q = new autodiff::Add("q", y, u3);
  autodiff::Add *z = new autodiff::Add("z", t, q);

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

  EXPECT_EQ(topological_ordering.size(), 10);
  EXPECT_EQ((*z->value())(0), 22.0);
  EXPECT_EQ((*x->gradient())(0), 22.0);

  z->DeleteThisUniqueSink();
}

TEST(Multiply, Test) {
  Eigen::MatrixXd x_value(1, 1);
  Eigen::MatrixXd y_value(1, 1);
  x_value << 2.0;
  y_value << 3.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Multiply *z = new autodiff::Multiply("z", x, y);
  autodiff::Multiply *l = new autodiff::Multiply("l", z, z);
  autodiff::Multiply *t = new autodiff::Multiply("t", l, x);
  autodiff::Multiply *r = new autodiff::Multiply("r", x, l);
  autodiff::Multiply *q = new autodiff::Multiply("q", r, t);  // x^6 y^4

  std::vector<autodiff::Variable *> topological_ordering = q->ForwardBackward();

  EXPECT_EQ(topological_ordering.size(), 7);
  EXPECT_EQ((*q->value())(0), 5184.0);
  EXPECT_EQ((*x->gradient())(0), 15552.0);  // dq/dx = 6 x^5 y^4
  EXPECT_EQ((*y->gradient())(0), 6912.0);  // dq/dy = 4 x^6 y^3

  q->DeleteThisUniqueSink();
}

TEST(AddMultiplyTransposeFlipSign, Test) {
  Eigen::MatrixXd x_value(2, 1);
  Eigen::MatrixXd y_value(2, 1);
  x_value << 1.0, 2.0;
  y_value << 3.0, 4.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Input *y = new autodiff::Input("y", &y_value);
  autodiff::Add *z = new autodiff::Add("z", x, y);  // x + y
  autodiff::Transpose *z_t = new autodiff::Transpose("z'", z);  // (x + y)^T
  autodiff::Add *t = new autodiff::Add("t", y, y);

  // l = (x + y)^T (y + y) =  2x^T y + 2y^T y
  autodiff::Multiply *l = new autodiff::Multiply("l", z_t, t);

  // p = - 2x^T y - 2y^T y
  autodiff::FlipSign *p = new autodiff::FlipSign("p", l);

  p->ForwardBackward();

  EXPECT_EQ((*p->value())(0), -72.0);

  // dl/dx = 2y = [6; 8]
  EXPECT_EQ((*x->gradient())(0), -6.0);
  EXPECT_EQ((*x->gradient())(1), -8.0);

  // dl/dy = 2x + 4y = [14; 20]
  EXPECT_EQ((*y->gradient())(0), -14.0);
  EXPECT_EQ((*y->gradient())(1), -20.0);

  p->DeleteThisUniqueSink();
}

TEST(Logistic, Test) {
  Eigen::MatrixXd x_value(1, 1);
  x_value << 1.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Add *y = new autodiff::Add("y", x, x);
  autodiff::Logistic *z = new autodiff::Logistic("z", y);

  z->ForwardBackward();

  EXPECT_NEAR((*z->value())(0), 0.8808, 1e-4);  // 2 s(2x)
  EXPECT_NEAR((*x->gradient())(0), 0.2100, 1e-4);  // 2 s(2x) (1-s(2x))

  z->DeleteThisUniqueSink();
}

TEST(Tanh, Test) {
  Eigen::MatrixXd x_value(1, 1);
  x_value << 1.0;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Add *y = new autodiff::Add("y", x, x);
  autodiff::Tanh *z = new autodiff::Tanh("z", y);

  // z = tanh(2x)
  // dz/dx = 2 (1 - tanh(2x)^2)
  z->ForwardBackward();

  EXPECT_NEAR((*z->value())(0), 0.9640, 1e-4);
  EXPECT_NEAR((*x->gradient())(0), 0.1413, 1e-4);

  z->DeleteThisUniqueSink();
}

TEST(SoftmaxPick, Test) {
  Eigen::MatrixXd x_value(3, 1);
  x_value << 1, 2, 3;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::Softmax *y = new autodiff::Softmax("y", x);
  autodiff::Pick *z = new autodiff::Pick("z", y, 1);

  z->ForwardBackward();

  EXPECT_NEAR((*z->value())(0), 0.2447, 1e-4);
  EXPECT_NEAR((*x->gradient())(0), -0.0220, 1e-4);
  EXPECT_NEAR((*x->gradient())(1), 0.1848, 1e-4);
  EXPECT_NEAR((*x->gradient())(2), -0.1628, 1e-4);

  z->DeleteThisUniqueSink();
}

TEST(PickNegativeLogSoftmax, Test) {
  Eigen::MatrixXd x_value(3, 1);
  x_value << 1, 2, 3;
  autodiff::Input *x = new autodiff::Input("x", &x_value);
  autodiff::PickNegativeLogSoftmax *z
      = new autodiff::PickNegativeLogSoftmax("x", x, 1);

  z->ForwardBackward();

  EXPECT_NEAR((*z->value())(0), 1.4076, 1e-4);
  EXPECT_NEAR((*x->gradient())(0), 0.0900, 1e-4);  // p(0)
  EXPECT_NEAR((*x->gradient())(1), -0.7553, 1e-4); // p(1) - 1
  EXPECT_NEAR((*x->gradient())(2), 0.6652, 1e-4);  // p(2)

  z->DeleteThisUniqueSink();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
