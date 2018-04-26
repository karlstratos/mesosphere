// Author: Karl Stratos (me@karlstratos.com)

#include "autodiff.h"

namespace autodiff {

void Variable::Backward(const std::vector<Variable *> &topological_ordering) {
  ASSERT(value_.rows() != 0, "Forward has not been called");
  ASSERT(value_.rows() == 1 && value_.cols() == 1, "Backward on a non-scalar: "
         << util_eigen::dimension_string(value_));
  gradient_ = Eigen::MatrixXd::Ones(1, 1);  // dx/dx = 1
  for (int i = topological_ordering.size() - 1; i >= 0; --i) {
    // Reverse topological ordering guarantees that the variable receives all
    // contributions to its gradient from children before propagating it.
    topological_ordering.at(i)->PropagateGradient();
  }
}

std::vector<Variable *> Variable::ForwardBackward() {
  std::vector<Variable *> topological_ordering;
  Forward(&topological_ordering);
  Backward(topological_ordering);
  return topological_ordering;
}

Input::Input(std::string name, Eigen::MatrixXd *input) : Variable(name) {
  input_ = input;
  gradient_ = Eigen::MatrixXd::Zero(input->rows(), input->cols());
}

void Input::Forward(std::vector<Variable *> *topological_ordering) {
  if (called_forward_) { return; }
  topological_ordering->push_back(this);
  called_forward_ = true;
}

Add::Add(std::string name, Variable *X, Variable *Y) : Variable(name) {
  AddParent(X);
  AddParent(Y);
  gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  if (X->NumColumns() > Y->NumColumns()) {
    ASSERT(Y->NumColumns() == 1, "Only matrix-vector add supported, given: "
           << X->Shape() << " + " << Y->Shape());
    matrix_vector_ = true;
  }
}

void Add::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  Parent(1)->Forward(topological_ordering);
  if (matrix_vector_) {
    value_ = Parent(0)->value()->colwise() +
             static_cast<Eigen::VectorXd>(*Parent(1)->value());
  } else {
    value_ = *Parent(0)->value() + *Parent(1)->value();
  }
  topological_ordering->push_back(this);
}

void Add::PropagateGradient() {
  *Parent(0)->gradient() += gradient_;
  *Parent(1)->gradient() += (matrix_vector_) ?
                            gradient_.rowwise().sum() :
                            gradient_;
}

Multiply::Multiply(std::string name, Variable *X, Variable *Y)
    : Variable(name) {
  AddParent(X);
  AddParent(Y);
  gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), Y->NumColumns());
}

void Multiply::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  Parent(1)->Forward(topological_ordering);
  value_ = *Parent(0)->value() * *Parent(1)->value();
  topological_ordering->push_back(this);
}

void Multiply::PropagateGradient() {
  *Parent(0)->gradient() += gradient_ * Parent(1)->value()->transpose();
  *Parent(1)->gradient() += Parent(0)->value()->transpose() * gradient_;
}

FlipSign::FlipSign(std::string name, Variable *X) : Variable(name) {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
}

void FlipSign::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = - *Parent(0)->value();
  topological_ordering->push_back(this);
}

Transpose::Transpose(std::string name, Variable *X) : Variable(name) {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(X->NumColumns(), X->NumRows());
}

void Transpose::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = Parent(0)->value()->transpose();
  topological_ordering->push_back(this);
}

Logistic::Logistic(std::string name, Variable *X) : Variable(name) {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(X->NumColumns(), X->NumRows());
}

void Logistic::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = Parent(0)->value()->unaryExpr(
      [](double x) { return 1 / (1 + exp(-x));});
  topological_ordering->push_back(this);
}

Tanh::Tanh(std::string name, Variable *X) : Variable(name) {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(X->NumColumns(), X->NumRows());
}

void Tanh::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = Parent(0)->value()->array().tanh();
  topological_ordering->push_back(this);
}

Softmax::Softmax(std::string name, Variable *X) : Variable(name) {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
}

void Softmax::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = util_eigen::softmax(*Parent(0)->value());
  topological_ordering->push_back(this);
}

void Softmax::PropagateGradient() {
  Eigen::MatrixXd v = gradient_.cwiseProduct(value_);
  *Parent(0)->gradient() += v - v.sum() * value_;
}

Pick::Pick(std::string name, Variable *X, size_t index)
    : Variable(name), index_(index) {
  ASSERT(X->NumColumns() == 1, "Can't pick a non-vector");
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(1, 1);
}

void Pick::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = Eigen::MatrixXd::Constant(1, 1, (*Parent(0)->value())(index_));
  topological_ordering->push_back(this);
}

void Pick::PropagateGradient() {
  (*Parent(0)->gradient())(index_) += gradient_(0);
}

PickNegativeLogSoftmax::PickNegativeLogSoftmax(std::string name, Variable *X,
                                               size_t index)
    : Variable(name), index_(index) {
  ASSERT(X->NumColumns() == 1, "Can't pick a non-vector");
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(1, 1);
}

void PickNegativeLogSoftmax::Forward(std::vector<Variable *>
                                     *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  cached_value_ = util_eigen::softmax(*Parent(0)->value());
  value_ = Eigen::MatrixXd::Constant(1, 1, -log(cached_value_(index_)));
  topological_ordering->push_back(this);
}

void PickNegativeLogSoftmax::PropagateGradient() {
  cached_value_(index_) -= 1.0;
  *Parent(0)->gradient() += gradient_(0) * cached_value_;
}

}  // namespace autodiff
