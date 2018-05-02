// Author: Karl Stratos (me@karlstratos.com)

#include "autodiff.h"

namespace autodiff {

Eigen::MatrixXd Variable::Forward() {
  std::vector<Variable *> topological_ordering;
  Forward(&topological_ordering);
  return *value();
}

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

Input::Input(std::string name, Eigen::MatrixXd *input_value) : Variable(name) {
  input_value_ = input_value;
  gradient_ = Eigen::MatrixXd::Zero(input_value->rows(), input_value->cols());
}

Input::Input(std::string name, Eigen::MatrixXd *input_value,
             std::vector<Input *> *inputs, bool frozen) : Variable(name),
                                                          frozen_(frozen) {
  input_value_ = input_value;
  inputs->push_back(this);
  gradient_ = Eigen::MatrixXd::Zero(input_value->rows(), input_value->cols());
}

void Input::Forward(std::vector<Variable *> *topological_ordering) {
  if (called_forward_) { return; }
  topological_ordering->push_back(this);
  called_forward_ = true;
}

Add::Add(Variable *X, Variable *Y) : Variable(X->name() + " + " + Y->name()) {
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

ReduceSum::ReduceSum(Variable *X) : Variable("sum(" + X->name() + ")") {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(1, 1);
}

void ReduceSum::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = Eigen::MatrixXd::Constant(1, 1, Parent(0)->value()->sum());
  topological_ordering->push_back(this);
}

ReduceAverage::ReduceAverage(Variable *X) : Variable("avg(" + X->name() + ")") {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(1, 1);
}

void ReduceAverage::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = Eigen::MatrixXd::Constant(1, 1, Parent(0)->value()->mean());
  topological_ordering->push_back(this);
}

Multiply::Multiply(Variable *X, Variable *Y)
    : Variable(X->name() + " * " + Y->name()) {
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

Dot::Dot(Variable *X, Variable *Y) : Variable("dot(" + X->name() + ", "
                                              + Y->name() +")") {
  ASSERT(std::min(X->NumRows(), X->NumColumns()) == 1 &&
         std::min(Y->NumRows(), Y->NumColumns()) == 1 &&
         std::max(X->NumRows(),
                  X->NumColumns()) == std::max(Y->NumRows(),
                                               Y->NumColumns()),
         "Dot between X " << X->Shape() << " Y: " << Y->Shape());
  AddParent(X);
  AddParent(Y);
  gradient_ = Eigen::MatrixXd::Zero(1, 1);
}

void Dot::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  Parent(1)->Forward(topological_ordering);
  if (Parent(0)->NumColumns() == 1 && Parent(1)->NumColumns() == 1) {
    value_ = Parent(0)->value()->transpose() * *Parent(1)->value();
  } else if (Parent(0)->NumColumns() == 1 && Parent(1)->NumRows() == 1) {
    value_ = *Parent(1)->value() * *Parent(0)->value();
  } else if (Parent(0)->NumRows() == 1 && Parent(1)->NumColumns() == 1) {
    value_ = *Parent(0)->value() * *Parent(1)->value();
  } else {  // dot(row, row)
    value_ = *Parent(0)->value() * Parent(1)->value()->transpose();
  }
  topological_ordering->push_back(this);
}

void Dot::PropagateGradient() {
  if (Parent(0)->NumRows() == Parent(1)->NumRows()) {
    *Parent(0)->gradient() +=
        (gradient_(0) * Parent(1)->value()->array()).matrix();
    *Parent(1)->gradient() +=
        (gradient_(0) * Parent(0)->value()->array()).matrix();;
  } else {
    *Parent(0)->gradient() +=
        (gradient_(0) * Parent(1)->value()->transpose().array()).matrix();
    *Parent(1)->gradient() +=
        (gradient_(0) * Parent(0)->value()->transpose().array()).matrix();
  }
}

FlipSign::FlipSign(Variable *X) : Variable("-" + X->name()) {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
}

void FlipSign::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = - *Parent(0)->value();
  topological_ordering->push_back(this);
}

Transpose::Transpose(Variable *X) : Variable(X->name() + "^T") {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(X->NumColumns(), X->NumRows());
}

void Transpose::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = Parent(0)->value()->transpose();
  topological_ordering->push_back(this);
}

Logistic::Logistic(Variable *X) : Variable("logistic(" + X->name() + ")") {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
}

void Logistic::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = Parent(0)->value()->unaryExpr(
      [](double x) { return 1 / (1 + exp(-x));});
  topological_ordering->push_back(this);
}

Tanh::Tanh(Variable *X) : Variable("tanh(" + X->name() + ")") {
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
}

void Tanh::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  value_ = Parent(0)->value()->array().tanh();
  topological_ordering->push_back(this);
}

Softmax::Softmax(Variable *X) : Variable("softmax(" + X->name() + ")") {
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
  Eigen::MatrixXd A = gradient_.cwiseProduct(value_);
  *Parent(0)->gradient() += A - value_ * A.colwise().sum().asDiagonal();
}

Pick::Pick(Variable *X, const std::vector<size_t> &indices)
    : Variable("pick(" + X->name() + ", [" +
               util_string::convert_to_string(indices) + "]"),
      indices_(indices) {
  ASSERT(X->NumColumns() == indices.size(), "X " << X->Shape()
         << " vs # " << indices.size());
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(1, indices.size());
}

void Pick::Forward(std::vector<Variable *> *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);

  value_.resize(1, indices_.size());
  for (size_t i = 0; i < indices_.size(); ++i) {
    value_(i) = (*Parent(0)->value())(indices_[i]);
  }
  topological_ordering->push_back(this);
}

void Pick::PropagateGradient() {
  for (size_t i = 0; i < Parent(0)->NumColumns(); ++i) {
    Parent(0)->gradient()->col(i)(indices_[i]) += gradient_(i);
  }
}

PickNegativeLogSoftmax::PickNegativeLogSoftmax(Variable *X,
                                               const std::vector<size_t>
                                               &indices)
    : Variable("pnls(" + X->name() + ", [" +
               util_string::convert_to_string(indices) + "]"),
      indices_(indices) {
  ASSERT(X->NumColumns() == indices_.size(), "X Shape: " << X->Shape()
         << ", # indices: " << indices.size());
  AddParent(X);
  gradient_ = Eigen::MatrixXd::Zero(1, indices_.size());
}

void PickNegativeLogSoftmax::Forward(std::vector<Variable *>
                                     *topological_ordering) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_ordering);
  softmax_cache_ = util_eigen::softmax(*Parent(0)->value());

  value_.resize(1, indices_.size());
  for (size_t i = 0; i < indices_.size(); ++i) {
    value_(i) = -log(softmax_cache_(indices_[i], i));
  }
  topological_ordering->push_back(this);
}

void PickNegativeLogSoftmax::PropagateGradient() {
  for (size_t i = 0; i < indices_.size(); ++i) {
    softmax_cache_(indices_[i], i) -= 1.0;
  }
  *Parent(0)->gradient() += softmax_cache_ * gradient_.asDiagonal();
}

}  // namespace autodiff
