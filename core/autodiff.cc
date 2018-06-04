// Author: Karl Stratos (me@karlstratos.com)

#include "autodiff.h"

namespace autodiff {

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> X,
                                    std::shared_ptr<Variable> Y) {
  bool matrix_vector = (X->NumColumns() != Y->NumColumns()) &&
                       (Y->NumColumns() == 1);
  ASSERT(X->NumRows() == Y->NumRows() && (X->NumColumns() == Y->NumColumns() ||
                                          matrix_vector),
         "Add: must be either matrix-matrix or matrix-vector, given "
         << X->Shape() << " and " << Y->Shape());
  auto Z = std::make_shared<Add>();
  Z->AddParent(X);
  Z->AddParent(Y);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  Z->set_matrix_vector(matrix_vector);
  return Z;
}

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> X,
                                    std::shared_ptr<Variable> Y) {
  bool matrix_vector = (X->NumColumns() != Y->NumColumns()) &&
                       (Y->NumColumns() == 1);
  ASSERT(X->NumRows() == Y->NumRows() && (X->NumColumns() == Y->NumColumns() ||
                                          matrix_vector),
         "Subtract: must be either matrix-matrix or matrix-vector, given "
         << X->Shape() << " and " << Y->Shape());
  auto Z = std::make_shared<Subtract>();
  Z->AddParent(X);
  Z->AddParent(Y);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  Z->set_matrix_vector(matrix_vector);
  return Z;
}

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> X,
                                    double scalar_value) {
  auto Z = std::make_shared<AddScalar>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  Z->set_scalar_value(scalar_value);
  return Z;
}

std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> X,
                                    std::shared_ptr<Variable> Y) {
  ASSERT(X->NumColumns() == Y->NumRows(),
         "Multiply: dimensions do not match, given "
         << X->Shape() << " and " << Y->Shape());
  auto Z = std::make_shared<Multiply>();
  Z->AddParent(X);
  Z->AddParent(Y);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), Y->NumColumns());
  return Z;
}

std::shared_ptr<Variable> operator%(std::shared_ptr<Variable> X,
                                    std::shared_ptr<Variable> Y) {
  ASSERT(X->NumRows() == Y->NumRows() && X->NumColumns() == Y->NumColumns(),
         "Multiply element-wise: dimensions do not match, given "
         << X->Shape() << " and " << Y->Shape());
  auto Z = std::make_shared<MultiplyElementwise>();
  Z->AddParent(X);
  Z->AddParent(Y);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> X,
                                    double scalar_value) {
  auto Z = std::make_shared<MultiplyScalar>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  Z->set_scalar_value(scalar_value);
  return Z;
}

std::shared_ptr<Variable> operator&(std::shared_ptr<Variable> X,
                                    std::shared_ptr<Variable> Y) {
  ASSERT(X->NumColumns() == Y->NumColumns(),
         "vertical cat between " << X->Shape() << ", " << Y->Shape());
  auto Z = std::make_shared<ConcatenateVertical>();
  Z->AddParent(X);
  Z->AddParent(Y);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows() + Y->NumRows(),
                                         X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> operator^(std::shared_ptr<Variable> X,
                                    std::shared_ptr<Variable> Y) {
  ASSERT(X->NumRows() == Y->NumRows(),
         "horizontal cat between " << X->Shape() << ", " << Y->Shape());
  auto Z = std::make_shared<ConcatenateHorizontal>();
  Z->AddParent(X);
  Z->AddParent(Y);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(),
                                         X->NumColumns() + Y->NumColumns());
  return Z;
}

std::shared_ptr<Variable> dot(std::shared_ptr<Variable> X,
                              std::shared_ptr<Variable> Y) {
  ASSERT(X->NumRows() == Y->NumRows() &&
         X->NumColumns() == Y->NumColumns(),
         "column-wise dot between " << X->Shape() << ", " << Y->Shape());
  auto Z = std::make_shared<Dot>();
  Z->AddParent(X);
  Z->AddParent(Y);
  *Z->gradient() = Eigen::MatrixXd::Zero(1, X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<FlipSign>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> sum(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<ReduceSum>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(1, 1);
  return Z;
}

std::shared_ptr<Variable> average(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<ReduceAverage>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(1, 1);
  return Z;
}

std::shared_ptr<Variable> transpose(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<Transpose>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumColumns(), X->NumRows());
  return Z;
}

std::shared_ptr<Variable> logistic(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<Logistic>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> tanh(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<Tanh>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> relu(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<ReLU>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> softmax(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<Softmax>();
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> pick(std::shared_ptr<Variable> X,
                               const std::vector<size_t> &indices) {
  ASSERT(X->NumColumns() == indices.size(), X->Shape() << ", vs "
         << indices.size() << " indices");
  auto Z = std::make_shared<Pick>();
  Z->set_indices(indices);
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(1, indices.size());
  return Z;
}

std::shared_ptr<Variable> cross_entropy(
    std::shared_ptr<Variable> X, const std::vector<size_t> &indices) {
  ASSERT(X->NumColumns() == indices.size(), X->Shape() << ", vs "
         << indices.size() << " indices");
  auto Z = std::make_shared<PickNegativeLogSoftmax>();
  Z->set_indices(indices);
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(1, indices.size());
  return Z;
}

std::shared_ptr<Variable> binary_cross_entropy(
    std::shared_ptr<Variable> X, const std::vector<bool> &flags) {
  ASSERT(X->NumRows() == 1, "X not a row vector: " << X->Shape());
  ASSERT(X->NumColumns() == flags.size(), X->Shape() << ", vs "
         << flags.size() << " flags");
  auto Z = std::make_shared<FlagNegativeLogistic>();
  Z->set_flags(flags);
  Z->AddParent(X);
  *Z->gradient() = Eigen::MatrixXd::Zero(1, flags.size());
  return Z;
}

Eigen::MatrixXd Variable::Forward() {
  std::vector<std::shared_ptr<Variable>> topological_order;
  Forward(&topological_order);
  return *value();
}

void Variable::Backward(const std::vector<std::shared_ptr<Variable>>
                        &topological_order) {
  ASSERT(value_.rows() != 0, "Forward has not been called");
  ASSERT(value_.rows() == 1 && value_.cols() == 1, "Backward on a non-scalar: "
         << util_eigen::dimension_string(value_));
  gradient_ = Eigen::MatrixXd::Ones(1, 1);  // dx/dx = 1
  for (int i = topological_order.size() - 1; i >= 0; --i) {
    // Reverse topological order guarantees that the variable receives all
    // contributions to its gradient from children before propagating it.
    topological_order.at(i)->PropagateGradient();
  }
}

double Variable::ForwardBackward() {
  std::vector<std::shared_ptr<Variable>> topological_order;
  Forward(&topological_order);
  Backward(topological_order);
  return value_(0);
}

void Input::Forward(std::vector<std::shared_ptr<Variable>> *topological_order) {
  if (called_forward_) { return; }
  topological_order->push_back(shared_from_this());
  called_forward_ = true;
}

void Add::Forward(std::vector<std::shared_ptr<Variable>> *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  Parent(1)->Forward(topological_order);
  if (matrix_vector_) {
    value_ = Parent(0)->value()->colwise() +
             static_cast<Eigen::VectorXd>(*Parent(1)->value());
  } else {
    value_ = *Parent(0)->value() + *Parent(1)->value();
  }
  topological_order->push_back(shared_from_this());
}

void Add::PropagateGradient() {
  *Parent(0)->gradient() += gradient_;
  *Parent(1)->gradient() += (matrix_vector_) ?
                            gradient_.rowwise().sum() :
                            gradient_;
}

void AddScalar::Forward(std::vector<std::shared_ptr<Variable>>
                        *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = Parent(0)->value()->array() + scalar_value_;
  topological_order->push_back(shared_from_this());
}

void Subtract::Forward(std::vector<std::shared_ptr<Variable>>
                       *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  Parent(1)->Forward(topological_order);
  if (matrix_vector_) {
    value_ = Parent(0)->value()->colwise() -
             static_cast<Eigen::VectorXd>(*Parent(1)->value());
  } else {
    value_ = *Parent(0)->value() - *Parent(1)->value();
  }
  topological_order->push_back(shared_from_this());
}

void Subtract::PropagateGradient() {
  *Parent(0)->gradient() += gradient_;
  *Parent(1)->gradient() -= (matrix_vector_) ?
                            gradient_.rowwise().sum() :
                            gradient_;
}

void ReduceSum::Forward(std::vector<std::shared_ptr<Variable>>
                        *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = Eigen::MatrixXd::Constant(1, 1, Parent(0)->value()->sum());
  topological_order->push_back(shared_from_this());
}

void ReduceAverage::Forward(std::vector<std::shared_ptr<Variable>>
                            *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = Eigen::MatrixXd::Constant(1, 1, Parent(0)->value()->mean());
  topological_order->push_back(shared_from_this());
}

void Multiply::Forward(std::vector<std::shared_ptr<Variable>>
                       *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  Parent(1)->Forward(topological_order);
  value_ = *Parent(0)->value() * *Parent(1)->value();
  topological_order->push_back(shared_from_this());
}

void Multiply::PropagateGradient() {
  *Parent(0)->gradient() += gradient_ * Parent(1)->value()->transpose();
  *Parent(1)->gradient() += Parent(0)->value()->transpose() * gradient_;
}

void MultiplyElementwise::Forward(std::vector<std::shared_ptr<Variable>>
                                  *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  Parent(1)->Forward(topological_order);
  value_.array() = Parent(0)->value()->array() * Parent(1)->value()->array();
  topological_order->push_back(shared_from_this());
}

void MultiplyElementwise::PropagateGradient() {
  Parent(0)->gradient()->array() += gradient_.array() *
                                    Parent(1)->value()->array();
  Parent(1)->gradient()->array() += Parent(0)->value()->array() *
                                    gradient_.array();
}

void MultiplyScalar::Forward(std::vector<std::shared_ptr<Variable>>
                             *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = scalar_value_ * Parent(0)->value()->array();
  topological_order->push_back(shared_from_this());
}

void ConcatenateVertical::Forward(std::vector<std::shared_ptr<Variable>>
                                  *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  Parent(1)->Forward(topological_order);
  value_.resize(Parent(0)->NumRows() + Parent(1)->NumRows(),
                Parent(0)->NumColumns());
  value_ << *Parent(0)->value(), *Parent(1)->value();
  topological_order->push_back(shared_from_this());
}

void ConcatenateVertical::PropagateGradient() {
  *Parent(0)->gradient() += gradient_.topRows(Parent(0)->NumRows());
  *Parent(1)->gradient() += gradient_.bottomRows(Parent(1)->NumRows());
}

void ConcatenateHorizontal::Forward(std::vector<std::shared_ptr<Variable>>
                                    *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  Parent(1)->Forward(topological_order);
  value_.resize(Parent(0)->NumRows(),
                Parent(0)->NumColumns() + Parent(1)->NumColumns());
  value_ << *Parent(0)->value(), *Parent(1)->value();
  topological_order->push_back(shared_from_this());
}

void ConcatenateHorizontal::PropagateGradient() {
  *Parent(0)->gradient() += gradient_.leftCols(Parent(0)->NumColumns());
  *Parent(1)->gradient() += gradient_.rightCols(Parent(1)->NumColumns());
}

void Dot::Forward(std::vector<std::shared_ptr<Variable>> *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  Parent(1)->Forward(topological_order);
  value_ = (Parent(0)->value()->transpose() * *Parent(1)->value()).diagonal();
  topological_order->push_back(shared_from_this());
}

void Dot::PropagateGradient() {
  *Parent(0)->gradient() += *Parent(1)->value() * gradient_.asDiagonal();
  *Parent(1)->gradient() += *Parent(0)->value() * gradient_.asDiagonal();
}

void FlipSign::Forward(std::vector<std::shared_ptr<Variable>>
                       *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = - *Parent(0)->value();
  topological_order->push_back(shared_from_this());
}

void Transpose::Forward(std::vector<std::shared_ptr<Variable>>
                        *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = Parent(0)->value()->transpose();
  topological_order->push_back(shared_from_this());
}

void Logistic::Forward(std::vector<std::shared_ptr<Variable>>
                       *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = Parent(0)->value()->unaryExpr(
      [](double x) { return 1 / (1 + exp(-x));});
  topological_order->push_back(shared_from_this());
}

void Tanh::Forward(std::vector<std::shared_ptr<Variable>> *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = Parent(0)->value()->array().tanh();
  topological_order->push_back(shared_from_this());
}

void ReLU::Forward(std::vector<std::shared_ptr<Variable>> *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = Parent(0)->value()->unaryExpr(
      [](double x) { return std::max(0.0, x); });
  topological_order->push_back(shared_from_this());
}

void Softmax::Forward(std::vector<std::shared_ptr<Variable>>
                      *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  value_ = util_eigen::softmax(*Parent(0)->value());
  topological_order->push_back(shared_from_this());
}

void Softmax::PropagateGradient() {
  Eigen::MatrixXd A = gradient_.cwiseProduct(value_);
  *Parent(0)->gradient() += A - value_ * A.colwise().sum().asDiagonal();
}

void Pick::Forward(std::vector<std::shared_ptr<Variable>> *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);

  value_.resize(1, indices_.size());
  for (size_t i = 0; i < indices_.size(); ++i) {
    value_(i) = (*Parent(0)->value())(indices_[i]);
  }
  topological_order->push_back(shared_from_this());
}

void Pick::PropagateGradient() {
  for (size_t i = 0; i < Parent(0)->NumColumns(); ++i) {
    Parent(0)->gradient()->col(i)(indices_[i]) += gradient_(i);
  }
}

void PickNegativeLogSoftmax::Forward(std::vector<std::shared_ptr<Variable>>
                                     *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  softmax_cache_ = util_eigen::softmax(*Parent(0)->value());

  value_.resize(1, indices_.size());
  for (size_t i = 0; i < indices_.size(); ++i) {
    value_(i) = -log(softmax_cache_(indices_[i], i));
  }
  topological_order->push_back(shared_from_this());
}

void PickNegativeLogSoftmax::PropagateGradient() {
  for (size_t i = 0; i < indices_.size(); ++i) {
    softmax_cache_(indices_[i], i) -= 1.0;
  }
  *Parent(0)->gradient() += softmax_cache_ * gradient_.asDiagonal();
}

void FlagNegativeLogistic::Forward(std::vector<std::shared_ptr<Variable>>
                                   *topological_order) {
  if (value_.rows() > 0) { return; }
  Parent(0)->Forward(topological_order);
  logistic_cache_ = Parent(0)->value()->unaryExpr(
      [](double x) { return 1 / (1 + exp(-x)); });
  value_.resize(1, flags_.size());
  for (size_t i = 0; i < flags_.size(); ++i) {
    value_(i) = (flags_[i]) ? -log(logistic_cache_(i)) :
                -log(1 - logistic_cache_(i));

  }
  topological_order->push_back(shared_from_this());
}

void FlagNegativeLogistic::PropagateGradient() {
  for (size_t i = 0; i < flags_.size(); ++i) {
    if (flags_[i]) { logistic_cache_(i) -= 1.0; }
  }
  *Parent(0)->gradient() += logistic_cache_.cwiseProduct(gradient_);
}

size_t Model::AddWeight(const Eigen::MatrixXd &weight, bool frozen) {
  ASSERT(!made_input_, "Cannot add new weights after creating input pointers "
         " because they corrupt addresses");
  size_t i = weights_.size();
  weights_.push_back(weight);
  gradients_.resize(i + 1);
  frozen_.push_back(frozen);
  return i;
}

size_t Model::AddTemporaryWeight(const Eigen::MatrixXd &temporary_weight) {
  ASSERT(!made_temporary_input_, "Cannot add new temporary weights after "
         "creating input pointers because they corrupt addresses");
  size_t temporary_index = temporary_weights_.size();
  temporary_weights_.push_back(temporary_weight);
  temporary_gradients_.resize(temporary_index + 1);
  return temporary_index;
}

std::shared_ptr<Input> Model::MakeInput(size_t i) {
  auto X = std::make_shared<autodiff::Input>();
  gradients_[i] = Eigen::MatrixXd::Zero(weights_[i].rows(), weights_[i].cols());
  X->set_value_address(&weights_[i]);
  X->set_gradient_address(&gradients_[i]);
  if (!frozen_[i]) { update_list_.push_back(i); }
  made_input_ = true;
  return X;
}

std::shared_ptr<Input> Model::MakeTemporaryInput(size_t temporary_index) {
  auto X = std::make_shared<autodiff::Input>();
  temporary_gradients_[temporary_index] =
      Eigen::MatrixXd::Zero(temporary_weights_[temporary_index].rows(),
                            temporary_weights_[temporary_index].cols());
  X->set_value_address(&temporary_weights_[temporary_index]);
  X->set_gradient_address(&temporary_gradients_[temporary_index]);
  made_temporary_input_ = true;
  return X;
}

void Model::CleanUpAfterUpdate() {
  update_list_.clear();
  temporary_weights_.clear();
  temporary_gradients_.clear();
  made_input_ = false;
  made_temporary_input_ = false;
}

void Updater::UpdateWeights() {
  for (size_t i : *model_->update_list()) {
    UpdateWeight(i);
    ++num_updates_[i];
  }
  model_->CleanUpAfterUpdate();
}

Adam::Adam(Model *model, double step_size) : Updater(model) {
  step_size_ = step_size;
  InitializeMoments();
}

Adam::Adam(Model *model, double step_size, double b1, double b2,
           double ep) : Updater(model), b1_(b1), b2_(b2), ep_(ep) {
  step_size_ = step_size;
  InitializeMoments();
}

void Adam::InitializeMoments() {
  first_moments_.resize(model_->NumWeights());
  second_moments_.resize(model_->NumWeights());
  for (size_t i = 0; i < model_->NumWeights(); ++i) {
    size_t num_rows = model_->weight(i)->rows();
    size_t num_columns = model_->weight(i)->cols();
    if (!model_->frozen(i)) {
      first_moments_[i] = Eigen::ArrayXXd::Zero(num_rows, num_columns);
      second_moments_[i] = Eigen::ArrayXXd::Zero(num_rows, num_columns);
    }
  }
}

void Adam::UpdateWeight(size_t i) {
  size_t update_num = num_updates_[i] + 1;
  first_moments_[i] = b1_ * first_moments_[i] +
                      (1 - b1_) * model_->gradient(i)->array();
  second_moments_[i] = b2_ * second_moments_[i] +
                       (1 - b2_) * model_->gradient(i)->array().pow(2);
  double update_rate =
      step_size_ * sqrt(1 - pow(b2_, update_num)) / (1 - pow(b1_, update_num));
  model_->weight(i)->array() -=
      update_rate * (first_moments_[i] / (second_moments_[i].sqrt() + ep_));
}

}  // namespace autodiff
