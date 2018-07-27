// Author: Karl Stratos (me@karlstratos.com)

#include "neural.h"

namespace neural {

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
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
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
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  Z->set_matrix_vector(matrix_vector);
  return Z;
}

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> X,
                                    double scalar_value) {
  auto Z = std::make_shared<AddScalar>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
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
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), Y->NumColumns());
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
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> X,
                                    double scalar_value) {
  auto Z = std::make_shared<MultiplyScalar>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
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
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows() + Y->NumRows(),
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
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(),
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
  Z->gradient_ = Eigen::MatrixXd::Zero(1, X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<FlipSign>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> sum(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<ReduceSum>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, 1);
  return Z;
}

std::shared_ptr<Variable> average(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<ReduceAverage>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, 1);
  return Z;
}

std::shared_ptr<Variable> transpose(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<Transpose>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumColumns(), X->NumRows());
  return Z;
}

std::shared_ptr<Variable> logistic(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<Logistic>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> tanh(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<Tanh>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> relu(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<ReLU>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> softmax(std::shared_ptr<Variable> X) {
  auto Z = std::make_shared<Softmax>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

std::shared_ptr<Variable> pick(std::shared_ptr<Variable> X,
                               const std::vector<size_t> &indices) {
  ASSERT(X->NumColumns() == indices.size(), X->Shape() << ", vs "
         << indices.size() << " indices");
  auto Z = std::make_shared<Pick>(indices);
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, indices.size());
  return Z;
}

std::shared_ptr<Variable> cross_entropy(
    std::shared_ptr<Variable> X, const std::vector<size_t> &indices) {
  ASSERT(X->NumColumns() == indices.size(), X->Shape() << ", vs "
         << indices.size() << " indices");
  auto Z = std::make_shared<PickNegativeLogSoftmax>(indices);
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, indices.size());
  return Z;
}

std::shared_ptr<Variable> binary_cross_entropy(
    std::shared_ptr<Variable> X, const std::vector<bool> &flags) {
  ASSERT(X->NumRows() == 1, "X not a row vector: " << X->Shape());
  ASSERT(X->NumColumns() == flags.size(), X->Shape() << ", vs "
         << flags.size() << " flags");
  auto Z = std::make_shared<FlagNegativeLogistic>(flags);
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, flags.size());
  return Z;
}

double Variable::get_value(size_t i) {
  Eigen::Ref<Eigen::MatrixXd> value = ref_value();
  if (value.rows() == 1) { return value(0, i); }
  else if (value.cols() == 1) { return value(i, 0); }
  else {
    ASSERT(false, "get_value(" << i << ") on variable with ref_value() shape "
           << util_eigen::dimension_string(ref_value()));
  }
}

double Variable::get_gradient(size_t i) {
  Eigen::Ref<Eigen::MatrixXd> gradient = ref_gradient();
  if (gradient.rows() == 1) { return gradient(0, i); }
  else if (gradient.cols() == 1) { return gradient(i, 0); }
  else {
    ASSERT(false, "get_gradient(" << i << ") on variable with ref_gradient() "
           "shape " << util_eigen::dimension_string(ref_gradient()));
  }
}

Eigen::Ref<Eigen::MatrixXd> Variable::Forward(
    std::vector<std::shared_ptr<Variable>> *topological_order) {
  // Do zero work if the value has been computed AND has been appended to order.
  if (ref_value().cols() > 0 &&
      (appended_to_topological_order_ || !topological_order)) {
    return ref_value();
  }

  // Ensure parents have their values (and appended to order if haven't been).
  for (size_t i = 0; i < NumParents(); ++i) {
    Parent(i)->Forward(topological_order);
  }
  if (ref_value().cols() == 0) { ComputeValue(); }  // COMPUTE VALUE IF NONE.

  // Appends to the order only if it has never been appended.
  if (topological_order && !appended_to_topological_order_) {
    topological_order->push_back(shared_from_this());
    appended_to_topological_order_ = true;
  }
  return ref_value();
}

void Variable::Backward(const std::vector<std::shared_ptr<Variable>>
                        &topological_order) {
  ASSERT(ref_value().cols() > 0, "Forward has not been called");
  ASSERT(ref_value().rows() == 1 && ref_value().cols() == 1, "Backward on a "
         "non-scalar: " << util_eigen::dimension_string(value_));
  ref_gradient()(0, 0) = 1;  // dx/dx = 1
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
  return get_value(0);
}

Input::Input(Eigen::MatrixXd *value_address,
             Eigen::MatrixXd *gradient_address) : Variable() {
  value_address_ = value_address;
  gradient_address_ = gradient_address;
}

InputColumn::InputColumn(Eigen::MatrixXd *value_address,
                         Eigen::MatrixXd *gradient_address,
                         size_t column_index) :
    Input(value_address, gradient_address), column_index_(column_index) { }

void Add::ComputeValue() {
  if (matrix_vector_) {
    value_ = Parent(0)->ref_value().colwise() +
             static_cast<Eigen::VectorXd>(Parent(1)->ref_value());
  } else {
    value_ = Parent(0)->ref_value() + Parent(1)->ref_value();
  }
}

void Add::PropagateGradient() {
  Parent(0)->ref_gradient() += gradient_;
  Parent(1)->ref_gradient() += (matrix_vector_) ?
                               gradient_.rowwise().sum() :
                               gradient_;
}

void Subtract::ComputeValue() {
  if (matrix_vector_) {
    value_ = Parent(0)->ref_value().colwise() -
             static_cast<Eigen::VectorXd>(Parent(1)->ref_value());
  } else {
    value_ = Parent(0)->ref_value() - Parent(1)->ref_value();
  }
}

void Subtract::PropagateGradient() {
  Parent(0)->ref_gradient() += gradient_;
  Parent(1)->ref_gradient() -= (matrix_vector_) ?
                               gradient_.rowwise().sum() :
                               gradient_;
}

void Multiply::PropagateGradient() {
  Parent(0)->ref_gradient() += gradient_ * Parent(1)->ref_value().transpose();
  Parent(1)->ref_gradient() += Parent(0)->ref_value().transpose() * gradient_;
}

void MultiplyElementwise::PropagateGradient() {
  Parent(0)->ref_gradient().array() += gradient_.array() *
                                       Parent(1)->ref_value().array();
  Parent(1)->ref_gradient().array() += Parent(0)->ref_value().array() *
                                       gradient_.array();
}

void ConcatenateVertical::ComputeValue() {
  value_.resize(Parent(0)->NumRows() + Parent(1)->NumRows(),
                Parent(0)->NumColumns());
  value_ << Parent(0)->ref_value(), Parent(1)->ref_value();
}

void ConcatenateVertical::PropagateGradient() {
  Parent(0)->ref_gradient() += gradient_.topRows(Parent(0)->NumRows());
  Parent(1)->ref_gradient() += gradient_.bottomRows(Parent(1)->NumRows());
}

void ConcatenateHorizontal::ComputeValue() {
  value_.resize(Parent(0)->NumRows(),
                Parent(0)->NumColumns() + Parent(1)->NumColumns());
  value_ << Parent(0)->ref_value(), Parent(1)->ref_value();
}

void ConcatenateHorizontal::PropagateGradient() {
  Parent(0)->ref_gradient() += gradient_.leftCols(Parent(0)->NumColumns());
  Parent(1)->ref_gradient() += gradient_.rightCols(Parent(1)->NumColumns());
}

void Dot::PropagateGradient() {
  Parent(0)->ref_gradient() += Parent(1)->ref_value() * gradient_.asDiagonal();
  Parent(1)->ref_gradient() += Parent(0)->ref_value() * gradient_.asDiagonal();
}

void Softmax::PropagateGradient() {
  Eigen::MatrixXd A = gradient_.cwiseProduct(value_);
  Parent(0)->ref_gradient() += A - value_ * A.colwise().sum().asDiagonal();
}

void Pick::ComputeValue() {
  value_.resize(1, indices_.size());
  for (size_t i = 0; i < indices_.size(); ++i) {
    value_(i) =  Parent(0)->get_value(indices_[i], i);
  }
}

void Pick::PropagateGradient() {
  for (size_t i = 0; i < Parent(0)->NumColumns(); ++i) {
    Parent(0)->ref_gradient().col(i)(indices_[i]) += gradient_(i);
  }
}

void PickNegativeLogSoftmax::ComputeValue() {
  softmax_cache_ = util_eigen::softmax(Parent(0)->ref_value());
  value_.resize(1, indices_.size());
  for (size_t i = 0; i < indices_.size(); ++i) {
    value_(i) = -log(softmax_cache_(indices_[i], i));
  }
}

void PickNegativeLogSoftmax::PropagateGradient() {
  for (size_t i = 0; i < indices_.size(); ++i) {
    softmax_cache_(indices_[i], i) -= 1.0;
  }
  Parent(0)->ref_gradient() += softmax_cache_ * gradient_.asDiagonal();
}

void FlagNegativeLogistic::ComputeValue() {
  logistic_cache_ = Parent(0)->ref_value().unaryExpr(
      [](double x) { return 1 / (1 + exp(-x)); });
  value_.resize(1, flags_.size());
  for (size_t i = 0; i < flags_.size(); ++i) {
    value_(i) = (flags_[i]) ? -log(logistic_cache_(i)) :
                -log(1 - logistic_cache_(i));
  }
}

void FlagNegativeLogistic::PropagateGradient() {
  for (size_t i = 0; i < flags_.size(); ++i) {
    if (flags_[i]) { logistic_cache_(i) -= 1.0; }
  }
  Parent(0)->ref_gradient() += logistic_cache_.cwiseProduct(gradient_);
}

size_t Model::AddWeight(const Eigen::MatrixXd &weight, bool frozen) {
  ASSERT(!made_input_, "Cannot add new weights after creating input pointers "
         " because they corrupt addresses");
  size_t weight_index = weights_.size();
  weights_.push_back(weight);
  gradients_.push_back(Eigen::MatrixXd::Zero(weight.rows(), weight.cols()));
  frozen_.push_back(frozen);
  return weight_index;
}

size_t Model::AddTemporaryWeight(const Eigen::MatrixXd &temporary_weight) {
  ASSERT(!made_temporary_input_, "Cannot add new temporary weights after "
         "creating input pointers because they corrupt addresses");
  size_t temporary_index = temporary_weights_.size();
  temporary_weights_.push_back(temporary_weight);
  temporary_gradients_.push_back(
      Eigen::MatrixXd::Zero(temporary_weight.rows(), temporary_weight.cols()));
  return temporary_index;
}

std::shared_ptr<Input> Model::MakeInput(size_t weight_index) {
  gradients_[weight_index].setZero();  // Clearing gradient
  if (!frozen_[weight_index]) { update_set_.insert(weight_index); }
  auto X = std::make_shared<neural::Input>(&weights_[weight_index],
                                           &gradients_[weight_index]);
  active_variables_.push_back(X);
  made_input_ = true;
  return X;
}

std::shared_ptr<InputColumn> Model::MakeInputColumn(
    size_t weight_index, size_t column_index) {
  gradients_[weight_index].col(column_index).setZero();  // Clearing gradient
  if (!frozen_[weight_index]) {
    update_column_set_[weight_index].insert(column_index);
  }
  auto X = std::make_shared<neural::InputColumn>(
      &weights_[weight_index], &gradients_[weight_index], column_index);
  active_variables_.push_back(X);
  made_input_ = true;
  return X;
}

std::shared_ptr<Input> Model::MakeTemporaryInput(size_t temporary_index) {
  temporary_gradients_[temporary_index].setZero();  // Clearing gradient
  auto X = std::make_shared<neural::Input>(
      &temporary_weights_[temporary_index],
      &temporary_gradients_[temporary_index]);
  active_variables_.push_back(X);
  made_temporary_input_ = true;
  return X;
}

void Model::ClearComputation() {
  update_set_.clear();
  update_column_set_.clear();
  temporary_weights_.clear();
  temporary_gradients_.clear();
  made_input_ = false;
  made_temporary_input_ = false;
  active_variables_.clear();  // Will now free active variables out of scope.
}

void Updater::UpdateWeights() {
  for (const auto &pair : *model_->update_column_set()) {
    size_t weight_index = pair.first;
    bool need_to_initialize_counts = (num_column_updates_.find(weight_index) ==
                                      num_column_updates_.end());
    if (need_to_initialize_counts) {
      num_column_updates_[weight_index] =
          Eigen::VectorXi::Zero(model_->weight(weight_index)->cols());
    }
    bool not_densely_updated = (model_->update_set()->find(weight_index) ==
                                model_->update_set()->end());
    if (not_densely_updated) {  // To avoid updating twice
      for (size_t column_index : pair.second)  {
        UpdateWeightColumn(weight_index, column_index);

        // Convention 1: updating an individual column does not increment the
        // number of updates for the entire weight.
        num_column_updates_[weight_index](column_index) += 1;
      }
    }
  }

  for (size_t weight_index : *model_->update_set()) {
    UpdateWeight(weight_index);
    ++num_updates_[weight_index];

    // Convention 2: updating the entire weight DOES increment the number of
    // updates for individual columns.
    bool sparsely_updated = (model_->update_column_set()->find(weight_index) !=
                             model_->update_column_set()->end());
    if (sparsely_updated) { num_column_updates_[weight_index].array() += 1; }
  }

  model_->ClearComputation();
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
  for (size_t weight_index = 0; weight_index < model_->NumWeights();
       ++weight_index) {
    size_t num_rows = model_->weight(weight_index)->rows();
    size_t num_columns = model_->weight(weight_index)->cols();
    if (!model_->frozen(weight_index)) {
      first_moments_[weight_index] = Eigen::ArrayXXd::Zero(num_rows,
                                                           num_columns);
      second_moments_[weight_index] = Eigen::ArrayXXd::Zero(num_rows,
                                                            num_columns);
    }
  }
}

void Adam::UpdateWeight(size_t weight_index) {
  size_t update_num = num_updates_[weight_index] + 1;
  first_moments_[weight_index] =
      b1_ * first_moments_[weight_index] +
      (1 - b1_) * model_->gradient(weight_index)->array();
  second_moments_[weight_index] =
      b2_ * second_moments_[weight_index] +
      (1 - b2_) * model_->gradient(weight_index)->array().pow(2);
  double update_rate =
      step_size_ * sqrt(1 - pow(b2_, update_num)) / (1 - pow(b1_, update_num));
  model_->weight(weight_index)->array() -=
      update_rate * (first_moments_[weight_index] /
                     (second_moments_[weight_index].sqrt() + ep_));
}

void Adam::UpdateWeightColumn(size_t weight_index, size_t column_index) {
  size_t update_num = num_column_updates_[weight_index](column_index) + 1;
  first_moments_[weight_index].col(column_index) =
      b1_ * first_moments_[weight_index].col(column_index) +
      (1 - b1_) * model_->gradient(weight_index)->col(column_index).array();
  second_moments_[weight_index].col(column_index) =
      b2_ * second_moments_[weight_index].col(column_index) +
      (1 - b2_) * model_->gradient(weight_index)->col(column_index)\
      .array().pow(2);
  double update_rate =
      step_size_ * sqrt(1 - pow(b2_, update_num)) / (1 - pow(b1_, update_num));
  model_->weight(weight_index)->col(column_index).array() -=
      update_rate * (first_moments_[weight_index].col(column_index) /
                     (second_moments_[weight_index].col(column_index).sqrt()
                      + ep_));
}

RNN::RNN(size_t num_layers, size_t dim_observation, size_t dim_state,
         size_t num_state_types, Model *model_address) :
    num_layers_(num_layers), dim_observation_(dim_observation),
    dim_state_(dim_state), num_state_types_(num_state_types),
    model_address_(model_address) {
  // These constant weights are added as empty values now but will be set
  // dynamically per sequence. Note you must also shape their gradients then!
  initial_state_index_ = model_address_->AddWeight(0, 0, "zero", true);
  for (size_t layer = 0; layer < num_layers_; ++layer) {
    observation_mask_indices_.push_back(model_address_->AddWeight(0, 0, "zero",
                                                                  true));
    state_mask_indices_.push_back(model_address_->AddWeight(0, 0, "zero",
                                                            true));
  }
}

std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>> RNN::Transduce(
    const std::vector<std::shared_ptr<Variable>> &observation_sequence,
    const std::vector<std::vector<std::shared_ptr<Variable>>>
    &initial_state_stack) {
  std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>>
      state_stack_sequence;
  state_stack_sequence.push_back(
      ComputeNewStateStack(observation_sequence[0], initial_state_stack, true));
  for (size_t position = 1; position < observation_sequence.size();
       ++position) {
    state_stack_sequence.push_back(
        ComputeNewStateStack(observation_sequence[position],
                             state_stack_sequence.back()));
  }
  return state_stack_sequence;
}

std::vector<std::vector<std::shared_ptr<Variable>>> RNN::ComputeNewStateStack(
    const std::shared_ptr<Variable> &observation,
    const std::vector<std::vector<std::shared_ptr<Variable>>>
    &previous_state_stack, bool is_beginning) {
  ASSERT(is_beginning || previous_state_stack.size() > 0,
         "No previous state stack given even though not beginning a sequence");
  std::vector<std::shared_ptr<Variable>> initial_state;
  if (is_beginning) {  // Starting a new sequence
    batch_size_ = observation->NumColumns();
    if (previous_state_stack.size() == 0) {
      *model_address_->weight(initial_state_index_) =
          Eigen::MatrixXd::Zero(dim_state_, batch_size_);
      *model_address_->gradient(initial_state_index_) =  // Shape!
          Eigen::MatrixXd::Zero(dim_state_, batch_size_);
      for (size_t i = 0; i < num_state_types_; ++i) {
        initial_state.push_back(
            model_address_->MakeInput(initial_state_index_));
      }
    }
    if (dropout_rate_ > 0.0) { ComputeDropoutWeights(); }
  }
  std::vector<std::vector<std::shared_ptr<Variable>>> state_stack;
  for (size_t layer = 0; layer < num_layers_; ++layer) {
    const auto &O = (layer == 0) ? observation : state_stack.back()[0];
    const auto &previous_state = (previous_state_stack.size() > 0) ?
                                 previous_state_stack[layer] : initial_state;
    state_stack.push_back(ComputeNewState(O, previous_state, layer));
  }
  return state_stack;
}

void RNN::UseDropout(double dropout_rate, size_t random_seed) {
  dropout_rate_ = dropout_rate;
  gen_.seed(random_seed);
}

void RNN::ComputeDropoutWeights() {
  double keep_rate = 1.0 - dropout_rate_;
  std::bernoulli_distribution d(keep_rate);
  for (size_t layer = 0; layer < num_layers_; ++layer) {
    size_t dim_in = (layer == 0) ? dim_observation_ : dim_state_;
    Eigen::MatrixXd observation_bernoulli =
        Eigen::MatrixXd::Zero(dim_in, batch_size_).unaryExpr(
            [&](double x) { return static_cast<double>(d(gen_)); });
    Eigen::MatrixXd state_bernoulli =
        Eigen::MatrixXd::Zero(dim_state_, batch_size_).unaryExpr(
            [&](double x) { return static_cast<double>(d(gen_)); });
    *model_address_->weight(observation_mask_indices_[layer]) =
        observation_bernoulli / keep_rate;
    *model_address_->weight(state_mask_indices_[layer]) =
        state_bernoulli / keep_rate;
    *model_address_->gradient(observation_mask_indices_[layer]) =  // Shape!
        Eigen::MatrixXd::Zero(dim_in, batch_size_);
    *model_address_->gradient(state_mask_indices_[layer]) =
        Eigen::MatrixXd::Zero(dim_state_, batch_size_);
  }
}

SimpleRNN::SimpleRNN(size_t num_layers, size_t dim_observation,
                     size_t dim_state, Model *model_address) :
    RNN(num_layers, dim_observation, dim_state, 1, model_address) {
  for (size_t layer = 0; layer < num_layers_; ++layer) {
    size_t dim_in = (layer == 0) ? dim_observation_ : dim_state_;
    U_indices_.push_back(model_address_->AddWeight(dim_state_, dim_in,
                                                   "unit-variance"));
    V_indices_.push_back(model_address_->AddWeight(dim_state_, dim_state,
                                                   "unit-variance"));
    b_indices_.push_back(model_address_->AddWeight(dim_state_, 1,
                                                   "unit-variance"));
  }
}

std::vector<std::shared_ptr<Variable>> SimpleRNN::ComputeNewState(
    const std::shared_ptr<Variable> &observation,
    const std::vector<std::shared_ptr<Variable>> &previous_state,
    size_t layer) {
  auto O = observation;
  auto previous_H = previous_state[0];
  if (dropout_rate_ > 0.0) {
    O = model_address_->MakeInput(observation_mask_indices_[layer]) % O;
    previous_H = model_address_->MakeInput(state_mask_indices_[layer])
                 % previous_H;
  }

  const auto &U = model_address_->MakeInput(U_indices_[layer]);
  const auto &V = model_address_->MakeInput(V_indices_[layer]);
  const auto &b = model_address_->MakeInput(b_indices_[layer]);

  auto new_state = tanh(U * O + V * previous_H + b);
  return {new_state};
}

void SimpleRNN::SetWeights(const Eigen::MatrixXd &U_weight,
                           const Eigen::MatrixXd &V_weight,
                           const Eigen::MatrixXd &b_weight, size_t layer) {
  *model_address_->weight(U_indices_[layer]) = U_weight;
  *model_address_->weight(V_indices_[layer]) = V_weight;
  *model_address_->weight(b_indices_[layer]) = b_weight;
}

LSTM::LSTM(size_t num_layers, size_t dim_observation, size_t dim_state,
           Model *model_address) :
    RNN(num_layers, dim_observation, dim_state, 2, model_address) {
  for (size_t layer = 0; layer < num_layers_; ++layer) {
    size_t dim_in = (layer == 0) ? dim_observation_ : dim_state_;
    raw_U_indices_.push_back(
        model_address_->AddWeight(dim_state_, dim_in, "unit-variance"));
    raw_V_indices_.push_back(
        model_address_->AddWeight(dim_state_, dim_state_, "unit-variance"));
    raw_b_indices_.push_back(
        model_address_->AddWeight(dim_state_, 1, "unit-variance"));
    input_U_indices_.push_back(
        model_address_->AddWeight(dim_state_, dim_in, "unit-variance"));
    input_V_indices_.push_back(
        model_address_->AddWeight(dim_state_, dim_state_, "unit-variance"));
    input_b_indices_.push_back(
        model_address_->AddWeight(dim_state_, 1, "unit-variance"));
    forget_U_indices_.push_back(
        model_address_->AddWeight(dim_state_, dim_in, "unit-variance"));
    forget_V_indices_.push_back(
        model_address_->AddWeight(dim_state_, dim_state_, "unit-variance"));
    forget_b_indices_.push_back(
        model_address_->AddWeight(dim_state_, 1, "unit-variance"));
    output_U_indices_.push_back(
        model_address_->AddWeight(dim_state_, dim_in, "unit-variance"));
    output_V_indices_.push_back(
        model_address_->AddWeight(dim_state_, dim_state_, "unit-variance"));
    output_b_indices_.push_back(
        model_address_->AddWeight(dim_state_, 1, "unit-variance"));
  }
}

std::vector<std::shared_ptr<Variable>> LSTM::ComputeNewState(
    const std::shared_ptr<Variable> &observation,
    const std::vector<std::shared_ptr<Variable>> &previous_state,
    size_t layer) {
  auto O = observation;
  auto previous_H = previous_state[0];

  if (dropout_rate_ > 0.0) {
    O = model_address_->MakeInput(observation_mask_indices_[layer]) % O;
    previous_H = model_address_->MakeInput(state_mask_indices_[layer])
                 % previous_H;
  }
  const auto &raw_U = model_address_->MakeInput(raw_U_indices_[layer]);
  const auto &raw_V = model_address_->MakeInput(raw_V_indices_[layer]);
  const auto &raw_b = model_address_->MakeInput(raw_b_indices_[layer]);

  auto raw_H = tanh(raw_U * O + raw_V * previous_H + raw_b);

  const auto &input_U = model_address_->MakeInput(input_U_indices_[layer]);
  const auto &input_V = model_address_->MakeInput(input_V_indices_[layer]);
  const auto &input_b = model_address_->MakeInput(input_b_indices_[layer]);

  auto input_gate = logistic(input_U * O + input_V * previous_H + input_b);
  auto gated_H = input_gate % raw_H;

  const auto &previous_C = previous_state[1];
  const auto &forget_U = model_address_->MakeInput(forget_U_indices_[layer]);
  const auto &forget_V = model_address_->MakeInput(forget_V_indices_[layer]);
  const auto &forget_b = model_address_->MakeInput(forget_b_indices_[layer]);

  auto forget_gate = logistic(forget_U * O + forget_V * previous_H + forget_b);
  auto gated_previous_C = forget_gate % previous_C;
  auto new_C = gated_H + gated_previous_C;

  const auto &output_U = model_address_->MakeInput(output_U_indices_[layer]);
  const auto &output_V = model_address_->MakeInput(output_V_indices_[layer]);
  const auto &output_b = model_address_->MakeInput(output_b_indices_[layer]);

  auto output_gate = logistic(output_U * O + output_V * previous_H + output_b);
  auto new_H = output_gate % tanh(new_C);

  return {new_H, new_C};
}

void LSTM::SetWeights(const Eigen::MatrixXd &raw_U_weight,
                      const Eigen::MatrixXd &raw_V_weight,
                      const Eigen::MatrixXd &raw_b_weight,
                      const Eigen::MatrixXd &input_U_weight,
                      const Eigen::MatrixXd &input_V_weight,
                      const Eigen::MatrixXd &input_b_weight,
                      const Eigen::MatrixXd &forget_U_weight,
                      const Eigen::MatrixXd &forget_V_weight,
                      const Eigen::MatrixXd &forget_b_weight,
                      const Eigen::MatrixXd &output_U_weight,
                      const Eigen::MatrixXd &output_V_weight,
                      const Eigen::MatrixXd &output_b_weight,
                      size_t layer) {
  *model_address_->weight(raw_U_indices_[layer]) = raw_U_weight;
  *model_address_->weight(raw_V_indices_[layer]) = raw_V_weight;
  *model_address_->weight(raw_b_indices_[layer]) = raw_b_weight;
  *model_address_->weight(input_U_indices_[layer]) = input_U_weight;
  *model_address_->weight(input_V_indices_[layer]) = input_V_weight;
  *model_address_->weight(input_b_indices_[layer]) = input_b_weight;
  *model_address_->weight(forget_U_indices_[layer]) = forget_U_weight;
  *model_address_->weight(forget_V_indices_[layer]) = forget_V_weight;
  *model_address_->weight(forget_b_indices_[layer]) = forget_b_weight;
  *model_address_->weight(output_U_indices_[layer]) = output_U_weight;
  *model_address_->weight(output_V_indices_[layer]) = output_V_weight;
  *model_address_->weight(output_b_indices_[layer]) = output_b_weight;
}

}  // namespace neural
