// Author: Karl Stratos (me@karlstratos.com)

#include "neural.h"

namespace neural {

sp<Variable> operator+(const sp<Variable> &X, const sp<Variable> &Y) {
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

sp<Variable> operator-(const sp<Variable> &X, const sp<Variable> &Y) {
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

sp<Variable> operator+(const sp<Variable> &X, double scalar_value) {
  auto Z = std::make_shared<AddScalar>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  Z->set_scalar_value(scalar_value);
  return Z;
}

sp<Variable> operator*(const sp<Variable> &X, const sp<Variable> &Y) {
  ASSERT(X->NumColumns() == Y->NumRows(),
         "Multiply: dimensions do not match, given "
         << X->Shape() << " and " << Y->Shape());
  auto Z = std::make_shared<Multiply>();
  Z->AddParent(X);
  Z->AddParent(Y);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), Y->NumColumns());
  return Z;
}

sp<Variable> operator%(const sp<Variable> &X, const sp<Variable> &Y) {
  ASSERT(X->NumRows() == Y->NumRows() && X->NumColumns() == Y->NumColumns(),
         "Multiply element-wise: dimensions do not match, given "
         << X->Shape() << " and " << Y->Shape());
  auto Z = std::make_shared<MultiplyElementwise>();
  Z->AddParent(X);
  Z->AddParent(Y);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

sp<Variable> operator*(const sp<Variable> &X, double scalar_value) {
  auto Z = std::make_shared<MultiplyScalar>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  Z->set_scalar_value(scalar_value);
  return Z;
}

sp<Variable> dot(const sp<Variable> &X, const sp<Variable> &Y) {
  ASSERT(X->NumRows() == Y->NumRows() &&
         X->NumColumns() == Y->NumColumns(),
         "column-wise dot between " << X->Shape() << ", " << Y->Shape());
  auto Z = std::make_shared<Dot>();
  Z->AddParent(X);
  Z->AddParent(Y);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, X->NumColumns());
  return Z;
}

sp<Variable> operator-(const sp<Variable> &X) {
  auto Z = std::make_shared<FlipSign>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

sp<Variable> sum(const sp<Variable> &X) {
  auto Z = std::make_shared<ReduceSum>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, 1);
  return Z;
}

sp<Variable> sum_cwise(const sp<Variable> &X) {
  auto Z = std::make_shared<ReduceSumColumnWise>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, X->NumColumns());
  return Z;
}

sp<Variable> sum_rwise(const sp<Variable> &X) {
  auto Z = std::make_shared<ReduceSumRowWise>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), 1);
  return Z;
}

sp<Variable> transpose(const sp<Variable> &X) {
  auto Z = std::make_shared<Transpose>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumColumns(), X->NumRows());
  return Z;
}

sp<Variable> logistic(const sp<Variable> &X) {
  auto Z = std::make_shared<Logistic>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

sp<Variable> log(const sp<Variable> &X) {
  auto Z = std::make_shared<Log>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

sp<Variable> tanh(const sp<Variable> &X) {
  auto Z = std::make_shared<Tanh>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

sp<Variable> relu(const sp<Variable> &X) {
  auto Z = std::make_shared<ReLU>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

sp<Variable> softmax(const sp<Variable> &X) {
  auto Z = std::make_shared<Softmax>();
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());
  return Z;
}

sp<Variable> sum(const sp_v1<Variable> &Xs) {
  auto Z = std::make_shared<Sum>();
  for (const auto &X : Xs) {
    ASSERT(X->NumRows() == Xs[0]->NumRows() &&
           X->NumColumns() == Xs[0]->NumColumns(),
           "sum between " << X->Shape() << ", " << Xs[0]->Shape());
    Z->AddParent(X);
  }
  Z->gradient_ = Eigen::MatrixXd::Zero(Xs[0]->NumRows(), Xs[0]->NumColumns());
  return Z;
}

sp<Variable> vcat(const sp_v1<Variable> &Xs) {
  size_t num_rows = 0;
  auto Z = std::make_shared<ConcatenateVertical>();
  for (const auto &X : Xs) {
    ASSERT(X->NumColumns() == Xs[0]->NumColumns(),
           "vertical cat between " << X->Shape() << ", " << Xs[0]->Shape());
    Z->AddParent(X);
    num_rows += X->NumRows();
  }
  Z->gradient_ = Eigen::MatrixXd::Zero(num_rows, Xs[0]->NumColumns());
  return Z;
}

sp<Variable> hcat(const sp_v1<Variable> &Xs) {
  size_t num_columns = 0;
  auto Z = std::make_shared<ConcatenateHorizontal>();
  for (const auto &X : Xs) {
    ASSERT(X->NumRows() == Xs[0]->NumRows(),
           "horizontal cat between " << X->Shape() << ", " << Xs[0]->Shape());
    Z->AddParent(X);
    num_columns += X->NumColumns();
  }
  Z->gradient_ = Eigen::MatrixXd::Zero(Xs[0]->NumRows(), num_columns);
  return Z;
}

sp<Variable> block(const sp<Variable> &X,
                   size_t start_row, size_t start_column,
                   size_t num_rows, size_t num_columns) {
  auto Z = std::make_shared<Block>();
  Z->AddParent(X);
  Z->SetBlock(start_row, start_column, num_rows, num_columns);
  return Z;
}

sp<Variable> pick(const sp<Variable> &X, const std::vector<size_t> &indices) {
  ASSERT(X->NumColumns() == indices.size(), X->Shape() << ", vs "
         << indices.size() << " indices");
  auto Z = std::make_shared<Pick>(indices);
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, indices.size());
  return Z;
}

sp<Variable> cross_entropy(const sp<Variable> &X,
                           const std::vector<size_t> &indices) {
  ASSERT(X->NumColumns() == indices.size(), X->Shape() << ", vs "
         << indices.size() << " indices");
  auto Z = std::make_shared<PickNegativeLogSoftmax>(indices);
  Z->AddParent(X);
  Z->gradient_ = Eigen::MatrixXd::Zero(1, indices.size());
  return Z;
}

sp<Variable> binary_cross_entropy(const sp<Variable> &X,
                                  const std::vector<bool> &flags) {
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

Eigen::Ref<Eigen::MatrixXd> Variable::Forward(sp_v1<Variable>
                                              *topological_order) {
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

void Variable::Backward(const sp_v1<Variable> &topological_order) {
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
  sp_v1<Variable> topological_order;
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

Eigen::Ref<Eigen::MatrixXd> Block::ref_value() {
  // If parent value hasn't been computed, neither has this node's.
  if (Parent(0)->ref_value().cols() == 0) { return Parent(0)->ref_value(); }

  if (start_row_ == 0 && num_rows_ == Parent(0)->NumRows()) {
    if (num_columns_ == 1) {
      return Parent(0)->ref_value().col(start_column_);
    } else {
      return Parent(0)->ref_value().middleCols(start_column_, num_columns_);
    }
  } else if (start_column_ == 0 && num_columns_ == Parent(0)->NumColumns()) {
    // Using row(i) gives type incompatible error...
    return Parent(0)->ref_value().middleRows(start_row_, num_rows_);
  } else {
    return Parent(0)->ref_value().block(start_row_, start_column_,
                                        num_rows_, num_columns_);
  }
}

Eigen::Ref<Eigen::MatrixXd> Block::ref_gradient() {
  if (start_row_ == 0 && num_rows_ == Parent(0)->NumRows()) {
    if (num_columns_ == 1) {
      return Parent(0)->ref_gradient().col(start_column_);
    } else {
      return Parent(0)->ref_gradient().middleCols(start_column_, num_columns_);
    }
  } else if (start_column_ == 0 && num_columns_ == Parent(0)->NumColumns()) {
    // Using row(i) gives type incompatible error...
    return Parent(0)->ref_gradient().middleRows(start_row_, num_rows_);
  } else {
    return Parent(0)->ref_gradient().block(start_row_, start_column_,
                                           num_rows_, num_columns_);
  }
}

void Block::SetBlock(size_t start_row, size_t start_column, size_t num_rows,
                     size_t num_columns) {
  start_row_ = start_row;
  start_column_ = start_column;
  num_rows_ = num_rows;
  num_columns_ = num_columns;
}

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
  Parent(0)->ref_gradient().noalias() += gradient_ *
                                         Parent(1)->ref_value().transpose();
  Parent(1)->ref_gradient().noalias() += Parent(0)->ref_value().transpose() *
                                         gradient_;
}

void MultiplyElementwise::PropagateGradient() {
  Parent(0)->ref_gradient().array() += gradient_.array() *
                                       Parent(1)->ref_value().array();
  Parent(1)->ref_gradient().array() += Parent(0)->ref_value().array() *
                                       gradient_.array();
}

void Sum::ComputeValue() {
  value_ = Parent(0)->ref_value();
  for (size_t i = 1; i < NumParents(); ++i) {
    value_ += Parent(i)->ref_value();
  }
}

void Sum::PropagateGradient() {
  for (size_t i = 0; i < NumParents(); ++i) {
    Parent(i)->ref_gradient() += gradient_;
  }
}

void ConcatenateVertical::ComputeValue() {
  value_.resize(gradient_.rows(), gradient_.cols());
  size_t start_row = 0;
  for (size_t i = 0; i < NumParents(); ++i) {
    value_.middleRows(start_row, Parent(i)->NumRows()) = Parent(i)->ref_value();
    start_row += Parent(i)->NumRows();
  }
}

void ConcatenateVertical::PropagateGradient() {
  size_t start_row = 0;
  for (size_t i = 0; i < NumParents(); ++i) {
    Parent(i)->ref_gradient() += gradient_.middleRows(start_row,
                                                      Parent(i)->NumRows());
    start_row += Parent(i)->NumRows();
  }
}

void ConcatenateHorizontal::ComputeValue() {
  value_.resize(gradient_.rows(), gradient_.cols());
  size_t start_column = 0;
  for (size_t i = 0; i < NumParents(); ++i) {
    value_.middleCols(start_column,
                      Parent(i)->NumColumns()) = Parent(i)->ref_value();
    start_column += Parent(i)->NumColumns();
  }
}

void ConcatenateHorizontal::PropagateGradient() {
  size_t start_column = 0;
  for (size_t i = 0; i < NumParents(); ++i) {
    Parent(i)->ref_gradient() += gradient_.middleCols(start_column,
                                                      Parent(i)->NumColumns());
    start_column += Parent(i)->NumColumns();
  }
}

void Dot::PropagateGradient() {
  Parent(0)->ref_gradient().noalias() += Parent(1)->ref_value() *
                                         gradient_.asDiagonal();
  Parent(1)->ref_gradient().noalias() += Parent(0)->ref_value() *
                                         gradient_.asDiagonal();
}

void Softmax::PropagateGradient() {
  Eigen::MatrixXd A = gradient_.cwiseProduct(value_);
  Parent(0)->ref_gradient() += A;
  Parent(0)->ref_gradient().noalias() -= value_ *
                                         A.colwise().sum().asDiagonal();
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
    value_(i) = -std::log(softmax_cache_(indices_[i], i));
  }
}

void PickNegativeLogSoftmax::PropagateGradient() {
  for (size_t i = 0; i < indices_.size(); ++i) {
    softmax_cache_(indices_[i], i) -= 1.0;
  }
  Parent(0)->ref_gradient().noalias() += softmax_cache_ *
                                         gradient_.asDiagonal();
}

void FlagNegativeLogistic::ComputeValue() {
  logistic_cache_ = Parent(0)->ref_value().unaryExpr(
      [](double x) { return 1 / (1 + exp(-x)); });
  value_.resize(1, flags_.size());
  for (size_t i = 0; i < flags_.size(); ++i) {
    value_(i) = (flags_[i]) ? -std::log(logistic_cache_(i)) :
                -std::log(1 - logistic_cache_(i));
  }
}

void FlagNegativeLogistic::PropagateGradient() {
  for (size_t i = 0; i < flags_.size(); ++i) {
    if (flags_[i]) { logistic_cache_(i) -= 1.0; }
  }
  Parent(0)->ref_gradient().noalias() += logistic_cache_.
                                         cwiseProduct(gradient_);
}

size_t Model::AddWeight(std::vector<size_t> row_partition_sizes,
                        std::vector<size_t> column_partition_sizes,
                        std::string initialization_method,
                        bool frozen) {
  size_t num_rows = std::accumulate(row_partition_sizes.begin(),
                                    row_partition_sizes.end(), 0);
  size_t num_columns = std::accumulate(column_partition_sizes.begin(),
                                       column_partition_sizes.end(), 0);
  Eigen::MatrixXd weight(num_rows, num_columns);
  size_t row_index = 0;
  for (size_t row_partition_size : row_partition_sizes) {
    size_t column_index = 0;
    for (size_t column_partition_size : column_partition_sizes) {
      weight.block(row_index, column_index,
                   row_partition_size, column_partition_size) =
          util_eigen::initialize(row_partition_size, column_partition_size,
                                 initialization_method);
      column_index += column_partition_size;
    }
    row_index += row_partition_size;
  }
  return AddWeight(weight, frozen);
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

sp<Input> Model::MakeInput(size_t weight_index) {
  auto search = active_weights_.find(weight_index);
  if (search != active_weights_.end()) { return search->second; }

  gradients_[weight_index].setZero();  // Clearing gradient
  active_weights_[weight_index] =
      std::make_shared<neural::Input>(&weights_[weight_index],
                                      &gradients_[weight_index]);
  if (!frozen_[weight_index]) { update_set_.insert(weight_index); }
  made_input_ = true;
  return active_weights_[weight_index];
}

sp<InputColumn> Model::MakeInputColumn(size_t weight_index,
                                       size_t column_index) {
  auto search1 = active_column_weights_.find(weight_index);
  if (search1 != active_column_weights_.end()) {
    auto search2 = search1->second.find(column_index);
    if (search2 != search1->second.end()) { return search2->second; }
  }

  gradients_[weight_index].col(column_index).setZero();  // Clearing gradient
  active_column_weights_[weight_index][column_index] =
      std::make_shared<neural::InputColumn>(&weights_[weight_index],
                                            &gradients_[weight_index],
                                            column_index);
  if (!frozen_[weight_index]) {
    update_column_set_[weight_index].insert(column_index);
  }
  made_input_ = true;
  return active_column_weights_[weight_index][column_index];
}

sp<Input> Model::MakeTemporaryInput(size_t temporary_index) {
  auto search = active_temporary_weights_.find(temporary_index);
  if (search != active_temporary_weights_.end()) { return search->second; }

  temporary_gradients_[temporary_index].setZero();  // Clearing gradient
  active_temporary_weights_[temporary_index] =
      std::make_shared<neural::Input>(&temporary_weights_[temporary_index],
                                      &temporary_gradients_[temporary_index]);
  made_temporary_input_ = true;
  return active_temporary_weights_[temporary_index];
}

void Model::ClearComputation() {
  update_set_.clear();
  update_column_set_.clear();
  temporary_weights_.clear();
  temporary_gradients_.clear();
  made_input_ = false;
  made_temporary_input_ = false;

  // This will now free the memory allocated to input nodes out of scope.
  active_weights_.clear();
  active_column_weights_.clear();
  active_temporary_weights_.clear();
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
  empty_observation_index_ = model_address_->AddWeight(dim_observation, 1,
                                                       "zero", true);
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

sp_v3<Variable> RNN::Transduce(const sp_v1<Variable> &observation_sequence,
                               const sp_v2<Variable> &initial_state_stack) {
  sp_v3<Variable> state_stack_sequence;
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

sp_v1<Variable> RNN::Batch(const sp_v2<Variable>
                           &unbatched_observation_sequences) {
  size_t max_length = 0;
  for (const auto &sequence : unbatched_observation_sequences) {
    max_length = std::max(max_length, sequence.size());
  }
  auto empty_observation = model_address_->MakeInput(empty_observation_index_);
  auto batch = [&](size_t position) {
    sp_v1<Variable> observations;
    for (size_t b = 0; b < unbatched_observation_sequences.size(); ++b) {
      const auto &sequence = unbatched_observation_sequences[b];
      observations.push_back(
          (position < unbatched_observation_sequences[b].size()) ?
          sequence[position] : empty_observation);
    }
    return hcat(observations);
  };

  sp_v1<Variable> batched_observation_sequence;
  for (size_t position = 0; position < max_length; ++position) {
    batched_observation_sequence.emplace_back(batch(position));
  }
  return batched_observation_sequence;
}

sp_v1<Variable> RNN::EncodeByFinalTop(const sp_v2<Variable>
                                      &unbatched_observation_sequences,
                                      bool reverse) {
  auto batched_observation_sequence = Batch(unbatched_observation_sequences);
  auto batched_state_sequence = Transduce(batched_observation_sequence);
  sp_v1<Variable> encodings;
  for (size_t i = 0; i < unbatched_observation_sequences.size(); ++i) {
    size_t length = unbatched_observation_sequences[i].size();
    auto encoding = column(batched_state_sequence[length - 1].back()[0], i);
    encodings.emplace_back(encoding);
  }
  return encodings;
}

sp_v2<Variable> RNN::ComputeNewStateStack(const sp<Variable> &observation,
                                          const sp_v2<Variable>
                                          &previous_state_stack,
                                          bool is_beginning) {
  ASSERT(is_beginning || previous_state_stack.size() > 0,
         "No previous state stack given even though not beginning a sequence");
  sp_v1<Variable> initial_state;
  if (is_beginning) {  // Starting a new sequence
    batch_size_ = observation->NumColumns();

    // Prepare initial state and dropout weights if necessary.
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
    if (dropout_rate_ > 0.0) { InitializeDropoutWeights(); }
  }
  sp_v2<Variable> state_stack;
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

void RNN::InitializeDropoutWeights() {
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

sp_v1<Variable> SimpleRNN::ComputeNewState(const sp<Variable> &observation,
                                           const sp_v1<Variable>
                                           &previous_state, size_t layer) {
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
    U_indices_.push_back(model_address_->AddWeight(
        {dim_state_, dim_state_, dim_state_, dim_state_},  {dim_in},
        "unit-variance"));
    V_indices_.push_back(model_address_->AddWeight(
        {dim_state_, dim_state_, dim_state_, dim_state_},  {dim_state_},
        "unit-variance"));
    b_indices_.push_back(model_address_->AddWeight(
        {dim_state_, dim_state_, dim_state_, dim_state_}, {1},
        "unit-variance"));
  }
}

sp_v1<Variable> LSTM::ComputeNewState(const sp<Variable> &observation,
                                      const sp_v1<Variable> &previous_state,
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

  auto stack_all = U * O + V * previous_H + b;
  auto raw_H = tanh(rows(stack_all, 0, dim_state_));
  auto stack_gates = logistic(rows(stack_all, dim_state_, 3 * dim_state_));

  auto gated_H = rows(stack_gates, 0, dim_state_) % raw_H;
  auto gated_previous_C = rows(stack_gates,
                               dim_state_, dim_state_) % previous_state[1];

  auto new_C = gated_H + gated_previous_C;
  auto new_H = rows(stack_gates, 2 * dim_state_, dim_state_) % tanh(new_C);
  return {new_H, new_C};
}

void LSTM::SetWeights(const Eigen::MatrixXd &U_weight,
                      const Eigen::MatrixXd &V_weight,
                      const Eigen::MatrixXd &b_weight, size_t layer) {
  *model_address_->weight(U_indices_[layer]) = U_weight;
  *model_address_->weight(V_indices_[layer]) = V_weight;
  *model_address_->weight(b_indices_[layer]) = b_weight;
}

}  // namespace neural
