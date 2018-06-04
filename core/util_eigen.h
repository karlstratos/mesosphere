// Author: Karl Stratos (me@karlstratos.com)
//
// Utility functions for the Eigen library.

#ifndef UTIL_EIGEN_H_
#define UTIL_EIGEN_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "util.h"

namespace util_eigen {

// Construct an Eigen matrix from std vectors representing rows.
inline Eigen::MatrixXd construct_matrix_from_rows(
    const std::vector<std::vector<double>> &rows) {
  size_t num_columns = rows[0].size();
  Eigen::MatrixXd matrix(rows.size(), num_columns);
  for (size_t i = 0; i < rows.size(); ++i) {
    ASSERT(rows[i].size() == num_columns, "Wrong matrix format");
    for (size_t j = 0; j < num_columns; ++j) { matrix(i, j) = rows[i][j]; }
  }
  return matrix;
}

// Column-wise stable log(sum_i e^{x_i}).
inline Eigen::MatrixXd logsumexp(const Eigen::MatrixXd& columns) {
  // logsumexp(x) = C + logsumexp(x - C) for any C. Choose C = max{x_i}.
  Eigen::RowVectorXd max_values = columns.colwise().maxCoeff();
  Eigen::ArrayXXd shifted_exp = (columns.rowwise() - max_values).array().exp();
  return max_values + shifted_exp.colwise().sum().log().matrix();
}

// Column-wise stable softmax.
inline Eigen::MatrixXd softmax(const Eigen::MatrixXd& columns) {
  // softmax(x) = softmax(x - C) for any C. Choose C = max{x_i}.
  Eigen::ArrayXXd shifted_exp = (columns.rowwise() -
                                 columns.colwise().maxCoeff()).array().exp();
  return shifted_exp.rowwise() / shifted_exp.colwise().sum();
}

// Initializes a matrix.
inline Eigen::MatrixXd initialize(size_t num_rows, size_t num_columns,
                                  std::string method) {
  Eigen::MatrixXd W;
  if (method == "unit-variance") {
    W = sqrt(3.0 / num_columns) * Eigen::MatrixXd::Random(num_rows,
                                                          num_columns);
  } else {
    ASSERT(false, "Unknown initialization method: " << method);
  }
  return W;
}

// Returns the dimensions of a matrix in string form.
template<typename EigenSparseOrDenseMatrix>
std::string dimension_string(const EigenSparseOrDenseMatrix& matrix) {
  return "(" + std::to_string(matrix.rows()) + " x " +
      std::to_string(matrix.cols()) + ")";
}

// Converts an unordered map (column -> {row: value}) to an Eigen sparse
// matrix.
template<typename T, typename EigenSparseMatrix>
EigenSparseMatrix convert_column_map(
    const std::unordered_map<size_t, std::unordered_map<size_t, T>>
    &column_map) {
  size_t num_rows = 0;
  size_t num_columns = 0;
  std::vector<Eigen::Triplet<T>> triplet_list;  // {(row, column, value)}
  size_t num_nonzeros = 0;
  for (const auto &column_pair: column_map) {
    num_nonzeros += column_pair.second.size();
  }
  triplet_list.reserve(num_nonzeros);
  for (const auto &column_pair: column_map) {
    size_t column = column_pair.first;
    if (column >= num_columns) { num_columns = column + 1; }
    for (const auto &row_pair: column_pair.second) {
      size_t row = row_pair.first;
      if (row >= num_rows) { num_rows = row + 1; }
      T value = row_pair.second;
      triplet_list.emplace_back(row, column, value);
    }
  }
  EigenSparseMatrix matrix(num_rows, num_columns);
  matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
  return matrix;
}

// Writes an Eigen dense matrix to a binary file.
template<typename EigenDenseMatrix>
void binary_write_matrix(const EigenDenseMatrix& matrix,
                         const std::string &file_path) {
  std::ofstream file(file_path, std::ios::out | std::ios::binary);
  ASSERT(file.is_open(), "Cannot open file: " << file_path);
  typename EigenDenseMatrix::Index num_rows = matrix.rows();
  typename EigenDenseMatrix::Index num_columns = matrix.cols();
  util_file::binary_write_primitive(num_rows, file);
  util_file::binary_write_primitive(num_columns, file);
  file.write(reinterpret_cast<const char *>(matrix.data()), num_rows *
             num_columns * sizeof(typename EigenDenseMatrix::Scalar));
}

// Reads an Eigen dense matrix from a binary file.
template<typename EigenDenseMatrix>
void binary_read_matrix(const std::string &file_path,
                        EigenDenseMatrix *matrix) {
  std::ifstream file(file_path, std::ios::in | std::ios::binary);
  ASSERT(file.is_open(), "Cannot open file: " << file_path);
  typename EigenDenseMatrix::Index num_rows;
  typename EigenDenseMatrix::Index num_columns;
  util_file::binary_read_primitive(file, &num_rows);
  util_file::binary_read_primitive(file, &num_columns);
  matrix->resize(num_rows, num_columns);
  file.read(reinterpret_cast<char*>(matrix->data()), num_rows *
            num_columns * sizeof(typename EigenDenseMatrix::Scalar));
}

// Computes the Moore–Penrose pseudo-inverse.
inline Eigen::MatrixXd compute_pseudoinverse(const Eigen::MatrixXd &matrix) {
  double tol = 1e-6;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU |
                                        Eigen::ComputeThinV);
  Eigen::VectorXd inverse_singular_values(svd.singularValues().size());
  for (size_t i = 0; i < svd.singularValues().size(); ++i) {
    if (svd.singularValues()(i) > tol) {
      inverse_singular_values(i) = 1.0 / svd.singularValues()(i);
    } else {
      inverse_singular_values(i) = 0.0;
    }
  }
  return svd.matrixV() * inverse_singular_values.asDiagonal()
      * svd.matrixU().transpose();
}

// Extends an orthonormal basis to subsume the given vector v.
inline void extend_orthonormal_basis(const Eigen::VectorXd &v,
                                     Eigen::MatrixXd *orthonormal_basis) {
  Eigen::MatrixXd orthogonal_projection =
      (*orthonormal_basis) * (*orthonormal_basis).transpose();
  Eigen::VectorXd projected_v = orthogonal_projection * v;
  Eigen::VectorXd new_direction = v - projected_v;
  new_direction /= new_direction.norm();

  orthonormal_basis->conservativeResize(orthonormal_basis->rows(),
                                        orthonormal_basis->cols() + 1);
  (*orthonormal_basis).col(orthonormal_basis->cols() - 1) = new_direction;
}

// Finds an orthonormal basis that spans the range of the matrix.
inline Eigen::MatrixXd find_range(const Eigen::MatrixXd &matrix) {
  ASSERT(matrix.cols() > 0, "Matrix has 0 columns");
  Eigen::MatrixXd orthonormal_basis;

  // Find the first basis element.
  orthonormal_basis = matrix.col(0);
  orthonormal_basis /= orthonormal_basis.norm();

  // Find the remaining basis elements.
  for (size_t col = 1; col < matrix.cols(); ++col) {
    if (col >= matrix.rows()) {
      // The dimension of the range is at most the number of rows.
      break;
    }
    extend_orthonormal_basis(matrix.col(col), &orthonormal_basis);
  }
  return orthonormal_basis;
}

// Computes principal component analysis (PCA). The format of the input
// matrix is: rows = samples, columns = dimensions.
inline void compute_pca(const Eigen::MatrixXd &original_sample_rows,
                        Eigen::MatrixXd *rotated_sample_rows,
                        Eigen::MatrixXd *rotation_matrix,
                        Eigen::VectorXd *variances) {
  // Center each dimension (column).
  Eigen::MatrixXd centered = original_sample_rows;
  Eigen::VectorXd averages = centered.colwise().sum() / centered.rows();
  for (size_t i = 0; i < centered.cols(); ++i) {
    Eigen::VectorXd average_column =
        Eigen::VectorXd::Constant(centered.rows(), averages(i));
    centered.col(i) -= average_column;
  }

  // Perform SVD.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU |
                                        Eigen::ComputeThinV);

  // Set values.
  *rotated_sample_rows =
      svd.matrixU() * svd.singularValues().asDiagonal();
  *rotation_matrix = svd.matrixV();
  (*variances).resize(svd.singularValues().size());
  for (size_t i = 0; i < svd.singularValues().size(); ++i) {
    (*variances)(i) =
        pow(svd.singularValues()(i), 2) / (centered.rows() - 1);
  }
}

// Generates a random projection matrix.
inline Eigen::MatrixXd generate_random_projection(size_t original_dimension,
                                                  size_t reduced_dimension,
                                                  size_t seed) {
  Eigen::MatrixXd projection_matrix(original_dimension, reduced_dimension);
  std::default_random_engine engine(seed);
  // Indyk and Motwani (1998)
  std::normal_distribution<double> normal(0.0, 1.0 / reduced_dimension);
  for (size_t row = 0; row < original_dimension; ++row) {
    for (size_t col = 0; col < reduced_dimension; ++col) {
      projection_matrix(row, col) = normal(engine);
    }
  }
  return projection_matrix;
}

// Returns true if two Eigen dense matrices are close in value.
template<typename EigenDenseMatrix>
bool check_near(const EigenDenseMatrix& matrix1,
                const EigenDenseMatrix& matrix2, double error_threshold) {
  if (matrix1.rows() != matrix2.rows() ||
      matrix2.cols() != matrix2.cols()) { return false; }
  for (size_t row = 0; row < matrix1.rows(); ++row) {
    for (size_t col = 0; col < matrix1.cols(); ++col) {
      if (fabs(matrix1(row, col) - matrix2(row, col))
          > error_threshold) { return false; }
    }
  }
  return true;
}

// Returns true if two Eigen dense matrices are close in absolute value.
template<typename EigenDenseMatrix>
bool check_near_abs(const EigenDenseMatrix& matrix1,
                    const EigenDenseMatrix& matrix2,
                    double error_threshold) {
  if (matrix1.rows() != matrix2.rows() ||
      matrix2.cols() != matrix2.cols()) { return false; }
  for (size_t row = 0; row < matrix1.rows(); ++row) {
    for (size_t col = 0; col < matrix1.cols(); ++col) {
      if (fabs(fabs(matrix1(row, col)) - fabs(matrix2(row, col)))
          > error_threshold) { return false; }
    }
  }
  return true;
}

// Computes the KL divergence of distribution 2 from distribution 1.
// WARNING: Assign distribution variables before passing them, e.g.,
//          don't do "kl_divergence(M.col(0), M.col(1));
template<typename EigenDenseVector>
double kl_divergence(const EigenDenseVector& distribution1,
                     const EigenDenseVector& distribution2) {
  double kl = 0.0;
  for (size_t i = 0; i < distribution1.size(); ++i) {
    if (distribution2(i) <= 0.0) {
      ASSERT(distribution1(i) <= 0.0, "KL is undefined");
    }
    if (distribution1(i) > 0.0) {
      kl += distribution1(i) * (log(distribution1(i)) -
                                log(distribution2(i)));
    }
  }
  return kl;
}

// http://eigen.tuxfamily.org/dox-devel/TopicCustomizing_NullaryExpr.html#title1
// -----------------------------------------------------------------------------
//
// Nullary-functor storing references to the input matrix and to the two arrays
// of indices, and implementing the required operator()(i,j):
template<class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor {
  const ArgType &m_arg;
  const RowIndexType &m_rowIndices;
  const ColIndexType &m_colIndices;
 public:
  typedef Eigen::Matrix<typename ArgType::Scalar,
                        RowIndexType::SizeAtCompileTime,
                        ColIndexType::SizeAtCompileTime,
                        ArgType::Flags&Eigen::RowMajorBit?Eigen::RowMajor:\
                        Eigen::ColMajor,
                        RowIndexType::MaxSizeAtCompileTime,
                        ColIndexType::MaxSizeAtCompileTime> MatrixType;
  indexing_functor(const ArgType& arg, const RowIndexType& row_indices,
                   const ColIndexType& col_indices)
      : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
  {}
  const typename ArgType::Scalar& operator() (Eigen::Index row,
                                              Eigen::Index col) const {
    return m_arg(m_rowIndices[row], m_colIndices[col]);
  }
};

// indexing(A,rows,cols) function creates the nullary expression.
template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType,RowIndexType,ColIndexType>,
                      typename indexing_functor<ArgType,
                                                RowIndexType,
                                                ColIndexType>::MatrixType>
indexing(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices,
         const ColIndexType& col_indices) {
  typedef indexing_functor<ArgType,RowIndexType,ColIndexType> Func;
  typedef typename Func::MatrixType MatrixType;
  return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(),
                                 Func(arg.derived(), row_indices, col_indices));
}
//------------------------------------------------------------------------------

}  // namespace util_eigen

#endif  // UTIL_EIGEN_H_
