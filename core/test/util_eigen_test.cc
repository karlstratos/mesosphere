// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include <vector>

#include "../util_eigen.h"

TEST(LogSumExp, Test) {
  Eigen::MatrixXd columns(3, 2);
  columns << 1, 1, 2, 2, 3, 3;
  Eigen::MatrixXd output = util_eigen::logsumexp(columns);
  EXPECT_NEAR(3.4076, output(0), 1e-4);
  EXPECT_NEAR(3.4076, output(1), 1e-4);

  Eigen::MatrixXd columns_large = columns.array() + 1000000;
  Eigen::MatrixXd output_large = util_eigen::logsumexp(columns_large);
  EXPECT_TRUE(isinf(columns_large.array().exp().colwise().sum().log()(0)));
  EXPECT_TRUE(isinf(columns_large.array().exp().colwise().sum().log()(1)));
  EXPECT_NEAR(1000003.4076, output_large(0), 1e-4);
  EXPECT_NEAR(1000003.4076, output_large(1), 1e-4);
}

TEST(Softmax, Test) {
  Eigen::MatrixXd columns(3, 2);
  columns << 1, 1, 2, 2, 3, 3;
  Eigen::MatrixXd output = util_eigen::softmax(columns);
  EXPECT_NEAR(0.0900, output(0, 0), 1e-4);
  EXPECT_NEAR(0.2447, output(1, 0), 1e-4);
  EXPECT_NEAR(0.6652, output(2, 0), 1e-4);
  EXPECT_NEAR(0.0900, output(0, 1), 1e-4);
  EXPECT_NEAR(0.2447, output(1, 1), 1e-4);
  EXPECT_NEAR(0.6652, output(2, 1), 1e-4);

  Eigen::MatrixXd output_large_positive = util_eigen::softmax(
      columns.array() + 1000000);
  EXPECT_NEAR(0.0900, output_large_positive(0, 0), 1e-4);
  EXPECT_NEAR(0.2447, output_large_positive(1, 0), 1e-4);
  EXPECT_NEAR(0.6652, output_large_positive(2, 0), 1e-4);
  EXPECT_NEAR(0.0900, output_large_positive(0, 1), 1e-4);
  EXPECT_NEAR(0.2447, output_large_positive(1, 1), 1e-4);
  EXPECT_NEAR(0.6652, output_large_positive(2, 1), 1e-4);

  Eigen::MatrixXd output_large_negative = util_eigen::softmax(
      columns.array() - 1000000);
  EXPECT_NEAR(0.0900, output_large_negative(0, 0), 1e-4);
  EXPECT_NEAR(0.2447, output_large_negative(1, 0), 1e-4);
  EXPECT_NEAR(0.6652, output_large_negative(2, 0), 1e-4);
  EXPECT_NEAR(0.0900, output_large_negative(0, 1), 1e-4);
  EXPECT_NEAR(0.2447, output_large_negative(1, 1), 1e-4);
  EXPECT_NEAR(0.6652, output_large_negative(2, 1),1e-4);
}

TEST(DimensionString, Test) {
  Eigen::MatrixXd matrix1 = Eigen::MatrixXd::Random(42, 84);
  EXPECT_EQ("(42 x 84)", util_eigen::dimension_string(matrix1));

  Eigen::SparseMatrix<double> matrix2(10, 2);
  EXPECT_EQ("(10 x 2)", util_eigen::dimension_string(matrix2));
}

class UtilEigen : public testing::Test {
 protected:
  // (1 x 3): short and fat
  // (3 x 3): square
  // (5 x 3): tall and thin
  std::vector<size_t> list_num_rows_ = {1, 3, 5};
  std::vector<size_t> list_num_columns_ = {3};
  double tol_ = 1e-10;
};

TEST_F(UtilEigen, WritingReadingDoubleMatrix) {
  for (size_t num_rows : list_num_rows_) {
    for (size_t num_columns : list_num_columns_) {
      std::string file_path = "util_eigen_test_temp";
      Eigen::MatrixXd matrix1 = Eigen::MatrixXd::Random(num_rows,
                                                        num_columns);
      util_eigen::binary_write_matrix(matrix1, file_path);
      Eigen::MatrixXd matrix2;
      util_eigen::binary_read_matrix(file_path, &matrix2);
      EXPECT_EQ(num_rows, matrix2.rows());
      EXPECT_EQ(num_columns, matrix2.cols());
      for (size_t row = 0; row < num_rows; ++row) {
        for (size_t col = 0; col < num_columns; ++col) {
          EXPECT_NEAR(matrix1(row, col), matrix2(row, col), tol_);
        }
      }
      remove(file_path.c_str());
    }
  }
}

TEST_F(UtilEigen, WritingReadingIntMatrix) {
  for (size_t num_rows : list_num_rows_) {
    for (size_t num_columns : list_num_columns_) {
      std::string file_path = "util_eigen_test_temp";
      Eigen::MatrixXi matrix1 = Eigen::MatrixXi::Random(num_rows,
                                                        num_columns);
      util_eigen::binary_write_matrix(matrix1, file_path);
      Eigen::MatrixXi matrix2;
      util_eigen::binary_read_matrix(file_path, &matrix2);
      EXPECT_EQ(num_rows, matrix2.rows());
      EXPECT_EQ(num_columns, matrix2.cols());
      for (size_t row = 0; row < num_rows; ++row) {
        for (size_t col = 0; col < num_columns; ++col) {
          EXPECT_EQ(matrix1(row, col), matrix2(row, col));
        }
      }
      remove(file_path.c_str());
    }
  }
}

TEST_F(UtilEigen, WritingReadingFloatVector) {
  size_t length = 3;
  std::string file_path = "util_eigen_test_temp";
  Eigen::VectorXf vector1 = Eigen::VectorXf::Random(length);
  util_eigen::binary_write_matrix(vector1, file_path);
  Eigen::VectorXf vector2;
  util_eigen::binary_read_matrix(file_path, &vector2);
  EXPECT_EQ(length, vector2.rows());
  for (size_t i = 0; i < length; ++i) {
    EXPECT_NEAR(vector1(i), vector2(i), tol_);
  }
  remove(file_path.c_str());
}

TEST_F(UtilEigen, PseudoInverse) {
  for (size_t num_rows : list_num_rows_) {
    for (size_t num_columns : list_num_columns_) {
      Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(num_rows,
                                                       num_columns);
      Eigen::MatrixXd matrix_pseudoinverse =
          util_eigen::compute_pseudoinverse(matrix);
      Eigen::MatrixXd product = (num_rows >= num_columns) ?
                                matrix_pseudoinverse * matrix :
                                matrix * matrix_pseudoinverse;
      size_t rank = std::min(num_rows, num_columns);
      for (size_t row = 0; row < rank; ++row) {
        for (size_t col = 0; col < rank; ++col) {
          if (row == col) {
            EXPECT_NEAR(1.0, product(row, col), tol_);
          } else {
            EXPECT_NEAR(0.0, product(row, col), tol_);
          }
        }
      }
    }
  }
}

TEST_F(UtilEigen, FindMatrixRange) {
  for (size_t num_rows : list_num_rows_) {
    for (size_t num_columns : list_num_columns_) {
      Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(num_rows,
                                                       num_columns);
      Eigen::MatrixXd orthonormal_basis =  util_eigen::find_range(matrix);

      // Check if orthonormal.
      Eigen::MatrixXd inner_product =
          orthonormal_basis.transpose() * orthonormal_basis;
      size_t rank = std::min(num_rows, num_columns);
      for (size_t row = 0; row < rank; ++row) {
        for (size_t col = 0; col < rank; ++col) {
          if (row == col) {
            EXPECT_NEAR(1.0, inner_product(row, col), tol_);
          } else {
            EXPECT_NEAR(0.0, inner_product(row, col), tol_);
          }
        }
      }

      // Check if the same range.
      Eigen::VectorXd v = Eigen::VectorXd::Random(num_rows);
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU);
      Eigen::MatrixXd projection1 =
          svd.matrixU() * svd.matrixU().transpose() * v;
      Eigen::MatrixXd projection2 =
          orthonormal_basis * orthonormal_basis.transpose() * v;
      for (size_t i = 0; i < projection1.size(); ++i) {
        EXPECT_NEAR(projection1(i), projection2(i), tol_);
      }
    }
  }
}

TEST_F(UtilEigen, ComputePCA) {
  // The "line-in-plane" example.
  Eigen::MatrixXd example(3, 2);
  example(0,0) = -1.0;
  example(0,1) = 1.0;
  example(1,0) = 0.0;
  example(1,1) = 1.0;
  example(2,0) = 1.0;
  example(2,1) = 1.0;

  Eigen::MatrixXd rotated_sample_rows;
  Eigen::MatrixXd rotation_matrix;
  Eigen::VectorXd variances;
  util_eigen::compute_pca(example, &rotated_sample_rows,
                          &rotation_matrix, &variances);

  Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(2, 2);
  ASSERT_TRUE(util_eigen::check_near(rotation_matrix, eye, tol_));
  EXPECT_NEAR(rotated_sample_rows(0, 0), -1.0, tol_);
  EXPECT_NEAR(rotated_sample_rows(0, 1), 0.0, tol_);
  EXPECT_NEAR(rotated_sample_rows(1, 0), 0.0, tol_);
  EXPECT_NEAR(rotated_sample_rows(1, 1), 0.0, tol_);
  EXPECT_NEAR(rotated_sample_rows(2, 0), 1.0, tol_);
  EXPECT_NEAR(rotated_sample_rows(2, 1), 0.0, tol_);
  EXPECT_NEAR(variances(0), 1.0, tol_);
  EXPECT_NEAR(variances(1), 0.0, tol_);
}

TEST(Miscellaneous, CheckNearDenseMatrices) {
  double error_threshold = 1e-10;
  Eigen::MatrixXd random_matrix1 = Eigen::MatrixXd::Random(10, 10);
  Eigen::MatrixXd random_matrix2 = Eigen::MatrixXd::Random(10, 10);
  ASSERT_FALSE(util_eigen::check_near(random_matrix1, random_matrix2,
                                      error_threshold));
  random_matrix2 = random_matrix1;
  ASSERT_TRUE(util_eigen::check_near(random_matrix1, random_matrix2,
                                     error_threshold));
  random_matrix2(0,0) += 2 * error_threshold;
  ASSERT_FALSE(util_eigen::check_near(random_matrix1, random_matrix2,
                                      error_threshold));
}

TEST(Miscellaneous, CheckNearDenseMatricesAbs) {
  double error_threshold = 1e-10;
  Eigen::MatrixXd random_matrix1 = Eigen::MatrixXd::Random(10, 10);
  Eigen::MatrixXd random_matrix2 = - random_matrix1;
  ASSERT_FALSE(util_eigen::check_near(random_matrix1, random_matrix2,
                                      error_threshold));
  ASSERT_TRUE(util_eigen::check_near_abs(random_matrix1, random_matrix2,
                                         error_threshold));
}

TEST(Indexing, Test) {
  //  1   2   3      [2 1] [0 2 1 1]       =>         7   9   8   8
  //  4   5   6                                       4   6   5   5
  //  7   8   9
  Eigen::MatrixXd A(3, 3);
  A << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  std::vector<size_t> r_vector = {2, 1};
  Eigen::Map<Eigen::Array<size_t, Eigen::Dynamic, 1>> r(r_vector.data(),
                                                        r_vector.size());
  std::vector<int> c_vector = {0, 2, 1, 1};
  Eigen::Map<Eigen::ArrayXi> c(c_vector.data(), c_vector.size());
  Eigen::MatrixXd B = util_eigen::indexing(A, r, c);
  EXPECT_EQ(7, B(0, 0));
  EXPECT_EQ(4, B(1, 0));
  EXPECT_EQ(9, B(0, 1));
  EXPECT_EQ(6, B(1, 1));
  EXPECT_EQ(8, B(0, 2));
  EXPECT_EQ(5, B(1, 2));
  EXPECT_EQ(8, B(0, 3));
  EXPECT_EQ(5, B(1, 3));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
