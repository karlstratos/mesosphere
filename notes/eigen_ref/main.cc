// Eigen::Ref can be used to refer to a matrix or a block without copying.

#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include <Eigen/Dense>

using namespace std;

Eigen::MatrixXd A = Eigen::MatrixXd::Ones(2, 4);
Eigen::Ref<Eigen::MatrixXd> RefA() { return A; }
Eigen::Ref<Eigen::MatrixXd> RefAcol(size_t c) { return A.col(c); }

double inverse_condition_number(const Eigen::Ref<Eigen::MatrixXd>& M) {
  const Eigen::VectorXd singular_values = M.jacobiSvd().singularValues();
  return singular_values(singular_values.size() - 1) / singular_values(0);
}

void set_to_zero(Eigen::Ref<Eigen::MatrixXd> M) { M.setZero(); }

int main() {
  std::cout << "A: " << std::endl << A << std::endl << std::endl;

  std::cout << "RefA() = Eigen::MatrixXd::Random(2, 4);" << std::endl;
  RefA() = Eigen::MatrixXd::Random(2, 4);
  std::cout << "A: " << std::endl << A << std::endl << std::endl;

  std::cout << "RefAcol(1) = Eigen::MatrixXd::Zero(2, 1);" << std::endl;
  RefAcol(1) = Eigen::MatrixXd::Zero(2, 1);
  std::cout << "A: " << std::endl << A << std::endl << std::endl;

  std::cout << "inverse_condition_number(A): no copy" << std::endl
            << inverse_condition_number(A) << std::endl << std::endl;

  std::cout << "inverse_condition_number(A.col(3)): no copy" << std::endl
            << inverse_condition_number(A.col(3)) << std::endl << std::endl;

  std::cout << "set_to_zero(A);" << std::endl;
  set_to_zero(A);
  std::cout << "A: " << std::endl << A << std::endl << std::endl;

  return 0;
}
