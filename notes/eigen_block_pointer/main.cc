// Demonstrates how we can manipulate pointers to Eigen matrix blocks.

#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include <Eigen/Dense>

using namespace std;


void add_ten(Eigen::MatrixXd::ColXpr c) {
  c += 10 * Eigen::MatrixXd::Ones(c.rows(), 1);
}

int main() {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(2, 4);
  std::cout << "A: " << std::endl << A << std::endl << std::endl;

  std::cout << "A.col(3): " << std::endl << A.col(3) << std::endl << std::endl;

  Eigen::MatrixXd::ColXpr c3 = A.col(3);
  std::cout << "Eigen::MatrixXd::ColXpr c3" << std::endl << c3 << std::endl
            << std::endl;

  std::cout << "c3 += Eigen::MatrixXd::Ones(2, 1)" << std::endl;
  c3 += Eigen::MatrixXd::Ones(2, 1);
  std::cout << "Eigen::MatrixXd::ColXpr c3" << std::endl << c3 << std::endl
            << std::endl;
  std::cout << "A: " << std::endl << A << std::endl << std::endl;

  std::cout << "add_ten(c3)" << std::endl;
  add_ten(c3);
  std::cout << "Eigen::MatrixXd::ColXpr c3" << std::endl << c3 << std::endl
            << std::endl;
  std::cout << "A: " << std::endl << A << std::endl << std::endl;

  Eigen::MatrixXd v = Eigen::MatrixXd::Random(2, 1);
  std::cout << "v: " << std::endl << v << std::endl << std::endl;

  std::cout << "v += c3" << std::endl;
  v += c3;
  std::cout << "v: " << std::endl << v << std::endl << std::endl;

  std::cout << "c3 += v" << std::endl;
  c3 += v;
  std::cout << "Eigen::MatrixXd::ColXpr c3" << std::endl << c3 << std::endl
            << std::endl;
  std::cout << "A: " << std::endl << A << std::endl << std::endl;

  return 0;
}
