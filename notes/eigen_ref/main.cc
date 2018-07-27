// Eigen::Ref can be used to refer to a matrix *or* a block without copying.
// The referred matrix must have its memory fully allocated: cannot be resized.

#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include <Eigen/Dense>

using namespace std;

Eigen::MatrixXd A = Eigen::MatrixXd::Random(1, 2);
Eigen::Ref<Eigen::MatrixXd> ref_from_address(Eigen::MatrixXd *address) {
  return *address;
}

int main() {
  std::cout << "A: " << std::endl << A << std::endl << std::endl;

  Eigen::MatrixXd *A_address = &A;
  std::cout << "A_address: " << A_address << std::endl;

  std::cout << "A = Eigen::MatrixXd::Random(2, 4);" << std::endl;
  A = Eigen::MatrixXd::Random(2, 4);
  std::cout << A << std::endl << std::endl;

  std::cout << "ref_from_address(A_address) =  Eigen::MatrixXd::Random(2, 4);"
            << " // Okay, since shape matches" << std::endl;
  ref_from_address(A_address) =  Eigen::MatrixXd::Random(2, 4);
  std::cout << A << std::endl << std::endl;

  std::cout << "ref_from_address(A_address).col(2) = "
            << "Eigen::MatrixXd::Zero(2, 1); // Also okay" << std::endl;
  ref_from_address(A_address).col(2) =  Eigen::MatrixXd::Zero(2, 1);
  std::cout << A << std::endl << std::endl;

  std::cout << "// ref_from_address(A_address) =  "
            << "Eigen::MatrixXd::Ones(2, 10); // ERROR! "
            << "\"DenseBase::resize() does not actually allow to resize.\""
            << std::endl << std::endl;
  // ref_from_address(A_address) =  Eigen::MatrixXd::Ones(2, 10);

  std::cout << "// First need to directly resize NOT USING REF. This destroys "
            << "coeffs if the number of coeffs is different." << std::endl;
  std::cout << "A_address->resize(2, 10);  " << std::endl;
  A_address->resize(2, 10);
  // ref_from_address(A_address).resize(2, 4);  // This gives error.
  std::cout << A << std::endl << std::endl;

  std::cout << "ref_from_address(A_address) =  "
            << "Eigen::MatrixXd::Ones(2, 10); // Now okay" << std::endl;
  ref_from_address(A_address) =  Eigen::MatrixXd::Ones(2, 10);
  std::cout << A << std::endl << std::endl;

  std::cout << "ref_from_address(A_address) << 1, 3, 5, 7, 9, 11, 13, 15, 17, "
            << "19, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20; // Also okay"
            << std::endl;
  ref_from_address(A_address) <<
      1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20;
  std::cout << A << std::endl << std::endl;

  std::cout << "// Aside: resizing is conservative when the number of coeffs "
            << "remains the same." << std::endl;
  std::cout << "A_address->resize(2, 10);  " << std::endl;
  A_address->resize(4, 5);
  std::cout << A << std::endl << std::endl;

  std::cout << "// Coefficient access" << std::endl;
  std::cout << "ref_from_address(A_address)(1, 2) = "
            << ref_from_address(A_address)(1, 2) << std::endl;
  std::cout << "ref_from_address(A_address).data()[9] = "
            << ref_from_address(A_address).data()[9] << std::endl;

  return 0;
}
