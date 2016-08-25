#include "async_kalman.hpp"
#include <iostream>
#include <iomanip>
int main ( int argc , char *argv[] ) {

  std::cout << std::scientific;
  Eigen::IOFormat f(5, Eigen::DontAlignCols, " ", "\n", "[", "]", "[", "]");

  M A(2, 2);
  A << 1  , 0.1,
       0.2, 1.1;
  M B(2, 1);
  B << 3,
       4;
  M Q(2, 2);
  Q << 2, 0.1,
      0.1, 3;

  KalmanIntegrator ki(2);

  M Ad(2, 2);
  M Bd(2, 1);
  M Qd(2, 2);
  ki.integrate(A, B, Q, 0.1, Ad, Bd, Qd);

  std::cout << Ad.format(f) << std::endl;
  std::cout << Bd.format(f) << std::endl;
  std::cout << Qd.format(f) << std::endl;

  return 1;
}
