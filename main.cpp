#include "_kalman_filter.hpp"
#include <iostream>

int main ( int argc , char *argv[] ) {

  M<2, 2> A;
  A << 1  , 0.1,
       0.2, 1.1;
  M<2, 1> B;
  B << 3,
       4;
  M<2, 2> Q;
  Q << 2, 0.1,
      0.1, 3;


  KalmanIntegrator<2, 1> ki;

  M Ad(2, 2);
  M Bd(2, 1);
  M Qd(2, 2);
  ki.integrate(A, B, Q, 0.1, Ad, Bd, Qd);

  std::cout << Ad << std::endl;
  std::cout << Bd << std::endl;
  std::cout << Qd << std::endl;

  return 1;
}
