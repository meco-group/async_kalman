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

  M x(2, 1);
  x << 1 , 0.1;

  M S(2, 2);
  S << 3, 0,
      2, 7;

  M u(1, 1);
  u << 2;

  KalmanPropagator ks(2, 1);
  M xp(2, 1);
  M Sp(2, 2);
  ks.propagate(x, S, 0.1, xp, Sp, A, B, Q, u);

  std::cout << xp.format(f) << std::endl;
  std::cout << Sp.format(f) << std::endl;

  M D(2,1);
  D << 0.7,
       0.13;

  M C(2,2);
  C << 0.3  , 4,
      1, 2;

  M z(2,1);
  z << 1 , 0.2;

  M R(2,2);
  R << 2, 0.1,
      0.1, 2;

  KalmanObserver km(2, 1, 2);
  km.observe(x, S, xp, Sp, C, D, R, z, u);

  std::cout << xp.format(f) << std::endl;
  std::cout << (Sp*Sp.transpose()).format(f) << std::endl;

  KalmanObserver km2(2, 1, 1);
  km2.observe(x, S, xp, Sp, C.row(0), D.row(0), R.row(0).col(0), z.row(0), u);

  std::cout << xp.format(f) << std::endl;
  std::cout << (Sp*Sp.transpose()).format(f) << std::endl;

  return 1;
}
