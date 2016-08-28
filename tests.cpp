#include "async_kalman.hpp"
#include <iostream>
#include <iomanip>
int main ( int argc , char *argv[] ) {

  std::cout << std::scientific;
  Eigen::IOFormat f(5, Eigen::DontAlignCols, " ", "\n", "[", "]", "[", "]");

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

  M<2, 2> Ad;
  M<2, 1> Bd;
  M<2, 2> Qd;
  ki.integrate(A, B, Q, 0.1, Ad, Bd, Qd);

  std::cout << Ad.format(f) << std::endl;
  std::cout << Bd.format(f) << std::endl;
  std::cout << Qd.format(f) << std::endl;

  M<2, 1> x;
  x << 1 , 0.1;

  M<2, 2> S;
  S << 3, 0,
       2, 7;

  M<1, 1> u(1, 1);
  u << 2;

  KalmanPropagator<2, 1> ks;
  M<2, 1> xp(2, 1);
  M<2, 2> Sp(2, 2);
  ks.propagate(x, S, 0.1, xp, Sp, A, B, Q, u);

  std::cout << xp.format(f) << std::endl;
  std::cout << Sp.format(f) << std::endl;


  M<2, 1> D;
  D << 0.7,
       0.13;

  M<2, 2> C;
  C << 0.3  , 4,
      1, 2;

  M<2, 1> z;
  z << 1 , 0.2;

  M<2, 2> R;
  R << 2, 0.1,
      0.1, 2;

  KalmanObserver<2, 1, 2> km;
  km.observe(x, S, xp, Sp, C, D, R, z, u);

  std::cout << xp.format(f) << std::endl;
  std::cout << (Sp*Sp.transpose()).format(f) << std::endl;

  KalmanObserver<2, 1, 1> km2;
  km2.observe(x, S, xp, Sp, C.row(0), D.row(0), R.row(0).col(0), z.row(0), u);

  std::cout << xp.format(f) << std::endl;
  std::cout << (Sp*Sp.transpose()).format(f) << std::endl;

  M<2, 2> P = S*S.transpose();

  class Measurements {
  public:
    SimpleMeasurement<2, 1, 2> s;
  };

  KalmanFilter<2, 1, Measurements> kf(A, B, Q);
  kf.reset(0.0, x, P);

  auto e = kf.create_event(1.0);
  e->active_measurement = &e->m.s;
  e->m.s.set(C, D, R, z);

  M<2, 1> x2;
  M<2, 2> P2;

  kf.predict(3, x2, P2);

  return 1;
}
