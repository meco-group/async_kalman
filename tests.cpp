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

  auto e = kf.pop_event();
  e->active_measurement = &e->m.s;
  e->m.s.set(C, D, R, z);
  kf.add_event(1.0, e);

  M<2, 1> x2;
  M<2, 2> P2;

  kf.predict(-0.5, x2, P2);
  std::cout << x2.format(f) << std::endl;
  std::cout << P2.format(f) << std::endl;
  kf.predict(0.1, x2, P2);
  std::cout << x2.format(f) << std::endl;
  std::cout << P2.format(f) << std::endl;
  kf.predict(1.2, x2, P2);
  std::cout << x2.format(f) << std::endl;
  std::cout << P2.format(f) << std::endl;
  kf.predict(0, x2, P2);
  std::cout << x2.format(f) << std::endl;
  std::cout << P2.format(f) << std::endl;
  kf.predict(1, x2, P2);

  {
    std::cout << "KinematicKalmanFilter" << std::endl;

    class Measurements {
    public:
      SimpleMeasurement<2, 0, 1> s;
    };

    KinematicKalmanFilter<2, Measurements> kf(5);

    M<2, 1> x2;
    M<2, 2> P2;

    M<2, 1> x0;
    M<2, 2> P0 = M<2, 2>::Identity(2, 2);

    kf.reset(0.0, x0, P0);

    M<1, 2> C;
    C << 1, 0;

    M<1, 0> D;

    M<1, 1> z;
    z << 2;

    M<1, 1> R;
    R << 0.1;

    for (int i=1;i<8;++i) {

      auto e = kf.pop_event();
      e->active_measurement = &e->m.s;
      e->m.s.set(C, D, R, z);
      kf.add_event(i, e);

      kf.predict(i, x2, P2);
      std::cout << x2.format(f) << std::endl;
      std::cout << P2.format(f) << std::endl;
    }


  }

  return 1;
}
