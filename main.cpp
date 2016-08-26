#include "async_kalman.hpp"
#include <iostream>
int main ( int argc , char *argv[] ) {
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

  std::cout << Ad << std::endl;
  std::cout << Bd << std::endl;
  std::cout << Qd << std::endl;
  M S(2,2);
  S << 3, 0,
      2, 7;

  M R(2,2);
  R << 2, 0.1,
      0.1, 2;

  Eigen::LLT<M> Qf(Qd);
  Eigen::LLT<M> Rf(R);
  M SQ = Qf.matrixL();
  std::cout << SQ << std::endl;

  M SR = Rf.matrixL();

  int n = 2;
  M x(2,1);
  x << 1 , 0.1;

  M u(1,1);
  u << 2;

  M D(2,1);
  D << 0.7,
       0.13;

  M C(2,2);
  C << 0.3  , 4,
      1, 2;

  // Propagate x
  x = Ad*x+Bd*u;
  std::cout << "here" << x << std::endl;
  // Propagate cov
  M Mm(2*n,n);


  Mm << (Ad*S).transpose(), SQ.transpose();

  std::cout << "here" << Mm << std::endl;

  Eigen::HouseholderQR<M> qr(2*n, n);

  qr.compute(Mm);
  M Qb = qr.matrixQR().block(0,0,n,n).triangularView<Eigen::Upper>().transpose();
  std::cout << Qb << std::endl;



  return 1;
}
