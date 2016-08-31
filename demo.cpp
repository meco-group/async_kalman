#include "kalman_ourbot.hpp"
#include <iostream>
#include <iomanip>


void f(double t, double&x, double&y, double&theta) {
  x = sin(t);
  y = sin(2*t);
  //x = t;
  //y = 2*t;
  theta = t;
}

void transform_R_dR(double theta, M<2,2>& A, M<2,2>& B) {
  double st = sin(theta);
  double ct = cos(theta);
  A << ct, -st, st, ct;
  B << -st, -ct, ct, -st;
}

int main ( int argc , char *argv[] ) {

  std::cout << std::scientific;
  
  Eigen::IOFormat fm(5, Eigen::DontAlignCols, " ", "\n", "[", "]", "[", "]");

  OdometryFilter<3> of(1, 1, 0.1);
  of.unknown(-1);

  M<3, 2> Mref;
  Mref << 0.1, 0.1, 0.1, -0.1, 0, 0;
  M<3, 2> Mmeas;
  M<2, 2> R, dR;
  M<2, 1> v;
  M<1, 3> I = M<1, 3>::Ones(1, 3);
  M<6, 1> xp;
  M<6, 6> Pp;

  double x, y, theta;

  double t = 0;
  while(true) {
    f(t, x, y, theta);
    transform_R_dR(theta, R, dR);

    v << x,y;
    std::cout << v << std::endl;
    Mmeas = (R*Mref.transpose()+v*I).transpose();
    of.observe_markers(t, Mmeas, Mref, 0.1);
    of.predict(t, xp, Pp);
    std::cout << xp.format(fm) << std::endl;
    std::cout << Pp.format(fm) << std::endl;
    t+= 0.01;

    if (t>=2) break;
  }

  return 1;
}
