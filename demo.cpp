#include "kalman_odometry.hpp"
#include <iostream>
#include <iomanip>
#include <random>

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
  Eigen::IOFormat fm(5, Eigen::DontAlignCols, ",", ",", "[", "]", "[", "]");

  {
    OdometryFilter<3> of(1, 1, 0.1);

    of.unknown(-1);
    M<3, 2> Mref;
    Mref << 0.1, 0.1, 0.1, -0.1, 0, 0;
    M<3, 2> Mmeas = Mref;
    //of.observe_markers(-10000, Mmeas, Mref, pow(0.01, 2));
    M<3, 1> xp;
    M<3, 3> Pp;
    of.predict(3, xp, Pp);
  }
  int n = 200;
  OdometryFilter<3> of(1, 1, 0.1, 50);

  of.unknown(-1);

  M<3, 2> Mref;
  Mref << 0.1, 0.1, 0.1, -0.1, 0, 0;
  M<3, 2> Mmeas;
  M<2, 2> R, dR;
  M<2, 1> v;
  M<1, 3> I = M<1, 3>::Ones(1, 3);
  M<3, 1> xp;
  M<3, 3> Pp;

  double x, y, theta;

  std::default_random_engine generator;
  generator.seed(0);
  std::uniform_real_distribution<double> distribution(0.0, 6.0);

  for(int i=0;i<n;++i) {
    double t = distribution(generator);
    //t = i/(n+0.0)*6;
    std::cout << "====" << std::endl;
    std::cout << "time = " << t << std::endl;
    f(t, x, y, theta);
    transform_R_dR(theta, R, dR);

    v << x,y;
    Mmeas = (R*Mref.transpose()+v*I).transpose();
    of.observe_markers(t, Mmeas, Mref, pow(0.01, 2));
    of.predict(7, xp, Pp);
    std::cout << "x = " << xp.format(fm) << std::endl;
    std::cout << "P = " << Pp.format(fm) << std::endl;
  }

}
