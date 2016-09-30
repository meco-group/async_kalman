#include "kalman_odometry.hpp"
#include <iostream>
#include <iomanip>

template<int N, int Ny>
void assertEqual(const M<N, Ny>& a, const M<N, Ny>& b) {
  bool res = a==b;

  if (!res) {
    std::cerr << "Assertion error " << a << "==" << b << "." << std::endl;
    exit(1);
  }
}

int main( int argc , char *argv[] ){
    std::cout << std::scientific;
    Eigen::IOFormat f(5, Eigen::DontAlignCols, " ", "\n", "[", "]", "[", "]");

    std::cout << "OdometryFilter" << std::endl;

    // state and state covariance
    M<3, 1> x2;
    M<3, 3> P2;

    // marker positions in local frame x1,y1,x2,y2,x3,y3
    M<3, 2> Mref;
    Mref << 0.1, 0.1, 0.1, -0.1, 0, 0;
    M<3, 2> Mmeas;

    // init kalman
    OdometryFilter<3> kalman(10000, 10000, 10000);
    kalman.unknown(0);

    kalman.predict(0, x2, P2); assertEqual(x2, {0, 0, 0});
    kalman.predict(1, x2, P2); assertEqual(x2, {0, 0, 0});

    kalman.observe_odo(0, 5,0,0);
    kalman.predict(1, x2, P2); assertEqual(x2, {5, 0, 0});
    kalman.predict(2, x2, P2); assertEqual(x2, {10, 0, 0});
    kalman.observe_odo(1, -5,0,0);
    kalman.predict(2, x2, P2); assertEqual(x2, {0, 0, 0});

    kalman.observe_odo(2, 0, 3, 0);
    kalman.predict(3, x2, P2); assertEqual(x2, {0, 3, 0});

    double omega = 0.1;
    double theta = omega*1;
    kalman.observe_odo(2, 0, 0, omega);
    kalman.predict(3, x2, P2); assertEqual(x2, {0, 0, theta});

    kalman.observe_odo(3, 1, 0, 0);
    kalman.predict(4, x2, P2); assertEqual(x2, {cos(theta), sin(theta), omega});

    {
      // init kalman
      OdometryFilter<3> kalman(1, 1, 1);
      kalman.unknown(0);


      Mmeas << 0.6, 1.1, 0.6, 0.9, 0.5, 1;
      kalman.observe_markers(1.0, Mmeas, Mref, 0.1);
      std::cout << "prediction at t=5s" << std::endl;
      kalman.predict(6, x2, P2);
      std::cout << x2.format(f) << std::endl;
      std::cout << P2.format(f) << std::endl;


      Mmeas << 0.7, 1.1, 0.7, 0.9, 0.6, 1;
      kalman.observe_markers(2.0, Mmeas, Mref, 0.1);
      kalman.predict(6, x2, P2);
      std::cout << x2.format(f) << std::endl;
      std::cout << P2.format(f) << std::endl;

      Mmeas << 0.8, 1.1, 0.8, 0.9, 0.7, 1;
      kalman.observe_markers(3.0, Mmeas, Mref, 0.1);
      kalman.predict(3, x2, P2);
      std::cout << x2.format(f) << std::endl;
      std::cout << P2.format(f) << std::endl;

      Mmeas << 0.8, 1.1, 0.8, 0.9, 0.7, 1;
      kalman.observe_markers(4.0, Mmeas, Mref, 0.1);
      kalman.predict(4, x2, P2);
      std::cout << x2.format(f) << std::endl;
      std::cout << P2.format(f) << std::endl;

      Mmeas << 0.8, 1.1, 0.8, 0.9, 0.7, 1;
      kalman.observe_markers(5.0, Mmeas, Mref, 0.1);
      kalman.predict(5, x2, P2);
      std::cout << x2.format(f) << std::endl;
      std::cout << P2.format(f) << std::endl;

    }

    // of.observe_odo(4, 5,0,0, 1,2,0.1);

    // of.predict(5, x2, P2);
    // std::cout << x2.format(f) << std::endl;
    // std::cout << P2.format(f) << std::endl;

}
