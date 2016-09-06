#include "kalman_odometry.hpp"
#include <iostream>
#include <iomanip>

int main( int argc , char *argv[] ){
    std::cout << std::scientific;
    Eigen::IOFormat f(5, Eigen::DontAlignCols, " ", "\n", "[", "]", "[", "]");

    std::cout << "OdometryFilter" << std::endl;

    // state and state covariance
    M<6, 1> x2;
    M<6, 6> P2;


    // marker positions in local frame x1,y1,x2,y2,x3,y3
    M<3, 2> Mref;
    Mref << 0.1, 0.1, 0.1, -0.1, 0, 0;
    M<3, 2> Mmeas;

    // init kalman
    OdometryFilter<3> kalman(1, 1, 0.1);
    kalman.unknown(0);

    // this works:
    kalman.predict(0, x2, P2);
    std::cout << x2.format(f) << std::endl;
    std::cout << P2.format(f) << std::endl;

    // this not:
    kalman.predict(1, x2, P2);
    std::cout << x2.format(f) << std::endl;
    std::cout << P2.format(f) << std::endl;




    // kalman.observe_odo(0, 5,0,0, 1,2,0.1);

    // Mmeas << 0.6, 1.1, 0.6, 0.9, 0.5, 1;
    // kalman.observe_markers(1.0, Mmeas, Mref, 0.1);
    // std::cout << "prediction at t=5s" << std::endl;
    // kalman.predict(5, x2, P2);
    // std::cout << x2.format(f) << std::endl;
    // std::cout << P2.format(f) << std::endl;



    // Mmeas << 0.7, 1.1, 0.7, 0.9, 0.6, 1;
    // of.observe_markers(2.0, Mmeas, Mref, 0.1);
    // of.predict(5, x2, P2);
    // std::cout << x2.format(f) << std::endl;
    // std::cout << P2.format(f) << std::endl;

    // Mmeas << 0.8, 1.1, 0.8, 0.9, 0.7, 1;
    // of.observe_markers(3.0, Mmeas, Mref, 0.1);
    // of.predict(5, x2, P2);
    // std::cout << x2.format(f) << std::endl;
    // std::cout << P2.format(f) << std::endl;

    // of.observe_odo(4, 5,0,0, 1,2,0.1);

    // of.predict(5, x2, P2);
    // std::cout << x2.format(f) << std::endl;
    // std::cout << P2.format(f) << std::endl;

}
