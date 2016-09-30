#include "kalman_odometry.hpp"
#include <iostream>

int main(){
    OdometryFilter<3>* kf;
    kf = new OdometryFilter<3>(0.1, 0.1, 0.1);


    return 0;
}
