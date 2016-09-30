#include "kalman_odometry.hpp"
#include <iostream>
#include <iomanip>
#include <math.h>
#include <fstream>

using namespace std;

int main( int argc , char *argv[] ){
    std::cout << std::scientific;
    Eigen::IOFormat f(5, Eigen::DontAlignCols, " ", "\n", "[", "]", "[", "]");

    std::cout << "OdometryFilter" << std::endl;

    // state and state covariance
    M<3, 1> x;
    M<3, 3> P;

    // marker positions in local frame x1,y1,x2,y2,x3,y3
    M<3, 2> Mref;
    M<3, 2> Mmeas;
    Mref << 0.1, 0.1, 0.1, -0.1, 0.0, 0.0;

    // init kalman
    OdometryFilter<3> kalman(0.1, 0.1, 0.1);
    kalman.unknown(-1);

    double time = 0.0;
    double ts = 0.01; // sample time
    double delay = 0.2; // delay + sample time of marker measurements
    double start_time = 2.0;
    double velocity_time = 4.0;
    double end_time = 2.0;
    double velocity = 0.8;

    double start_position = 1.0;;

    double current_vel = 0.0;
    int cnt = 0;
    int delay_cnt = int(delay/ts);

    std::vector<double> real_position((end_time+velocity_time+start_time)/ts);

    ofstream file;
    file.open("log.txt");

    // start
    while(true){
      // update time
      time = cnt*ts;
      // integrate simulated position
      if (cnt == 0){
        real_position[cnt] = start_position;
      } else {
        real_position[cnt] = real_position[cnt-1] + ts*current_vel;
      }
      // change velocity
      if (time == start_time){
        current_vel = velocity;
      }
      if (time == start_time+velocity_time){
        current_vel = 0.0;
      }
      // kalman
      std::cout << time << std::endl;
      kalman.observe_odo(time, current_vel, 0, 0);
      if (cnt%delay_cnt == 0){
        double pos = real_position[cnt-delay_cnt];
        Mmeas << 0.1+pos, 0.1, 0.1+pos, -0.1, pos, 0;
        kalman.observe_markers(time-delay, Mmeas, Mref, 0.1);
      }
      kalman.predict(time, x, P);
      file << x[0] << "\n";
      if (time >= start_time+velocity+end_time){
        break;
      }
      cnt++;
    }
    file.close();
    return 0;
}
