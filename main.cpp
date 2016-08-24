#include "async_kalman.hpp"
#include <iostream>
int main ( int argc , char *argv[] ) {
  M A(2, 2);
  A << 1  , 0.1,
       0.1, 1;
  M B(2, 1);
  B << 3,
       4;
  std::cout << c2d(A, B, 0.1) << std::endl;
  return 1;
}
