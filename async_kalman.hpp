#include <Eigen/Dense>
#include <vector>

typedef Eigen::MatrixXd M;

// Need an 8-byte integer since libslicot0 is compiled with  fdefault-integer-8
typedef long long int f_int;

bool c2d(const M& A, double t, M& expA, M& expA_int);

/**
  x' = Ax + Bu + w

  With Cov(w)=Q
*/
class KalmanIntegrator {
public:
  KalmanIntegrator(int n);

  void integrate(const M& A, const M& B, const M&Q, double t, M& Ad, M& Bd, M& Qd);

private:

  std::vector<f_int> iwork_;
  std::vector<double> dwork_;

  int n_;
  M Aei; // placeholder for int expm(A) t
  M F;
  M Fd;
  M Fei;
};
