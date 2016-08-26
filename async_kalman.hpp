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

class KalmanPropagator {
public:
  KalmanPropagator(int n, int nb);

  void propagate(const M& x, const M& S, double t, M& xp, M& Sp, const M& A, const M& B, const M& Q, const M& u);

private:
  int n_;
  int nb_;

  M Mm;
  Eigen::HouseholderQR<M> qr;

  KalmanIntegrator ki;

  M Ad;
  M Bd;
  M Qd;
  M SQ;

  Eigen::LLT<M> Qf;
};

class KalmanObserver {
public:
  KalmanObserver(int n, int nb, int ny);

  void observe(const M& x, const M& S, M& xp, M& Sp, const M& C, const M& D, const M& R, const M& z, const M& u);

private:
  int n_;
  int nb_;
  int ny_;

  M Mm;
  Eigen::HouseholderQR<M> qr;


  M SR;

  Eigen::LLT<M> Rf;

  M zp;
  M SZ;
  M L;
  M LZ;

  Eigen::LLT<M> Sf;
};
