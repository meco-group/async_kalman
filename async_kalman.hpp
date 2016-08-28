#include <Eigen/Dense>
#include <vector>

template <int Nr, int Nc>
using M = typename Eigen::Matrix<double, Nr, Nc>;

typedef Eigen::MatrixXd Md;

// Need an 8-byte integer since libslicot0 is compiled with  fdefault-integer-8
typedef long long int f_int;

// https://eigen.tuxfamily.org/dox/classEigen_1_1LLT.html

void slicot_mb05nd(int n, double delta, const double* A, int lda,
                   double* ex, int ldex, double* exint, int ldexin,
                   double tol, f_int* iwork, double* dwork, int ld_work, int& info);

template<int N>
void expm(const M<N, N>& A, double t, M<N, N>& Ae, M<N, N>& Aei, std::vector<f_int>& iwork, std::vector<double>& dwork) {
  int info;
  slicot_mb05nd(N, t, A.data(), N, Ae.data(), N, Aei.data(), N, 1e-7, &iwork[0], &dwork[0], 2*N*N, info);
}


/**
  x' = Ax + Bu + w

  With Cov(w)=Q
*/
template <int N , int Nb>
class KalmanIntegrator {
public:
  KalmanIntegrator() : iwork_(2*N), dwork_(2*N*N*4) {
    F.setConstant(0);
  };

  void integrate(const M<N, N>& A, const M<N, Nb>& B, const M<N, N>&Q, double t,
                 M<N, N>& Ad, M<N, Nb>& Bd, M<N, N>& Qd) {
    expm(A, t, Ad, Aei, iwork_, dwork_);

    Bd = Aei*B;

    // trick Van Loan, 1978
    F.block(0,  0,  N, N) = -A;
    F.block(N, N, N, N) = A.transpose();
    F.block(0,  N, N, N) = Q;

    expm(F, t, Fd, Fei, iwork_, dwork_);
    Qd = Fd.block(N, N, N, N).transpose()*Fd.block(0, N, N, N);
  }

private:

  std::vector<f_int> iwork_;
  std::vector<double> dwork_;

  M<N, N> Aei; // placeholder for int expm(A) t
  M<2*N, 2*N> F;
  M<2*N, 2*N> Fd;
  M<2*N, 2*N> Fei;
};

template <int N, int Nu>
class KalmanPropagator {
public:
  KalmanPropagator() {}

  void propagate(const M<N, 1>& x, const M<N, N>& S, double t, M<N, 1>& xp, M<N, N>& Sp, const M<N, N>& A, const M<N, Nu>& B, const M<N, N>& Q, const M<Nu, 1>& u) {
    // Discretize system
    ki.integrate(A, B, Q, t, Ad, Bd, Qd);

    // Propagate state
    xp = Ad*x+Bd*u;

    // Propagate covariance
    Qf.compute(Qd);
    SQ = Qf.matrixL();

    Mm << (Ad*S).transpose(), SQ.transpose();
    qr.compute(Mm);
    Sp = qr.matrixQR().block(0, 0, N, N).template triangularView<Eigen::Upper>().transpose();
  }

private:

  M<2*N, N> Mm;
  Eigen::HouseholderQR< M<2*N, N> > qr;

  KalmanIntegrator<N, Nu> ki;

  M<N, N> Ad;
  M<N, Nu> Bd;
  M<N, N> Qd;
  M<N, N> SQ;

  Eigen::LLT< M<N, N> > Qf;
};

/**
+
+class KalmanObserver {
+public:
+  KalmanObserver(int n, int nb, int ny);
+
+  void observe(const M& x, const M& S, M& xp, M& Sp, const M& C, const M& D, const M& R, const M& z, const M& u);
+
+private:
+  int n_;
+  int nb_;
+  int ny_;
+
+  M Mm;
+  Eigen::HouseholderQR<M> qr;
+
+
+  M SR;
+
+  Eigen::LLT<M> Rf;
+
+  M zp;
+  M SZ;
+  M L;
+  M LZ;
+
+  Eigen::LLT<M> Sf;
+};


template <int Nx, int Ny, int Nu>
class SimpleMeasurement : public Measurment {
  M2<Ny, Nx> C_;
  M2<Ny, Nu> D_;
  M2<Ny, Ny> R_;

  void get(const M1<Nx> &x, const M2<Nx, Nx>& s, M2<Ny, Nx>& C, M2<Ny, Nu>& D, M2<Ny, Ny>& R) {
    C = C_;
    D = D_;
    R = R_;
  }
};
*/
