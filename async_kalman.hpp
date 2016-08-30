#include <Eigen/Dense>
#include <vector>
#include <map>
#include <iostream>

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

template <int N, int Nu, int Ny>
class KalmanObserver {
public:
  KalmanObserver() {};

  void observe(const M<N, 1>& x, const M<N, N>& S, M<N, 1>& xp, M<N, N>& Sp, const M<Ny, N>& C, const M<Ny, Nu>& D, const M<Ny, Ny>& R, const M<Ny, 1>& z, const M<Nu, 1>& u) {
    // Output
    zp = C*x+D*u;

    // Propagate covariance
    Rf.compute(R);
    SR = Rf.matrixL();

    Mm << (C*S).transpose(), SR.transpose();

    qr.compute(Mm);
    SZ = qr.matrixQR().block(0, 0, Ny, Ny).template triangularView<Eigen::Upper>().transpose();

    L = C*(S*S.transpose());

    SZ.template triangularView<Eigen::Lower>().solveInPlace(L);
    SZ.template triangularView<Eigen::Lower>().transpose().solveInPlace(L);

    xp = x + L.transpose()*(z - zp);
    LZ = L.transpose()*SZ;

    Sp = S;
    for (int i=0;i<Ny;++i) {
      Eigen::internal::llt_inplace<double, Eigen::Lower>::rankUpdate(Sp, LZ.col(i), -1);
    }
  }

private:

  M<N + Ny, Ny> Mm;
  Eigen::HouseholderQR< M<N + Ny, Ny> > qr;


  M<Ny, Ny> SR;

  Eigen::LLT< M<Ny, Ny> > Rf;

  M<Ny, 1> zp;
  M<Ny, Ny> SZ;
  M<Ny, N> L;
  M<N, Ny> LZ;

  Eigen::LLT< M<N, N> > Sf;
};

template <int N, int Nu>
class KalmanMeasurement {
  public:
  virtual void observe(const M<N, 1>& x, const M<N, N>& S, const M<Nu, 1>& u, M<N, 1>& xp, M<N, N>& Sp) = 0;
};

template <int N, int Nu, int Ny>
class SimpleMeasurement : public KalmanMeasurement<N, Nu> {
public:
  virtual void observe(const M<N, 1>& x, const M<N, N>& S, const M<Nu, 1>& u, M<N, 1>& xp, M<N, N>& Sp) {
    ko.observe(x, S, xp, Sp, C_, D_, R_, z_, u);
  }
  void set(const M<Ny, N>&C, const M<Ny, Nu> &D, const M<Ny, Ny>&R, const M<Ny, 1>&z) {
    C_ = C;
    D_ = D;
    R_ = R;
    z_ = z;
  }
private:
  M<Ny, N> C_;
  M<Ny, Nu> D_;
  M<Ny, Ny> R_;
  M<Ny, 1> z_;

  KalmanObserver<N, Nu, Ny> ko;
};




template <int N, int Nu, class Measurements>
class KalmanEvent {
public:
  // Mutable part
  M<N,  N> S_cache;
  M<N,  1> x_cache;
  M<Nu, 1> u_cache;

  // Immutable
  Measurements m;

  KalmanMeasurement<N, Nu> * active_measurement;

};

template <int N, int Nu, class Measurements>
class EventBuffer {
public:
  typedef std::multimap< double, KalmanEvent<N, Nu, Measurements>* > EventMap;
  typedef typename EventMap::iterator EventMapIt;
  EventBuffer(int max_size) : event_pool_(max_size), max_size_(max_size) {}

  // Get available event
  KalmanEvent<N, Nu, Measurements>* pop_event() {
    if (pool_count < event_pool_.size()) {
      // If pool not exhausted; return an Event from the pool
      return &event_pool_[pool_count++];
    } else {
      // If pool exhausted, start recycling events from the buffer
      EventMapIt lastit = buffer_.begin();
      KalmanEvent<N, Nu, Measurements>* ret = lastit->second;
      buffer_.erase(lastit);
      return ret;
    }
  }

  EventMapIt add_event(double t, KalmanEvent<N, Nu, Measurements>* e) {
    return buffer_.insert(std::pair<double, KalmanEvent<N, Nu, Measurements>* >(t, e));
  }

  EventMapIt end() {
    return buffer_.end();
  }

  KalmanEvent<N, Nu, Measurements>* get_event(double t, double& te) {
    auto it = buffer_.upper_bound(t);
    if (it==buffer_.begin()) return 0;
    --it;
    te = it->first;
    return it->second;
  }

private:
  int max_size_;
  std::vector< KalmanEvent<N, Nu, Measurements> > event_pool_;
  EventMap buffer_;
  KalmanEvent<N, Nu, Measurements> dummy;

  int pool_count = 0;

  /**
    ordered by key
    key not unique?
    fast to discard 'old' items (bottom of the queue)
    fast to look up first element with key> some value // std::lower_bound on vector/deque
 **/


};

template <int N, int Nu, class Measurements>
class KalmanFilter {
public:
  KalmanFilter(const M<N, N>&A, const M<N, Nu>&B, const M<N, N>&Q) : buffer_(100), A_(A), B_(B), Q_(Q) {}

  KalmanEvent<N, Nu, Measurements>* pop_event() {
    return buffer_.pop_event();
  }
  void add_event(double t, KalmanEvent<N, Nu, Measurements>* e) {
    // Add event to the buffer
    auto it_insert = buffer_.add_event(t, e);

    // Update Kalman cache
    typename std::multimap< double, KalmanEvent<N, Nu, Measurements>* >::iterator it_ref = it_insert;

    if (it_insert->second->active_measurement) --it_ref;

    // Obtain previous starting value
    const M<N, 1>* x_ref = &it_ref->second->x_cache;
    const M<N, N>* S_ref = &it_ref->second->S_cache;
    M<Nu, 1> u_ref;
    double t_ref =  it_ref->first;

    for (auto it=it_insert;it!=buffer_.end();++it) {
      if (it->second->active_measurement) {
        // Propagate from starting value to current
        kp.propagate(*x_ref, *S_ref, it->first-t_ref, it->second->x_cache, it->second->S_cache, A_, B_, Q_, u_ref);
        // Measurement update
        it->second->active_measurement->observe(it->second->x_cache, it->second->S_cache, u_ref, it->second->x_cache, it->second->S_cache);
      }
      // Current value becomes new starting value
      M<N, 1>* x_ref = &it->second->x_cache;
      M<N, N>* S_ref = &it->second->S_cache;
      t_ref =  it->first;
    }

  }
  void reset(double t, const M<N, 1>&x, const M<N, N>&P) {
    SP.compute(P);
    auto e = pop_event();
    e->active_measurement = 0;
    e->x_cache = x;
    e->S_cache = SP.matrixL();
    add_event(t, e);
  }
  void input(double t, const M<Nu, 1>&u) {

  }

  void predict(double t, M<N, 1>& x, M<N, N>& P) {
    double te;
    auto ref = buffer_.get_event(t, te);
    if (ref) {
      kp.propagate(ref->x_cache, ref->S_cache, t-te, x, P, A_, B_, Q_, ref->u_cache);
      P = P*P.transpose();
    } else {
      x.setConstant(NAN);
      P.setConstant(NAN);
    }
  }

private:
  EventBuffer<N, Nu, Measurements> buffer_;
  Eigen::LLT< M<N, N> > SP;
  KalmanPropagator<N, Nu> kp;


  M<N, N> A_;
  M<N, Nu> B_;
  M<N, N> Q_;
};

template <int order, class Measurements>
class KinematicKalmanFilter : public KalmanFilter<order, 0, Measurements> {
public:
  static M<order, order> A() {
    M<order, order> ret;
    ret << M<order, 1>(), M<order, order-1>::Identity(order, order-1);
    return ret;
  }

  static M<order, 0> B() {
    return M<order, 0>();
  }

  static M<order, order> Q(double psd) {
    M<order, order> ret;
    ret(order-1, order-1) = psd;
    return ret;
  }

  KinematicKalmanFilter(double psd) : KalmanFilter<order, 0, Measurements>(A(), B(), Q(psd) ) {

  }


};
