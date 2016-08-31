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

  void observe(const M<N, 1>& x, const M<N, N>& S, M<N, 1>& xp, M<N, N>& Sp, const M<Ny, N>& C, const M<Ny, Ny>& R, const M<Ny, 1>& z) {
    // Output
    zp = C*x;

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

  KalmanFilter() : buffer_(100) {}
  void set_dynamics(const M<N, N>&A, const M<N, Nu>&B, const M<N, N>&Q) { A_ = A; B_ = B; Q_ = Q; }

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

template<typename T>
constexpr T pack_add(T v) {
  return v;
}

template<typename T, typename... Args>
constexpr T pack_add(T first, Args... args) {
  return first + pack_add(args...);
}

template <class Measurements, int... order>
class KinematicKalmanFilter:  public KalmanFilter< pack_add(order...), 0, Measurements> {
public:
  KinematicKalmanFilter(const std::vector<double>& psd) : KalmanFilter<pack_add(order...), 0, Measurements>() {
    constexpr int N = pack_add(order...);
    std::vector<int> orders = {order...};

    // Contruct A
    M<N, N> A;
    A.setConstant(0);
    int offset_x = 0;
    offset_.push_back(offset_x);
    for (int i=0;i<orders.size();++i) {
      int n = orders[i];
      A.block(offset_x, offset_x, n, n) << Md::Zero(n, 1), Md::Identity(n, n-1);
      offset_x+= orders[i];
      offset_.push_back(offset_x);
    }

    // Contruct Q
    M<N, N> Q;
    Q.setConstant(0);
    offset_x = 0;
    for (int i=0;i<orders.size();++i) {
      int n = orders[i];
      Q(offset_x+n-1, offset_x+n-1) = psd[i];
      offset_x+= orders[i];
    }

    // Kinematic model has no input
    M<N, 0> B;

    this->set_dynamics(A, B, Q);
  }

  std::vector<int> offset_;

};

class OdometryGenericMeasurement  {
public:
  enum {OFF_X = 0, OFF_Y = 2, OFF_THETA = 4};
  void transform_R_dR(double theta, M<2,2>& A, M<2,2>& B) {
    double st = sin(theta);
    double ct = cos(theta);
    A << ct, -st, st, ct;
    B << -st, -ct, ct, -st;
  }
};


class OdometryMeasurement : public KalmanMeasurement<6, 0>, OdometryGenericMeasurement {
public:
  virtual void observe(const M<6, 1>& x, const M<6, 6>& S, const M<0, 1>& u, M<6, 1>& xp, M<6, 6>& Sp) {
    M<2, 2> R, dR;

    transform_R_dR(x(OFF_THETA), R, dR);

    M<2, 1> v;
    v << x(OFF_X+1), x(OFF_Y+1);

    M<3, 6> H;
    H.setConstant(0);
    H(2, OFF_THETA+1) = 1;
    H.block(0, OFF_X+1, 2, 1) = R.transpose().block(0, 0, 2, 1);
    H.block(0, OFF_Y+1, 2, 1) = R.transpose().block(0, 1, 2, 1);
    H.block(0, OFF_THETA, 2, 1) = dR*v;

    M<2, 1> h = R.transpose()*v;
    M<3, 1> r;
    r << h-H.block(0, 0, 2, 6)*x, 0;

    ko.observe(x, S, xp, Sp, H, Sigma, V-r);
  }
  void set(double V_X, double V_Y, double omega, double sigma_X, double sigma_Y, double sigma_omega) {
    V << V_X, V_Y, omega;
    Sigma << sigma_X, 0, 0,   0, sigma_Y, 0,  0, 0, sigma_omega;
  }


private:
  M<3, 1> V;
  M<3, 3> Sigma;

  KalmanObserver<6, 0, 3> ko;
};

template<int Nm>
class markerMeasurement : public KalmanMeasurement<6, 0>, OdometryGenericMeasurement {
public:
  virtual void observe(const M<6, 1>& x, const M<6, 6>& S, const M<0, 1>& u, M<6, 1>& xp, M<6, 6>& Sp) {
    M<2, 2> R, dR;

    transform_R_dR(x(OFF_THETA), R, dR);
    M<2, 1> p;
    p << x(OFF_X), x(OFF_Y);
    M<2, Nm> pr = p*M<1, Nm>::Ones(1, Nm);

    M<2, Nm> m = pattern_meas_.transpose()  - (R*pattern_ref_.transpose() + pr);

    M<2*Nm, 6> H;
    H.setConstant(0);

    for (int i=0;i<Nm;++i) {
      H.block(2*i, OFF_THETA, 2, 1) = dR*pattern_ref_.block(i, 0, 1, 2).transpose();
      H(2*i,   OFF_X) = 1;
      H(2*i+1, OFF_Y) = 1;
    }

    // Eigen is Column major
    M <2*Nm, 1> z = Eigen::Map< M<2*Nm, 1> > (m.data(), 2*Nm) + H*x;
    M <2*Nm, 2*Nm> Rm = M<2*Nm, 2*Nm>::Identity(2*Nm, 2*Nm)*sigma_;

    ko.observe(x, S, xp, Sp, H, Rm, z);
  }
  void set(const M<Nm, 2>& pattern_meas, const M<Nm, 2>& pattern_ref, double sigma) {
    pattern_meas_ = pattern_meas;
    pattern_ref_ = pattern_ref;
    sigma_ = sigma;
  }

private:
  M<Nm,2> pattern_meas_;
  M<Nm,2> pattern_ref_;
  double sigma_;

  KalmanObserver<6, 0, 2*Nm> ko;
};

template<int N>
class OdometryMeasurements {
  public:
  OdometryMeasurement odo;
  markerMeasurement<N> markers;
};

template<int Nm>
class OdometryFilter : public KinematicKalmanFilter<OdometryMeasurements<Nm>, 2, 2, 2> {
  public:
  OdometryFilter(double psd_x, double psd_y, double psd_theta) : KinematicKalmanFilter<OdometryMeasurements<Nm>, 2, 2, 2>({psd_x, psd_y, psd_theta}) {

  }

  void unknown(double t, double sigma=1e6) {
    this->reset(t, M<6,1>::Zero(6, 1), sigma*M<6,6>::Identity(6, 6));
  }

  void observe_odo(double t, double V_X, double V_Y, double omega, double sigma_X, double sigma_Y, double sigma_omega) {
    auto e = this->pop_event();
    e->active_measurement = &e->m.odo;
    e->m.odo.set(V_X, V_Y, omega, sigma_X, sigma_Y, sigma_omega);
    this->add_event(t, e);
  }

  void observe_markers(double t, const M<Nm,2>& pattern_meas, const M<Nm,2>& pattern_ref, double sigma) {
    auto e = this->pop_event();
    e->active_measurement = &e->m.markers;
    e->m.markers.set(pattern_meas, pattern_ref, sigma);
    this->add_event(t, e);
  }
};
