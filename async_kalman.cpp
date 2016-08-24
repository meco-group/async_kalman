#include "async_kalman.hpp"
#include <exception>
#include <iostream>
#include <vector>

// Need an 8-byte integer since libslicot0 is compiled with  fdefault-integer-8
typedef long long int f_int;

extern "C" {
  int mb05nd_(f_int* n, double* delta, const double*A, f_int* lda,
                      double* ex, f_int* ldex, double* exint, f_int* ldexin,
                      double* tol, f_int *iwork, double* dwork, f_int *ld_work, f_int *info);
}

void slicot_mb05nd(int n, double delta, const double* A, int lda,
                   double* ex, int ldex, double* exint, int ldexin,
                   double tol, f_int* iwork, double* dwork, int ld_work, int& info) {
   f_int n_ = n;
   f_int lda_ = lda;
   f_int ldex_ = ldex;
   f_int ldexin_ = ldexin;
   f_int info_ = info;
   f_int ld_work_ = ld_work;

   mb05nd_(&n_, &delta, A, &lda_, ex, &ldex_, exint, &ldexin_, &tol, iwork, dwork, &ld_work_, &info_);

   info = info_;

   if (info_<0) {
     std::cerr << "mb05nd wrond arguments" << std::endl;
   } else if (info_>0) {
     std::cerr << "mb05nd wrond arguments" << std::endl;
   }


}

M c2d(const M& A, const M& B, double t) {
  int n = A.rows();
  M F(n,n);
  M H(n,n);
  std::vector<f_int> iwork(n);
  int ld_work = 2*n*n;
  std::vector<double> dwork(ld_work);
  int info;
  slicot_mb05nd( n, t, A.data(), n, F.data(), n, H.data(), n, 1e-7, &iwork[0], &dwork[0], ld_work, info);
  return F;
}
