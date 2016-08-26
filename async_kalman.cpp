#include "async_kalman.hpp"
#include <exception>
#include <iostream>




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
   f_int info_ = 0;
   f_int ld_work_ = ld_work;

   mb05nd_(&n_, &delta, A, &lda_, ex, &ldex_, exint, &ldexin_, &tol, iwork, dwork, &ld_work_, &info_);

   info = info_;

   if (info_!=0) {
     std::cerr << "mb05nd error:" << info_ << std::endl;
   }

}

void expm(const Md& A, double t, Md& Ae, Md& Aei, std::vector<f_int>& iwork, std::vector<double>& dwork) {
  int n = A.rows();
  int info;
  slicot_mb05nd(n, t, A.data(), n, Ae.data(), n, Aei.data(), n, 1e-7, &iwork[0], &dwork[0], 2*n*n, info);
}
