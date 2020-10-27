/* =====================================================================================
TChem version 2.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

This file is part of TChem. TChem is open source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the licese is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */
#ifndef __TCHEM_IMPL_DENSE_QR_HPP__
#define __TCHEM_IMPL_DENSE_QR_HPP__

#include "TChem_Util.hpp"

#include "KokkosBatched_ApplyQ_Decl.hpp"
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_QR_Decl.hpp"
#include "KokkosBatched_Trsv_Decl.hpp"

namespace TChem {
namespace Impl {

struct DenseQR
{
  template<typename RealType1DViewType, typename RealType2DViewType>
  inline static void host_factorize(const RealType2DViewType& A,
                                    const RealType1DViewType& tau)
  {
#if defined(TCHEM_ENABLE_TPL_OPENBLAS) || defined(TCHEM_ENABLE_TPL_MKL)
    assert(tau.stride(0) == 1 && "tau must be contiguous");
    const ordinal_type m = A.extent(0), n = A.extent(1);

    const auto layout = A.stride(0) == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    const ordinal_type lda = A.stride(A.stride(0) == 1);

    if (std::is_same<real_type, float>::value) {
      LAPACKE_sgeqrf(layout, m, n, (float*)A.data(), lda, (float*)tau.data());
    } else if (std::is_same<real_type, double>::value) {
      LAPACKE_dgeqrf(layout, m, n, (double*)A.data(), lda, (double*)tau.data());
    } else {
      // error
      printf("Error: DenseQR only support real_type i.e., float and double\n");
    }
#else
    printf("Error: LAPACKE is not enabled; use MKL or OpenBLAS\n");
#endif
  }

  template<typename RealType1DViewType, typename RealType2DViewType>
  inline static void host_1d_solve(const RealType2DViewType& A,
                                   const RealType1DViewType& tau,
                                   const RealType1DViewType& x,
                                   const RealType1DViewType& b)
  {
#if defined(TCHEM_ENABLE_TPL_OPENBLAS) || defined(TCHEM_ENABLE_TPL_MKL)
    assert(tau.stride(0) == 1);
    const ordinal_type m = A.extent(0), n = A.extent(1), min_mn = m > n ? n : m;

    const auto layout_lapacke =
      A.stride(0) == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
    const auto layout_cblas = A.stride(0) == 1 ? CblasColMajor : CblasRowMajor;

    const ordinal_type lda = A.stride(A.stride(0) == 1),
                       ldx = layout_lapacke == LAPACK_COL_MAJOR ? m : 1,
                       x_stride = x.stride(0);

    /// don't touch b
    for (ordinal_type i = 0; i < m; ++i)
      x(i) = b(i);

    if (std::is_same<real_type, float>::value) {
      LAPACKE_sormqr(layout_lapacke,
                     'L',
                     'T',
                     m,
                     1,
                     min_mn,
                     (const float*)A.data(),
                     lda,
                     (const float*)tau.data(),
                     (float*)x.data(),
                     ldx);
      cblas_strsv(layout_cblas,
                  CblasUpper,
                  CblasNoTrans,
                  CblasNonUnit,
                  m,
                  (const float*)A.data(),
                  lda,
                  (float*)x.data(),
                  x_stride);
    } else if (std::is_same<real_type, double>::value) {
      LAPACKE_dormqr(layout_lapacke,
                     'L',
                     'T',
                     m,
                     1,
                     min_mn,
                     (const double*)A.data(),
                     lda,
                     (const double*)tau.data(),
                     (double*)x.data(),
                     ldx);
      cblas_dtrsv(layout_cblas,
                  CblasUpper,
                  CblasNoTrans,
                  CblasNonUnit,
                  m,
                  (const double*)A.data(),
                  lda,
                  (double*)x.data(),
                  x_stride);
    } else {
      // error
      printf("Error: DenseQR only support real_type i.e., float and double\n");
    }
#else
    printf("Error: LAPACKE or CBLAS are not enabled; use MKL or OpenBLAS\n");
#endif
  }

  template<typename RealType1DViewType,
           typename RealType2DViewType>
  inline static void host_2d_solve(
    const RealType2DViewType& A, // dim(A) = m x n
    const RealType1DViewType& tau,
    const RealType2DViewType& x, // dim(x) = m x l
    const RealType2DViewType& b)
  {
#if defined(TCHEM_ENABLE_TPL_OPENBLAS) || defined(TCHEM_ENABLE_TPL_MKL)
    assert(tau.stride(0) == 1);

    const ordinal_type m = A.extent(0), n = A.extent(1), nrhs = b.extent(1),
                       min_mn = m > n ? n : m;

    const auto layout_lapacke =
      A.stride(0) == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
    const auto layout_cblas = A.stride(0) == 1 ? CblasColMajor : CblasRowMajor;

    const ordinal_type lda = A.stride(A.stride(0) == 1),
                       ldx = x.stride(x.stride(0) == 1), x_stride = x.stride(0);

    /// don't touch b
    for (ordinal_type i = 0; i < m; ++i)
      for (ordinal_type j = 0; j < nrhs; ++j)
        x(i, j) = b(i, j);

    if (std::is_same<real_type, float>::value) {
      LAPACKE_sormqr(layout_lapacke,
                     'L',
                     'T',
                     m,
                     nrhs,
                     min_mn,
                     (const float*)A.data(),
                     lda,
                     (const float*)tau.data(),
                     (float*)x.data(),
                     ldx);
      cblas_strsm(layout_cblas,
                  CblasLeft,
                  CblasUpper,
                  CblasNoTrans,
                  CblasNonUnit,
                  m,
                  nrhs,
                  float(1),
                  (const float*)A.data(),
                  lda,
                  (float*)x.data(),
                  ldx);
    } else if (std::is_same<real_type, double>::value) {
      LAPACKE_dormqr(layout_lapacke,
                     'L',
                     'T',
                     m,
                     nrhs,
                     min_mn,
                     (const double*)A.data(),
                     lda,
                     (const double*)tau.data(),
                     (double*)x.data(),
                     ldx);
      cblas_dtrsm(layout_cblas,
                  CblasLeft,
                  CblasUpper,
                  CblasNoTrans,
                  CblasNonUnit,
                  m,
                  nrhs,
                  double(1),
                  (const double*)A.data(),
                  lda,
                  (double*)x.data(),
                  ldx);
    } else {
      // error
      printf("Error: DenseQR only support real_type i.e., float and double\n");
    }
#else
    printf("Error: LAPACKE or CBLAS are not enabled; use MKL or OpenBLAS\n");
#endif
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION static void device_factorize(
    const MemberType& member,
    const RealType2DViewType& A,
    const RealType1DViewType& tau,
    const RealType1DViewType& work)
  {
    assert(tau.stride(0) == 1);
    KokkosBatched::
      TeamVectorQR<MemberType, KokkosBatched::Algo::QR::Unblocked>::invoke(
        member, A, tau, work);
    member.team_barrier();
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION static void device_1d_solve(
    const MemberType& member,
    const RealType2DViewType& A,
    const RealType1DViewType& tau,
    const RealType1DViewType& x,
    const RealType1DViewType& b,
    const RealType1DViewType& work)
  {
    const real_type one(1);
    KokkosBatched::
      TeamVectorCopy<MemberType, KokkosBatched::Trans::NoTranspose>::invoke(
        member, b, x);
    member.team_barrier();
    KokkosBatched::TeamVectorApplyQ<
      MemberType,
      KokkosBatched::Side::Left,
      KokkosBatched::Trans::Transpose,
      KokkosBatched::Algo::ApplyQ::Unblocked>::invoke(member, A, tau, x, work);
    member.team_barrier();
    KokkosBatched::TeamVectorTrsv<
      MemberType,
      KokkosBatched::Uplo::Upper,
      KokkosBatched::Trans::NoTranspose,
      KokkosBatched::Diag::NonUnit,
      KokkosBatched::Algo::Trsv::Unblocked>::invoke(member, one, A, x);
    member.team_barrier();
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION static void device_2d_solve(
    const MemberType& member,
    const RealType2DViewType& A,
    const RealType1DViewType& tau,
    const RealType2DViewType& x,
    const RealType2DViewType& b,
    const RealType1DViewType& work)
  {
    const real_type one(1);
    KokkosBatched::
      TeamVectorCopy<MemberType, KokkosBatched::Trans::NoTranspose>::invoke(
        member, b, x);
    member.team_barrier();
    KokkosBatched::TeamVectorApplyQ<
      MemberType,
      KokkosBatched::Side::Left,
      KokkosBatched::Trans::Transpose,
      KokkosBatched::Algo::ApplyQ::Unblocked>::invoke(member, A, tau, x, work);
    member.team_barrier();
    // this is not a good way but I do not have team vector version of trsm for
    // now
    const ordinal_type nrhs = b.extent(1);
    for (ordinal_type j = 0; j < nrhs; ++j) {
      auto xx = Kokkos::subview(x, Kokkos::ALL(), j);
      KokkosBatched::TeamVectorTrsv<
        MemberType,
        KokkosBatched::Uplo::Upper,
        KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Diag::NonUnit,
        KokkosBatched::Algo::Trsv::Unblocked>::invoke(member, one, A, xx);
    }
    member.team_barrier();
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename RealTypeXDViewType>
  KOKKOS_INLINE_FUNCTION static void team_factorize_and_solve(
    const MemberType& member,
    const RealType2DViewType& A,
    const RealType1DViewType& tau,
    const RealTypeXDViewType& x,
    const RealTypeXDViewType& b,
    const RealType1DViewType& w)
  {
    constexpr int rank = RealTypeXDViewType::rank;
    if (std::is_same<Kokkos::Impl::ActiveExecutionMemorySpace,
                     Kokkos::HostSpace>::value) {
#if defined(TCHEM_ENABLE_TPL_OPENBLAS) || defined(TCHEM_ENABLE_TPL_MKL)
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        host_factorize(A, tau);
        if (rank == 1) {
          RealType1DViewType xx(x.data(), x.extent(0));
          RealType1DViewType bb(b.data(), b.extent(0));
          host_1d_solve(A, tau, xx, bb);
        } else {
          RealType2DViewType xx(x.data(), x.extent(0), x.extent(1));
          RealType2DViewType bb(b.data(), b.extent(0), x.extent(1));
          host_2d_solve(A, tau, xx, bb);
        }
      });
#else
      /// execution device is host device and tpl blas or mkl are not available
      device_factorize(member, A, tau, w);
      if (rank == 1) {
        RealType1DViewType xx(x.data(), x.extent(0));
        RealType1DViewType bb(b.data(), b.extent(0));
        device_1d_solve(member, A, tau, xx, bb, w);
      } else {
        RealType2DViewType xx(x.data(), x.extent(0), x.extent(1));
        RealType2DViewType bb(b.data(), b.extent(0), x.extent(1));
        device_2d_solve(member, A, tau, xx, bb, w);
      }
#endif
    } else {
      device_factorize(member, A, tau, w);
      if (rank == 1) {
        RealType1DViewType xx(x.data(), x.extent(0));
        RealType1DViewType bb(b.data(), b.extent(0));
        device_1d_solve(member, A, tau, xx, bb, w);
      } else {
        RealType2DViewType xx(x.data(), x.extent(0), x.extent(1));
        RealType2DViewType bb(b.data(), b.extent(0), x.extent(1));
        device_2d_solve(member, A, tau, xx, bb, w);
      }
    }
  }
};

} // namespace Impl
} // namespace TChem

#endif
