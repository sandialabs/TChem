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
#ifndef __TCHEM_IMPL_DENSE_UTV_HPP__
#define __TCHEM_IMPL_DENSE_UTV_HPP__

#include "TChem_Impl_DenseQR.hpp"
#include "TChem_Util.hpp"

#include "KokkosBatched_ApplyPivot_Decl.hpp"
#include "KokkosBatched_SolveUTV_Decl.hpp"
#include "KokkosBatched_UTV_Decl.hpp"

namespace TChem {
namespace Impl {

///
/// UTV factorization and solve for under determined systems
/// A P^T = U T V
///
/// Input:
///  A[m,n]: input matrix having a numeric rank r
/// Output:
///  jpiv[n]: column pivot index (base index is 0)
///  U[m,r]: left orthogoanl matrix
///  T[r,r]: lower triangular matrix; overwritten on A
///  V[r,n]: right orthogonal matrix
///  matrix_rank: numeric rank of matrix A
/// Workspace
///  tau[min(m,n)]: householder coefficients
///

struct DenseUTV
{
  template<typename OrdinalType1DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  inline static void host_factorize( /// input
    const RealType2DViewType& A,
    /// outut
    const OrdinalType1DViewType& jpiv,
    const RealType1DViewType& tau,
    const RealType2DViewType& U,
    const RealType2DViewType& V,
    ordinal_type& matrix_rank)
  {
#if defined(TCHEM_ENABLE_TPL_OPENBLAS) || defined(TCHEM_ENABLE_TPL_MKL)
    assert(tau.stride(0) == 1 && "tau must be contiguous");
    assert(jpiv.stride(0) == 1 && "jpiv must be contiguous");
    if (A.stride(0) == 1) {
      assert(U.stride(0) == 1 && "A and U are not consistent");
      assert(V.stride(0) == 1 && "A and V are not consistent");
    } else if (A.stride(1) == 1) {
      assert(U.stride(1) == 1 && "A and U are not consistent");
      assert(V.stride(1) == 1 && "A and V are not consistent");
    } else {
      assert(false && "A is not colum major nor row major");
    }

    const int lapack_layouts[2] = { LAPACK_ROW_MAJOR, LAPACK_COL_MAJOR };
    const auto layout = lapack_layouts[A.stride(0) == 1];
    const auto layout_transpose = lapack_layouts[layout == LAPACK_ROW_MAJOR];

    const ordinal_type m = A.extent(0), n = A.extent(1), min_mn = m > n ? n : m;

    const ordinal_type lda = A.stride(A.stride(0) == 1),
                       ldu = U.stride(U.stride(0) == 1),
                       ldv = V.stride(V.stride(0) == 1);

    const real_type zero(0);

    if (std::is_same<real_type, float>::value) {
    } else if (std::is_same<real_type, double>::value) {
      /// factorize qr with column pivoting
      LAPACKE_dgeqp3(layout,
                     m,
                     n,
                     (double*)A.data(),
                     lda,
                     (int*)jpiv.data(),
                     (double*)tau.data());

      /// find the matrix matrix_rank
      {
        real_type max_diagonal_A(std::abs(A(0, 0)));
        for (ordinal_type i = 1; i < min_mn; ++i)
          max_diagonal_A = std::max(max_diagonal_A, std::abs(A(i, i)));

        const real_type eps = Kokkos::ArithTraits<real_type>::epsilon();
        const real_type threshold(max_diagonal_A * eps);
        matrix_rank = min_mn;
        for (ordinal_type i = 0; i < min_mn; ++i) {
          if (std::abs(A(i, i)) < threshold) {
            matrix_rank = i;
            break;
          }
        }
      }

      /// modify jpiv from 1 index to 0 index
      {
        int* jptr = (int*)jpiv.data();
        for (ordinal_type i = 0; i < n; ++i)
          --jptr[i];
      }

      if (matrix_rank < min_mn) {
        /// copy householder vectors to U
        {
          for (int i = 0; i < m; ++i)
            for (int j = 0; j < matrix_rank; ++j)
              U(i, j) = A(i, j);
        }
        LAPACKE_dorgqr(layout,
                       m,
                       matrix_rank,
                       matrix_rank,
                       (double*)U.data(),
                       ldu,
                       (double*)tau.data());

        /// clean zeros to make strict upper triangular
        for (int i = 0; i < matrix_rank; ++i)
          for (int j = 0; j < i; ++j)
            A(i, j) = zero;

        /// qr on the transpose R
        LAPACKE_dgeqrf(layout_transpose,
                       n,
                       matrix_rank,
                       (double*)A.data(),
                       lda,
                       (double*)tau.data());

        /// copy householder vectors to V
        for (int i = 0; i < matrix_rank; ++i)
          for (int j = 0; j < n; ++j)
            V(i, j) = A(i, j);

        /// form V with transposed layout
        LAPACKE_dorgqr(layout_transpose,
                       n,
                       matrix_rank,
                       matrix_rank,
                       (double*)V.data(),
                       ldv,
                       (double*)tau.data());
      }
    } else {
      // error
      printf("Error: DenseUTV only support real_type i.e., float and double\n");
    }
#else
    printf("Error: LAPACKE is not enabled; use MKL or OpenBLAS\n");
#endif
  }

  template<typename OrdinalType1DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  inline static void host_1d_solve(
    const ordinal_type& matrix_rank,
    const RealType2DViewType& U,
    const RealType2DViewType& T, // lower triangular
    const RealType2DViewType& V,
    const OrdinalType1DViewType& jpiv,
    const RealType1DViewType& tau,
    const RealType1DViewType& x,
    const RealType1DViewType& b,
    const RealType1DViewType& w)
  {
#if defined(TCHEM_ENABLE_TPL_OPENBLAS) || defined(TCHEM_ENABLE_TPL_MKL)
    assert(w.stride(0) == 1 && "work must be contiguous");
    assert(jpiv.stride(0) == 1 && "jpiv must be contiguous");
    if (T.stride(0) == 1) {
      assert(U.stride(0) == 1 && "T and U are not consistent");
      assert(V.stride(0) == 1 && "T and V are not consistent");
    } else if (T.stride(1) == 1) {
      assert(U.stride(1) == 1 && "T and U are not consistent");
      assert(V.stride(1) == 1 && "T and V are not consistent");
    } else {
      assert(false && "A is not colum major nor row major");
    }

    const ordinal_type m = U.extent(0), n = V.extent(1), min_mn = m > n ? n : m;

    if (matrix_rank < min_mn) {
      /// under-determined problems
      const auto layout = T.stride(1) == 1 ? CblasRowMajor : CblasColMajor;

      const ordinal_type ldu = U.stride(U.stride(0) == 1),
                         ldt = T.stride(T.stride(0) == 1),
                         ldv = V.stride(V.stride(0) == 1);

      const int x_stride = x.stride(0), w_stride = w.stride(0);

      const real_type one(1), zero(0);

      /// don't touch b
      for (ordinal_type i = 0; i < n; ++i)
        w(i) = b(i);

      if (std::is_same<real_type, float>::value) {

      } else if (std::is_same<real_type, double>::value) {
        cblas_dgemv(layout,
                    CblasTrans,
                    m,
                    matrix_rank,
                    one,
                    (const double*)U.data(),
                    ldu,
                    (const double*)w.data(),
                    w_stride,
                    zero,
                    (double*)x.data(),
                    x_stride);
        cblas_dtrsv(layout,
                    CblasLower,
                    CblasNoTrans,
                    CblasNonUnit,
                    matrix_rank,
                    (const double*)T.data(),
                    ldt,
                    (double*)x.data(),
                    x_stride);
        cblas_dgemv(layout,
                    CblasTrans,
                    matrix_rank,
                    n,
                    one,
                    (const double*)V.data(),
                    ldv,
                    (const double*)x.data(),
                    x_stride,
                    zero,
                    (double*)w.data(),
                    w_stride);
      } else {
        // error
        printf(
          "Error: DenseUTV only support real_type i.e., float and double\n");
      }
    } else {
      /// full rank
      DenseQR::host_1d_solve(T, tau, w, b);
    }

    /// apply jpiv to solution
    {
      const int* jptr = (const int*)jpiv.data();
      for (ordinal_type i = 0; i < n; ++i)
        x(jptr[i]) = w(i);
    }
#else
    printf("Error: LAPACKE or CBLAS are not enabled; use MKL or OpenBLAS\n");
#endif
  }

  template<typename OrdinalType1DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  inline static void host_2d_solve(const ordinal_type& matrix_rank,
                                   const RealType2DViewType& U,
                                   const RealType2DViewType& T,
                                   const RealType2DViewType& V,
                                   const OrdinalType1DViewType& jpiv,
                                   const RealType1DViewType& tau,
                                   const RealType2DViewType& x,
                                   const RealType2DViewType& b,
                                   const RealType2DViewType& w)
  {
#if defined(TCHEM_ENABLE_TPL_OPENBLAS) || defined(TCHEM_ENABLE_TPL_MKL)
    // assert(w.stride(0) == 1 && "work must be contiguous");
    assert(jpiv.stride(0) == 1 && "jpiv must be contiguous");
    if (T.stride(0) == 1) {
      assert(U.stride(0) == 1 && "T and U are not consistent");
      assert(V.stride(0) == 1 && "T and V are not consistent");
    } else if (T.stride(1) == 1) {
      assert(U.stride(1) == 1 && "T and U are not consistent");
      assert(V.stride(1) == 1 && "T and V are not consistent");
    } else {
      assert(false && "A is not colum major nor row major");
    }
    assert(x.extent(1) == b.extent(1) && "nrhs are different between x and b");
    assert(x.extent(1) == w.extent(1) && "nrhs are different between x and w");

    const ordinal_type m = U.extent(0), n = V.extent(1), min_mn = m > n ? n : m,
                       nrhs = x.extent(1);

    if (matrix_rank < min_mn) {
      /// under-determined system
      const auto layout = T.stride(1) == 1 ? CblasRowMajor : CblasColMajor;

      const ordinal_type ldu = U.stride(layout == CblasColMajor),
                         ldt = T.stride(layout == CblasColMajor),
                         ldv = V.stride(layout == CblasColMajor),
                         ldx = x.stride(layout == CblasColMajor),
                         ldw = w.stride(layout == CblasColMajor);

      const real_type one(1), zero(0);

      /// don't touch b
      for (ordinal_type i = 0; i < m; ++i)
        for (ordinal_type j = 0; j < nrhs; ++j)
          w(i, j) = b(i, j);

      if (std::is_same<real_type, float>::value) {
      } else if (std::is_same<real_type, double>::value) {
        cblas_dgemm(layout,
                    CblasTrans,
                    CblasNoTrans,
                    matrix_rank,
                    nrhs,
                    m,
                    one,
                    (const double*)U.data(),
                    ldu,
                    (const double*)w.data(),
                    ldw,
                    zero,
                    (double*)x.data(),
                    ldx);

        cblas_dtrsm(layout,
                    CblasLeft,
                    CblasLower,
                    CblasNoTrans,
                    CblasNonUnit,
                    matrix_rank,
                    nrhs,
                    one,
                    (const double*)T.data(),
                    ldt,
                    (double*)x.data(),
                    ldx);

        cblas_dgemm(layout,
                    CblasTrans,
                    CblasNoTrans,
                    n,
                    nrhs,
                    matrix_rank,
                    one,
                    (const double*)V.data(),
                    ldv,
                    (const double*)x.data(),
                    ldx,
                    zero,
                    (double*)w.data(),
                    ldw);
      } else {
        // error
        printf(
          "Error: DenseUTV only support real_type i.e., float and double\n");
      }
    } else {
      DenseQR::host_2d_solve(T, tau, w, b);
    }

    /// apply jpiv to solution
    {
      const int* jptr = (const int*)jpiv.data();
      for (ordinal_type i = 0; i < n; ++i) {
        const ordinal_type id = jptr[i];
        for (ordinal_type j = 0; j < nrhs; ++j)
          x(id, j) = w(i, j);
      }
    }

#else
    printf("Error: LAPACKE or CBLAS are not enabled; use MKL or OpenBLAS\n");
#endif
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename RealTypeXDViewType>
  KOKKOS_INLINE_FUNCTION static void team_factorize_and_solve(
    const MemberType& member,
    const RealType2DViewType& A,
    const RealTypeXDViewType& x,
    const RealTypeXDViewType& b,
    const RealType1DViewType& w,
    ordinal_type& matrix_rank)
  {
    constexpr int rank = RealTypeXDViewType::rank;
    const int m = A.extent(0), n = A.extent(1), min_mn = m > n ? n : m;
    real_type* wptr = w.data();
    RealType2DViewType U(wptr, m, n);
    wptr += U.span();
    RealType2DViewType V(wptr, n, n);
    wptr += V.span();

    using pivot_view_type =
      Kokkos::View<ordinal_type*, Kokkos::Impl::ActiveExecutionMemorySpace>;
    pivot_view_type jpiv((ordinal_type*)wptr, min_mn);
    wptr += jpiv.span();

    if (std::is_same<Kokkos::Impl::ActiveExecutionMemorySpace,
                     Kokkos::HostSpace>::value) {
#if defined(TCHEM_ENABLE_TPL_OPENBLAS) || defined(TCHEM_ENABLE_TPL_MKL)
      RealType1DViewType tau(wptr, min_mn);
      wptr += tau.span();

      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        host_factorize(A, jpiv, tau, U, V, matrix_rank);
        if (rank == 1) {
          RealType1DViewType xx(x.data(), n);
          RealType1DViewType bb(b.data(), n);
          RealType1DViewType tt(wptr, n);
          wptr += tt.span();
          assert(int(wptr - w.data()) <= int(w.extent(0)) &&
                 "workspace is used more than allocated");
          host_1d_solve(matrix_rank, U, A, V, jpiv, tau, xx, bb, tt);
        } else {
          const ordinal_type nrhs(b.extent(1));
          RealType2DViewType xx(x.data(), n, nrhs);
          RealType2DViewType bb(b.data(), n, nrhs);
          RealType2DViewType tt(wptr, n, nrhs);
          wptr += tt.span();
          assert(int(wptr - w.data()) <= int(w.extent(0)) &&
                 "workspace is used more than allocated");
          host_2d_solve(matrix_rank, U, A, V, jpiv, tau, xx, bb, tt);
        }
      });
#else
      const ordinal_type nrhs(b.extent(1));
      RealType1DViewType work(wptr, 3 * m + nrhs * n);
      assert(int(wptr - w.data()) <= int(w.extent(0)) &&
             "workspace is used more than allocated");
      KokkosBatched::
        TeamVectorUTV<MemberType, KokkosBatched::Algo::UTV::Unblocked>::invoke(
          member, A, jpiv, U, V, work, matrix_rank);
      member.team_barrier();
      if (rank == 1) {
        RealType1DViewType xx(x.data(), n);
        RealType1DViewType bb(b.data(), n);
        KokkosBatched::TeamVectorSolveUTV<MemberType,
                                          KokkosBatched::Algo::UTV::Unblocked>::
          invoke(member, matrix_rank, U, A, V, jpiv, xx, bb, work);
      } else {
        RealType2DViewType xx(x.data(), n, nrhs);
        RealType2DViewType bb(b.data(), n, nrhs);
        KokkosBatched::TeamVectorSolveUTV<MemberType,
                                          KokkosBatched::Algo::UTV::Unblocked>::
          invoke(member, matrix_rank, U, A, V, jpiv, xx, bb, work);
      }
      member.team_barrier();
#endif
    } else {
      const ordinal_type nrhs(b.extent(1));
      RealType1DViewType work(wptr, 3 * m + nrhs * n);
      assert(int(wptr - w.data()) <= int(w.extent(0)) &&
             "workspace is used more than allocated");
      KokkosBatched::
        TeamVectorUTV<MemberType, KokkosBatched::Algo::UTV::Unblocked>::invoke(
          member, A, jpiv, U, V, work, matrix_rank);
      member.team_barrier();
      if (rank == 1) {
        RealType1DViewType xx(x.data(), n);
        RealType1DViewType bb(b.data(), n);
        KokkosBatched::TeamVectorSolveUTV<MemberType,
                                          KokkosBatched::Algo::UTV::Unblocked>::
          invoke(member, matrix_rank, U, A, V, jpiv, xx, bb, work);
      } else {
        RealType2DViewType xx(x.data(), n, nrhs);
        RealType2DViewType bb(b.data(), n, nrhs);
        KokkosBatched::TeamVectorSolveUTV<MemberType,
                                          KokkosBatched::Algo::UTV::Unblocked>::
          invoke(member, matrix_rank, U, A, V, jpiv, xx, bb, work);
      }
      member.team_barrier();
    }
  }
};

} // namespace Impl
} // namespace TChem

#endif
