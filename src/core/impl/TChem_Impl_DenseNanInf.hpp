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
#ifndef __TCHEM_IMPL_DENSE_NANINF_HPP__
#define __TCHEM_IMPL_DENSE_NANINF_HPP__

#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

struct DenseNanInf
{
  template<typename MemberType, typename RealTypeXDViewType>
  KOKKOS_INLINE_FUNCTION static void team_check_sanity(
    const MemberType& member,
    const RealTypeXDViewType& A,
    /* */ bool& is_valid)
  {
    constexpr int rank = RealTypeXDViewType::rank;
    ordinal_type num_nan_inf(0), m(0), n(0), as0(0), as1(0);
    if (rank == 1) {
      m = A.extent(0);
      n = 1;
      as0 = A.stride(0);
      as1 = 1;
    } else {
      m = A.extent(0);
      n = A.extent(1);
      as0 = A.stride(0);
      as1 = A.stride(1);
    }

    using ats = Kokkos::ArithTraits<real_type>;
    const auto aptr = A.data();
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, m * n),
      [&](const ordinal_type& ij, ordinal_type& update) {
        const ordinal_type i = ij / n, j = ij % n;
        const real_type val = aptr[i * as0 + j * as1];
        update += (ats::isNan(val) || ats::isInf(val));
      },
      num_nan_inf);
    member.team_barrier();
    is_valid = (num_nan_inf == 0);
  }
};

} // namespace Impl
} // namespace TChem

#endif
