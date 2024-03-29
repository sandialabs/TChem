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
#ifndef __TCHEM_IMPL_CPMIXMS_HPP__
#define __TCHEM_IMPL_CPMIXMS_HPP__

#include "TChem_Impl_CpSpecMs.hpp"
#include "TChem_Util.hpp"
#include "Tines_Internal.hpp"

namespace TChem {

namespace Impl {

  template<typename ValueType, typename DeviceType>
  struct CpMixMs
  {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    /// sacado is value type
    using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;

    template<typename MemberType,
             typename KineticModelConstDataType>
    KOKKOS_INLINE_FUNCTION static value_type team_invoke(
      const MemberType& member,
      /// input
      const value_type& t, /// temperature
      const value_type_1d_view_type& Ys,
      const value_type_1d_view_type& cpks,
      /// const input from kinetic model
      const KineticModelConstDataType& kmcd)
    {
      CpSpecMsFcn<value_type, device_type>::team_invoke(member, t, cpks, kmcd);

      using reducer_type = Tines::SumReducer<value_type>;
      typename reducer_type::value_type cpmix(0);

      // RealType cpmix(0);
      Kokkos::parallel_reduce(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i, typename reducer_type::value_type& update) {
          update += Ys(i) * cpks(i);
        }, reducer_type(cpmix));

  #if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
      if (member.league_rank() == 0) {
        FILE* fs = fopen("CpMixMs.team_invoke.test.out", "a+");
        fprintf(fs, ":: CpMixMs::team_invoke\n");
        fprintf(fs, ":::: input\n");
        fprintf(fs, "     nSpec %3d, t %e\n", kmcd.nSpec, t);
        for (int i = 0; i < kmcd.nSpec; ++i)
          fprintf(
            fs, "     i %3d, Ys %e, cpks (computed) %e\n", i, Ys(i), cpks(i));
        fprintf(fs, ":::: output\n");
        fprintf(fs, "     cpmix %e\n", cpmix);
      }
  #endif
      return cpmix;
    }
  };

  struct CpMixMsDerivative
  {
    template<typename MemberType,
             typename RealType1DViewType,
             typename KineticModelConstDataType>
    KOKKOS_INLINE_FUNCTION static real_type team_invoke(
      const MemberType& member,
      /// input
      const real_type& t,           /// temperature
      const RealType1DViewType& Ys, /// temperature
      /// workspace
      const RealType1DViewType& cpks, /// work
      /// const input from kinetic model
      const KineticModelConstDataType& kmcd)
    {
      CpSpecMsDerivative::team_invoke(member, t, cpks, kmcd);

      real_type cpmix(0);
      Kokkos::parallel_reduce(
        Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i, real_type& update) {
          update += Ys(i) * cpks(i);
        },
        cpmix);
  #if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
      if (member.league_rank() == 0) {
        FILE* fs = fopen("CpMixMsDerivative.team_invoke.test.out", "a+");
        fprintf(fs, ":: CpMixMsDerivative::team_invoke\n");
        fprintf(fs, ":::: input\n");
        fprintf(fs, "     nSpec %3d, t %e\n", kmcd.nSpec, t);
        for (int i = 0; i < kmcd.nSpec; ++i)
          fprintf(
            fs, "     i %3d, Ys %e, cpks (computed) %e\n", i, Ys(i), cpks(i));
        fprintf(fs, ":::: output\n");
        fprintf(fs, "     cpmix %e\n", cpmix);
      }
  #endif
      return cpmix;
    }
  };

} // namespace Impl
} // namespace TChem

#endif
