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
#ifndef __TCHEM_IMPL_ENTHALPYSPECML_HPP__
#define __TCHEM_IMPL_ENTHALPYSPECML_HPP__

#include "TChem_Impl_CpSpecMl.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

//
template<typename ValueType, typename DeviceType>
struct EnthalpySpecMlFcn
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;

  template<typename MemberType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const value_type& t,
    /// output (nspec)
    const value_type_1d_view_type& hi,
    /// workspace (nspec)
    const value_type_1d_view_type& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    //constant number
    using CpSpecMl = CpSpecMlFcn<value_type,device_type>;

    const scalar_type one[4] = { 0.5, (1.0 / 3.0), 0.25, 0.2 };
    const auto tLoc = getValueInRangev2(kmcd.TthrmMin, kmcd.TthrmMax, t);

    // const RealType tln = ats<RealType>::log(tLoc);
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec), [&](const ordinal_type& i) {
        const ordinal_type ipol = tLoc > kmcd.Tmi(i);
        // this assumes nNASAinter_ = 2, nCpCoef_ = 5 (confirm this)
        // icpst = i*7*2+ipol*7 ;
        hi(i) =
          tLoc *
            (kmcd.cppol(i, ipol, 0) +
             tLoc *
               (kmcd.cppol(i, ipol, 1) * one[0] +
                tLoc * (kmcd.cppol(i, ipol, 2) * one[1] +
                        tLoc * (kmcd.cppol(i, ipol, 3) * one[2] +
                                tLoc * (kmcd.cppol(i, ipol, 4) * one[3]))))) +
          kmcd.cppol(i, ipol, 5);

        hi(i) *= kmcd.Runiv;
      });

    const auto delT = t - tLoc;
    if (ats<value_type>::abs(delT) > REACBALANCE()) {
      CpSpecMl::team_invoke(member, tLoc, cpks, kmcd);
      Kokkos::parallel_for(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i) { hi(i) += cpks(i) * delT; });
    }


#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("EnthalpySpecMl.team_invoke.test.out", "a+");
      fprintf(fs, ":: EnthalpySpecMl::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs, "     nSpec %3d, t %e\n", kmcd.nSpec, t);
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "     i %3d, hi %e, cpks %e\n", i, hi(i), cpks(i));
    }
#endif
  }
};

} // namespace Impl
} // namespace TChem

#endif
