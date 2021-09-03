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
#ifndef __TCHEM_IMPL_ENTROPY0SPECML_HPP__
#define __TCHEM_IMPL_ENTROPY0SPECML_HPP__

#include "TChem_Impl_CpSpecMl.hpp"
#include "TChem_Impl_CpSpecMs.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {
template<typename ValueType, typename DeviceType>
struct Entropy0SpecMlFcn
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
    const value_type_1d_view_type& s0i,
    /// workspace
    const value_type_1d_view_type& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    using CpSpecMl = CpSpecMlFcn<value_type,device_type>;

    const scalar_type one[3] = { 0.5, (1.0 / 3.0), 0.25 };
    const value_type tLoc = getValueInRangev2(kmcd.TthrmMin, kmcd.TthrmMax, t);
    const value_type delT = t - tLoc;
    const value_type tln = ats<value_type>::log(tLoc);
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec), [&](const ordinal_type& i) {
        const ordinal_type ipol = tLoc > kmcd.Tmi(i);
        // this assumes nNASAinter_ = 2, nCpCoef_ = 5 (confirm this)
        // icpst = i*7*2+ipol*7 ;
        s0i(i) =
          kmcd.cppol(i, ipol, 0) * tln +
          tLoc * (kmcd.cppol(i, ipol, 1) +
                  tLoc * (kmcd.cppol(i, ipol, 2) * one[0] +
                          tLoc * (kmcd.cppol(i, ipol, 3) * one[1] +
                                  tLoc * kmcd.cppol(i, ipol, 4) * one[2]))) +
          kmcd.cppol(i, ipol, 6);
        s0i(i) *= kmcd.Runiv;
      });

    /* Check if temperature outside bounds */
    if (ats<value_type>::abs(delT) > REACBALANCE()) {
      CpSpecMl::team_invoke(member, tLoc, cpks, kmcd);
      const value_type t_tLoc = t / tLoc;
      Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) {
                             s0i(i) += cpks(i) * ats<value_type>::log(t_tLoc);
                           });
    }
  }
};


struct Entropy0SpecMlFcnDerivative
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    /// output (nspec)
    const RealType1DViewType& ei,
    /// workspace (cpks)
    const RealType1DViewType& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    using kmcd_type = KineticModelConstDataType;
    using device_type = typename kmcd_type::exec_space_type;
    using CpSpecMs = CpSpecMsFcn<real_type, device_type>;

    const real_type tLoc = getValueInRange(kmcd.TthrmMin, kmcd.TthrmMax, t);
    const real_type delT = t - tLoc;
    const real_type tln = ats<real_type>::log(tLoc);

    if (ats<real_type>::abs(delT) > REACBALANCE()) {
      CpSpecMs::team_invoke(member, t, cpks, kmcd);
      Kokkos::parallel_for(
        Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i) { ei(i) = cpks(i) / tLoc; });
    } else {
      Kokkos::parallel_for(
        Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i) {
          const ordinal_type ipol = tLoc > kmcd.Tmi(i);
          // this assumes nNASAinter_ = 2, nCpCoef_ = 5 (confirm this)
          // icpst = i*7*2+ipol*7 ;
          ei(i) = (kmcd.cppol(i, ipol, 0) / tLoc + kmcd.cppol(i, ipol, 1) +
                   tLoc * (kmcd.cppol(i, ipol, 2) +
                           tLoc * (kmcd.cppol(i, ipol, 3) +
                                   tLoc * kmcd.cppol(i, ipol, 4))));
          ei(i) *= kmcd.Runiv;
        });
    }
  }
};
using Entropy0SpecMlDerivative = Entropy0SpecMlFcnDerivative;

} // namespace Impl
} // namespace TChem

#endif
