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
#ifndef __TCHEM_IMPL_TRANSIENTCONTSTIRREDTANKREACTORRHS_HPP__
#define __TCHEM_IMPL_TRANSIENTCONTSTIRREDTANKREACTORRHS_HPP__

#include "TChem_Util.hpp"
#include "TChem_Impl_ReactionRates.hpp"
#include "TChem_NetProductionRatePerMass.hpp"
#include "TChem_Impl_ReactionRatesSurface.hpp"
#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_EnthalpySpecMs.hpp"


namespace TChem {
namespace Impl {

struct TransientContStirredTankReactorRHS
{
  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    const real_type ReacRateWork = TChem::NetProductionRatePerMass::getWorkSpaceSize(kmcd);
    const real_type ReacRateSurfWork = ReactionRatesSurface
                                              ::getWorkSpaceSize(kmcd,kmcdSurf);

    //
    const ordinal_type workspace_size = ReacRateWork > ReacRateSurfWork
                                          ? ReacRateWork
                                          : ReacRateSurfWork;

    return (workspace_size + 4*kmcd.nSpec + kmcdSurf.nSpec);
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename ContStirredTankReactorConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec) mass fraction
    const RealType1DViewType& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p,   // pressure


    /// output
    const RealType1DViewType& dT,  /// temperature
    const RealType1DViewType& dYs,  /// (kmcd.nSpec) // species
    const RealType1DViewType& dZs,  /// (kmcdSurf.nSpec) // surface species

    /// workspace
    const RealType1DViewType& omega,
    const RealType1DViewType& omegaSurfGas,
    const RealType1DViewType& omegaSurf,
    const RealType1DViewType& hks,
    const RealType1DViewType& cpks,
    // gas species
    const RealType1DViewType& work,

    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    // const input for plug flow reactor
    const ContStirredTankReactorConstDataType& cstr)
  {


    /// 1. compute species enthalies and cp mix

    EnthalpySpecMs ::team_invoke(member, t, hks, cpks, kmcd);

    const real_type cpmix = CpMixMs::team_invoke(member, t, Ys, cpks, kmcd);


    ///  compute molar reaction rates
    ReactionRates ::team_invoke(member,
                                       t,
                                       p,
                                       Ys,
                                       omega,
                                       work,
                                       kmcd);

    // compute mix enthalpy

    real_type enthapyMix(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        update += hks(k) * Ys(k);

    },
    enthapyMix);

    member.team_barrier();
    /// 2. compute catalysis production rates
    ReactionRatesSurface ::team_invoke(member,
                                            t,
                                            p,
                                            Ys,
                                            Zs,
                                            omegaSurfGas,
                                            omegaSurf,
                                            work,
                                            kmcd,
                                            kmcdSurf);


    member.team_barrier();

    /// 3. transform molar reaction rates to mass reaction rates
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& k) {
                           omega(k) *= kmcd.sMass(k);        // kg/m3/2
                           omegaSurfGas(k) *= kmcd.sMass(k); // kg/m2/s
                         });

    member.team_barrier();

    real_type sumSkWk(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        update += omegaSurfGas(k);
      },
      sumSkWk);

    member.team_barrier();

    real_type sum_hkdYkdt(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        dYs(k) = (omega(k)*cstr.Vol + // gas contribution from gas phase
                  omegaSurfGas(k)*cstr.Acat + // gas contribution from surface phase
                   cstr.mdotIn * (cstr.Yi(k) -  Ys(k)) - Ys(k)*sumSkWk*cstr.Acat ) /
                  (density * cstr.Vol);         // species equation
        update += hks(k) * dYs(k);
      }, sum_hkdYkdt);

#if defined(TCHEM_ENABLE_PROBLEM_DAE_CSTR)
    //
    real_type Zsum(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcdSurf.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        dZs(k) = omegaSurf(k);  ///  surface species equation
        update += Zs(k); // Units of omega (kg/m3/s).
      },Zsum);

      member.team_barrier();
      const real_type one(1);
      dZs(kmcdSurf.nSpec - 1) = one - Zsum;

#else
    //
    const real_type ten(10.0);

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcdSurf.nSpec),
      [&](const ordinal_type& k) {
        // units of sitedensity are mol/cm2 (ten)
        dZs(k) = omegaSurf(k)/kmcdSurf.sitedensity/ten;  /// ; // surface species equation
    });

#endif


    member.team_barrier();


    dT(0) = - sum_hkdYkdt / cpmix +  (cstr.mdotIn * (cstr.EnthalpyIn -  enthapyMix ) - enthapyMix*sumSkWk*cstr.Acat ) /
                                     ( cpmix * density * cstr.Vol) ;
    member.team_barrier();



    // energy equation


#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("TransientContStirredTankReactorRHS.team_invoke.test.out", "a+");
      fprintf(fs, ":: TransientContStirredTankReactorRHS::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "Surface::     nSpec %3d, nReac %3d, site density %e\n",
              kmcdSurf.nSpec,
              kmcdSurf.nReac,
              kmcdSurf.sitedensity);
      fprintf(fs, "Gas::     nSpec %3d, nReac %3d\n", kmcd.nSpec, kmcd.nReac);
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "   i %3d,  Ys %e, \n", i, Ys(i));
      for (int i = 0; i < kmcdSurf.nSpec; ++i)
        fprintf(fs, "   i %3d,  Zs %e, \n", i, Zs(i));

      fprintf(fs, ":::: output\n");
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "   i %3d,  omega %e, \n", i, omega(i));
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "   i %3d,  omegaSurfGas %e, \n", i, omegaSurfGas(i));
      for (int i = 0; i < kmcdSurf.nSpec; ++i)
        fprintf(fs, "   i %3d,  omegaSurf %e, \n", i, omegaSurf(i));
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "   i %3d,  dYk %e, \n", i, dYs(i));
      for (int i = 0; i < kmcdSurf.nSpec; ++i)
        fprintf(fs, "   i %3d,  dZs %e, \n", i, dZs(i));
      fprintf(fs, ":::: output\n");
    }
#endif
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename ContStirredTankReactorConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    const RealType1DViewType& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p, // pressure
    /// output
    const RealType1DViewType& rhs, /// (kmcd.nSpec + 1)
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const ContStirredTankReactorConstDataType& cstr)
  {
    // const real_type zero(0);

    const ordinal_type len_rhs_w = getWorkSpaceSize(kmcd, kmcdSurf);
    if (len_rhs_w > ordinal_type(work.extent(0))) {
      Kokkos::abort("Error: workspace used is smaller than it "
                    "required::TChem_Impl_TransientContStirredTankReactorRHS\n");
      printf("work required %d, provided %d \n",
             len_rhs_w,
             ordinal_type(work.extent(0)));
    }

    auto w = (real_type*)work.data();

    auto omega = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto omegaSurfGas = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto omegaSurf = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;

    auto hks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = RealType1DViewType(w, kmcd.nSpec);

    const ordinal_type workspace_extent(work.extent(0));

    auto workRHS = RealType1DViewType(w, workspace_extent);

    auto dT = RealType1DViewType(rhs.data(), 1);
    auto dYs = RealType1DViewType(rhs.data() + 1, kmcd.nSpec);
    auto dZs = RealType1DViewType(rhs.data() + 1 + kmcd.nSpec, kmcdSurf.nSpec);

    team_invoke_detail(member,
                       t,
                       Ys,
                       Zs,
                       density,
                       p,
                       dT,
                       dYs,
                       dZs,
                       /// workspace
                       omega,
                       omegaSurfGas,
                       omegaSurf,
                       hks,
                       cpks,
                       workRHS,
                 // data from surface and gas phases
                       kmcd,
                       kmcdSurf,
                       cstr);
  }
};

} // namespace Impl
} // namespace TChem

#endif
