/* =====================================================================================
TChem version 2.1.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

This file is part of TChem. TChem is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */


#ifndef __TCHEM_IMPL_SOURCE_PFR_TERM_HPP__
#define __TCHEM_IMPL_SOURCE_PFR_TERM_HPP__

#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_MolarConcentrations.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "TChem_Impl_ReactionRates.hpp"
#include "TChem_Impl_ReactionRatesSurface.hpp"
#include "TChem_Impl_RhoMixMs.hpp"
#include "TChem_Util.hpp"
namespace TChem {
namespace Impl {

struct PlugFlowReactorRHS
{
  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    return (7 * kmcd.nSpec + 8 * kmcd.nReac + 5 * kmcdSurf.nSpec +
            6 * kmcdSurf.nReac);
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename OrdinalType1DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename PlugFlowReactorConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec) mass fraction
    const RealType1DViewType& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p,   // pressure
    const real_type& vel, // velocity

    /// output
    const RealType1DViewType& dT,   /// (1) //energy
    const RealType1DViewType& dYs,  /// (kmcd.nSpec) // species
    const RealType1DViewType& dZs,  /// (kmcdSurf.nSpec) // surface species
    const RealType1DViewType& drho, //(1) density
    const RealType1DViewType& du,   // velocity

    /// workspace
    const RealType1DViewType& omega,
    const RealType1DViewType& omegaSurfGas,
    const RealType1DViewType& omegaSurf,
    // gas species
    const RealType1DViewType& Xc,
    const RealType1DViewType& gk,
    const RealType1DViewType& hks,
    const RealType1DViewType& cpks,
    const RealType1DViewType& concX,
    const RealType1DViewType& concM,
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    const RealType1DViewType& ropFor,
    const RealType1DViewType& ropRev,
    const RealType1DViewType& Crnd,
    const OrdinalType1DViewType& iter,
    // surface species
    const RealType1DViewType& Surf_gk,
    const RealType1DViewType& Surf_hks,
    const RealType1DViewType& Surf_cpks,
    const RealType1DViewType& concXSurf,
    const RealType1DViewType& kforSurf,
    const RealType1DViewType& krevSurf,
    const RealType1DViewType& ropForSurf,
    const RealType1DViewType& ropRevSurf,
    const OrdinalType1DViewType& iterSurf,

    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    // const input for plug flow reactor
    const PlugFlowReactorConstDataType& pfrd)
  {

    const real_type one(1);

    const real_type Area(pfrd.Area);
    const real_type Pcat(pfrd.Pcat);

    /// 0. convert Ys to Xc
    // mass fraction to concentration
    // MolarConcentrations
    //   ::team_invoke(member,
    //                 t, p, Ys,
    //                 Xc,
    //                 kmcd);

    /// 1. compute molar reaction rates
    ReactionRates ::team_invoke_detail(member,
                                       t,
                                       p,
                                       Ys,
                                       omega,
                                       gk,
                                       hks,
                                       cpks,
                                       concX,
                                       concM,
                                       kfor,
                                       krev,
                                       ropFor,
                                       ropRev,
                                       Crnd,
                                       iter,
                                       kmcd);

    member.team_barrier();
    /// compute catalysis production rates
    ReactionRatesSurface ::team_invoke_detail(member,
                                              t,
                                              p,
                                              Ys,
                                              Zs,
                                              omegaSurfGas,
                                              omegaSurf,
                                              gk,
                                              hks,
                                              cpks,
                                              Surf_gk,
                                              Surf_hks,
                                              Surf_cpks,
                                              concX,
                                              concXSurf,
                                              kforSurf,
                                              krevSurf,
                                              ropForSurf,
                                              ropRevSurf,
                                              iterSurf, //
                                              kmcd,
                                              kmcdSurf);

    /// 3. compute density, cpmix
    const real_type rhomix = RhoMixMs::team_invoke(member, t, p, Ys, kmcd);
    const real_type cpmix = CpMixMs::team_invoke(member, t, Ys, cpks, kmcd);

    /// 4. compute species enthalies
    EnthalpySpecMs ::team_invoke(member, t, hks, cpks, kmcd);

    /// 2. transform molar reaction rates to mass reaction rates
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

    real_type sumSkWkhk(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        update += omegaSurfGas(k) * hks(k);
      },
      sumSkWkhk);

    real_type sumomgkWkhk(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        update += omega(k) * hks(k); // Units of omega (kg/m3/s).
      },
      sumomgkWkhk);

    member.team_barrier();

    real_type sumgYkoWk(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        dYs(k) =
          (Area * omega(k) + Pcat * omegaSurfGas(k) - Ys(k) * Pcat * sumSkWk) /
          (Area * density * vel);         // species equation
        update += dYs(k) / kmcd.sMass(k); // Units of omega (kg/m3/s).
      },
      sumgYkoWk);

    // energy equation
    dT(0) =
      -(Area * sumomgkWkhk + Pcat * sumSkWkhk) / (Area * density * vel * cpmix);

    const real_type Wmix = MolarWeights::team_invoke(member, Ys, kmcd);
    // momentum equation
    const real_type coef1 = 1. - p / (density * vel * vel);
    const real_type coef2 = -coef1 + 2.;

    member.team_barrier();
    du(0) = (-vel * Pcat * coef2 * sumSkWk -
             Area * density * kmcd.Runiv * (dT(0) / Wmix + t * sumgYkoWk)) /
            (Area * density * vel * coef1);
    // continuity equation

    member.team_barrier();
    drho(0) = (-Area * density * du(0) + Pcat * sumSkWk) / vel / Area;

    real_type Zsum(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcdSurf.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        dZs(k) =
          omegaSurf(k);  /// kmcdSurf.sitedensity; // surface species equation
        update += Zs(k); // Units of omega (kg/m3/s).
      },
      Zsum);

    member.team_barrier();
    dZs(kmcdSurf.nSpec - 1) = one - Zsum;
    member.team_barrier();

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("PlugFlowReactorRHS.team_invoke.test.out", "a+");
      fprintf(fs, ":: PlugFlowReactorRHS::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "Surface::     nSpec %3d, nReac %3d, site density %e\n",
              kmcdSurf.nSpec,
              kmcdSurf.nReac,
              kmcdSurf.sitedensity);
      fprintf(fs, "Gas::     nSpec %3d, nReac %3d\n", kmcd.nSpec, kmcd.nReac);
      fprintf(fs, "  Area %e,  pfrd.Pcat %e", pfrd.Area, pfrd.Pcat);
      fprintf(fs, "  t %e, p %e, velocity %e\n", t, p, vel);
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
      fprintf(fs, " drho(0) %e, \n", drho(0));
      fprintf(fs, " du(0) %e, \n", du(0));
      fprintf(fs, " dT(0) %e, \n", dT(0));
      fprintf(fs, ":::: output\n");
    }
#endif
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename PlugFlowReactorConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    const RealType1DViewType& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p, // pressure
    const real_type& u, // velocity
    /// output
    const RealType1DViewType& rhs, /// (kmcd.nSpec + 1)
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const PlugFlowReactorConstDataType& pfrd)
  {
    // const real_type zero(0);

    const ordinal_type len_rhs_w = getWorkSpaceSize(kmcd, kmcdSurf);
    if (len_rhs_w > ordinal_type(work.extent(0))) {
      Kokkos::abort("Error: workspace used is smaller than it "
                    "required::TChem_Impl_PlugFlowReactorRHS\n");
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

    auto Xc = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto gk = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto hks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto concX = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;

    auto concM = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto kfor = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto krev = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropFor = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropRev = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto Crnd = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;

    auto Surf_gk = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto Surf_hks = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto Surf_cpks = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto concXSurf = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;

    auto kforSurf = RealType1DViewType(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto krevSurf = RealType1DViewType(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto ropForSurf = RealType1DViewType(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto ropRevSurf = RealType1DViewType(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;

    auto iter = Kokkos::View<ordinal_type*,
                             Kokkos::LayoutRight,
                             typename WorkViewType::memory_space>(
      (ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;

    auto iterSurf = Kokkos::View<ordinal_type*,
                                 Kokkos::LayoutRight,
                                 typename WorkViewType::memory_space>(
      (ordinal_type*)w, kmcdSurf.nReac * 2);
    w += kmcdSurf.nReac * 2;

    auto dT = RealType1DViewType(rhs.data(), 1);
    auto dYs = RealType1DViewType(rhs.data() + 1, kmcd.nSpec);
    auto drho = RealType1DViewType(rhs.data() + 1 + kmcd.nSpec, 1);
    auto du = RealType1DViewType(rhs.data() + 2 + kmcd.nSpec, 1);
    auto dZs = RealType1DViewType(rhs.data() + 3 + kmcd.nSpec, kmcdSurf.nSpec);

    team_invoke_detail(member,
                       t,
                       Ys,
                       Zs,
                       density,
                       p,
                       u, // y variables
                       dT,
                       dYs,
                       dZs,
                       drho,
                       du, // rhs dydz
                       /// workspace
                       omega,
                       omegaSurfGas,
                       omegaSurf,
                       // gas
                       Xc,
                       gk,
                       hks,
                       cpks,
                       concX,
                       concM,
                       kfor,
                       krev,
                       ropFor,
                       ropRev,
                       Crnd,
                       iter,
                       // surface
                       Surf_gk,
                       Surf_hks,
                       Surf_cpks,
                       concXSurf,
                       kforSurf,
                       krevSurf,
                       ropForSurf,
                       ropRevSurf,
                       iterSurf,
                       // data from surface and gas phases
                       kmcd,
                       kmcdSurf,
                       pfrd);
  }
};

} // namespace Impl
} // namespace TChem

#endif
