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
#ifndef __TCHEM_IMPL_SOURCE_PFR_TERM_HPP__
#define __TCHEM_IMPL_SOURCE_PFR_TERM_HPP__

#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_MolarConcentrations.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "TChem_Impl_ReactionRates.hpp"
#include "TChem_Impl_ReactionRatesSurface.hpp"
#include "TChem_Util.hpp"


namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct PlugFlowReactorRHS
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;

  using ordinal_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;

  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type      = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;
  using pfr_data_type = PlugFlowReactorData;

  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {
    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<value_type,device_type>::getWorkSpaceSize(kmcd);

    return (7 * kmcd.nSpec + 8 * kmcd.nReac + 5 * kmcdSurf.nSpec +
            5 * kmcdSurf.nReac + work_kfor_rev_size);
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec) mass fraction
    const value_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
    const value_type& density,
    const value_type& p,   // pressure
    const value_type& vel, // velocity

    /// output
    const value_type_1d_view_type& dT,   /// (1) //energy
    const value_type_1d_view_type& dYs,  /// (kmcd.nSpec) // species
    const value_type_1d_view_type& dZs,  /// (kmcdSurf.nSpec) // surface species
    const value_type_1d_view_type& drho, //(1) density
    const value_type_1d_view_type& du,   // velocity

    /// workspace
    const value_type_1d_view_type& omega,
    const value_type_1d_view_type& omegaSurfGas,
    const value_type_1d_view_type& omegaSurf,
    // gas species
    const value_type_1d_view_type& Xc,
    const value_type_1d_view_type& gk,
    const value_type_1d_view_type& hks,
    const value_type_1d_view_type& cpks,
    const value_type_1d_view_type& concX,
    const value_type_1d_view_type& concM,
    const value_type_1d_view_type& kfor,
    const value_type_1d_view_type& krev,
    const value_type_1d_view_type& ropFor,
    const value_type_1d_view_type& ropRev,
    const value_type_1d_view_type& Crnd,
    const ordinal_type_1d_view_type& iter,
    const value_type_1d_view_type& work_kfor_rev,
    // surface species
    const value_type_1d_view_type& Surf_gk,
    const value_type_1d_view_type& Surf_hks,
    const value_type_1d_view_type& Surf_cpks,
    const value_type_1d_view_type& concXSurf,
    const value_type_1d_view_type& kforSurf,
    const value_type_1d_view_type& krevSurf,
    const value_type_1d_view_type& ropForSurf,
    const value_type_1d_view_type& ropRevSurf,
    const value_type_1d_view_type& CoverageFactor,

    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    // const input for plug flow reactor
    const pfr_data_type& pfrd)
  {

    using EnthalpySpecMs = EnthalpySpecMsFcn<value_type,device_type>;
    using ReactionRates  = ReactionRates<value_type,device_type>;
    using ReactionRatesSurface = ReactionRatesSurface<value_type,device_type>;
    using MolarWeights = MolarWeights<value_type,device_type>;
    using CpMixMs = CpMixMs<value_type, device_type>;

    const real_type one(1);

    const real_type Area(pfrd.Area);
    const real_type Pcat(pfrd.Pcat);

    /// 1. compute molar reaction rates
    ReactionRates::team_invoke_detail(member,
                                       t,
                                       p,
                                       density,
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
                                       work_kfor_rev,
                                       kmcd);

    member.team_barrier();
    /// compute catalysis production rates
    ReactionRatesSurface ::team_invoke_detail(member,
                                              t,
                                              p,
                                              density,
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
                                              CoverageFactor, //
                                              kmcd,
                                              kmcdSurf);

    member.team_barrier();
    /// 3. compute cpmix
    const value_type cpmix = CpMixMs::team_invoke(member, t, Ys, cpks, kmcd);

    /// 4. compute species enthalies
    EnthalpySpecMs::team_invoke(member, t, hks, cpks, kmcd);

    /// 2. transform molar reaction rates to mass reaction rates
    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& k) {
                           omega(k) *= kmcd.sMass(k);        // kg/m3/2
                           omegaSurfGas(k) *= kmcd.sMass(k); // kg/m2/s
                         });

    member.team_barrier();

    using reducer_type = Tines::SumReducer<value_type>;

    typename reducer_type::value_type sumSkWk(0);

    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, typename reducer_type::value_type& update) {
        update += omegaSurfGas(k);
      },
      reducer_type(sumSkWk));

    typename reducer_type::value_type sumSkWkhk(0);
    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, typename reducer_type::value_type& update) {
        update += omegaSurfGas(k) * hks(k);
      },
      reducer_type(sumSkWkhk));

    typename reducer_type::value_type sumomgkWkhk(0);
    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, typename reducer_type::value_type& update) {
        update += omega(k) * hks(k); // Units of omega (kg/m3/s).
      },
      reducer_type(sumomgkWkhk));

    member.team_barrier();

    typename reducer_type::value_type sumgYkoWk(0);
    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, typename reducer_type::value_type& update) {
        dYs(k) =
          (Area * omega(k) + Pcat * omegaSurfGas(k) - Ys(k) * Pcat * sumSkWk) /
          (Area * density * vel);         // species equation
        update += dYs(k) / kmcd.sMass(k); // Units of omega (kg/m3/s).
      },
      reducer_type(sumgYkoWk) );

    // energy equation
    dT(0) =
      -(Area * sumomgkWkhk + Pcat * sumSkWkhk) / (Area * density * vel * cpmix);

    const value_type Wmix = MolarWeights::team_invoke(member, Ys, kmcd);
    // momentum equation
    const value_type coef1 = 1. - p / (density * vel * vel);
    const value_type coef2 = -coef1 + 2.;

    member.team_barrier();
    du(0) = (-vel * Pcat * coef2 * sumSkWk -
             Area * density * kmcd.Runiv * (dT(0) / Wmix + t * sumgYkoWk)) /
            (Area * density * vel * coef1);
    // continuity equation

    member.team_barrier();
    drho(0) = (-Area * density * du(0) + Pcat * sumSkWk) / vel / Area;

    typename reducer_type::value_type Zsum(0);
    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nSpec),
      [&](const ordinal_type& k, typename reducer_type::value_type& update) {
        dZs(k) =
          omegaSurf(k);  /// kmcdSurf.sitedensity; // surface species equation
        update += Zs(k); // Units of omega (kg/m3/s).
      },
      reducer_type(Zsum));

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
           typename WorkViewType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type_1d_view_type& Ys, /// (kmcd.nSpec)
    const real_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p, // pressure
    const real_type& u, // velocity
    /// output
    const real_type_1d_view_type& rhs, /// (kmcd.nSpec + 1)
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const pfr_data_type& pfrd)
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

    auto omega = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto omegaSurfGas = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto omegaSurf = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;

    auto Xc = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto gk = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto hks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto concX = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;

    auto concM = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto kfor = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto krev = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropFor = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropRev = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto Crnd = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;

    auto Surf_gk = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto Surf_hks = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto Surf_cpks = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto concXSurf = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;

    auto kforSurf = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto krevSurf = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto ropForSurf = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto ropRevSurf = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;

    auto CoverageFactor = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;

    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<real_type,device_type>::getWorkSpaceSize(kmcd);
    auto work_kfor_rev = real_type_1d_view_type(w, work_kfor_rev_size);
    w += work_kfor_rev_size;


    using ordinal_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;


    auto iter = ordinal_type_1d_view_type((ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;


    auto dT = real_type_1d_view_type(rhs.data(), 1);
    auto dYs = real_type_1d_view_type(rhs.data() + 1, kmcd.nSpec);
    auto drho = real_type_1d_view_type(rhs.data() + 1 + kmcd.nSpec, 1);
    auto du = real_type_1d_view_type(rhs.data() + 2 + kmcd.nSpec, 1);
    auto dZs = real_type_1d_view_type(rhs.data() + 3 + kmcd.nSpec, kmcdSurf.nSpec);

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
                       work_kfor_rev,
                       // surface
                       Surf_gk,
                       Surf_hks,
                       Surf_cpks,
                       concXSurf,
                       kforSurf,
                       krevSurf,
                       ropForSurf,
                       ropRevSurf,
                       CoverageFactor,
                       // data from surface and gas phases
                       kmcd,
                       kmcdSurf,
                       pfrd);
  }

  template<typename MemberType,
           typename WorkViewType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke_sacado(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec)
    const value_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
    const value_type& density,
    const value_type& p, // pressure
    const value_type& u, // velocity
    /// output
    const value_type_1d_view_type& rhs, /// (kmcd.nSpec + 1)
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const pfr_data_type& pfrd)
  {

    // We need to add 1 to the number of equation becase of sacado
    const ordinal_type sacadoStorageDimension = ats<value_type>::sacadoStorageDimension(u);
    auto w = (real_type*)work.data();
    const ordinal_type len = value_type().length();

    auto omega = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;
    auto omegaSurfGas = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;
    auto omegaSurf = value_type_1d_view_type(w, kmcdSurf.nSpec, sacadoStorageDimension);
    w += kmcdSurf.nSpec*len;

    auto Xc = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;
    auto gk = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;
    auto hks = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;
    auto cpks = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;
    auto concX = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;

    auto concM = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto kfor = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto krev = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto ropFor = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto ropRev = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto Crnd = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;

    auto Surf_gk = value_type_1d_view_type(w, kmcdSurf.nSpec, sacadoStorageDimension);
    w += kmcdSurf.nSpec*len;
    auto Surf_hks = value_type_1d_view_type(w, kmcdSurf.nSpec, sacadoStorageDimension);
    w += kmcdSurf.nSpec*len;
    auto Surf_cpks = value_type_1d_view_type(w, kmcdSurf.nSpec, sacadoStorageDimension);
    w += kmcdSurf.nSpec*len;
    auto concXSurf = value_type_1d_view_type(w, kmcdSurf.nSpec, sacadoStorageDimension);
    w += kmcdSurf.nSpec*len;

    auto kforSurf = value_type_1d_view_type(w, kmcdSurf.nReac, sacadoStorageDimension);
    w += kmcdSurf.nReac*len;
    auto krevSurf = value_type_1d_view_type(w, kmcdSurf.nReac, sacadoStorageDimension);
    w += kmcdSurf.nReac*len;
    auto ropForSurf = value_type_1d_view_type(w, kmcdSurf.nReac, sacadoStorageDimension);
    w += kmcdSurf.nReac*len;
    auto ropRevSurf = value_type_1d_view_type(w, kmcdSurf.nReac, sacadoStorageDimension);
    w += kmcdSurf.nReac*len;

    auto CoverageFactor = value_type_1d_view_type(w, kmcdSurf.nReac, sacadoStorageDimension);
    w += kmcdSurf.nReac*len;

    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<value_type,device_type>::getWorkSpaceSize(kmcd);
    auto work_kfor_rev = value_type_1d_view_type(w, work_kfor_rev_size, sacadoStorageDimension);
    w += work_kfor_rev_size*len;


    using ordinal_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;
    auto iter = ordinal_type_1d_view_type((ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;

    using range_type = Kokkos::pair<ordinal_type, ordinal_type>;

    const value_type_1d_view_type dT   = Kokkos::subview(rhs,range_type(0, 1));
    const value_type_1d_view_type dYs  = Kokkos::subview(rhs,range_type(1, kmcd.nSpec + 1));
    const value_type_1d_view_type drho = Kokkos::subview(rhs,range_type(kmcd.nSpec + 1, kmcd.nSpec + 2));
    const value_type_1d_view_type du   = Kokkos::subview(rhs,range_type(kmcd.nSpec + 2, kmcd.nSpec + 3));
    const value_type_1d_view_type dZs  = Kokkos::subview(rhs,range_type(kmcd.nSpec + 3, kmcd.nSpec + 3 + kmcdSurf.nSpec ));

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
                       work_kfor_rev,
                       // surface
                       Surf_gk,
                       Surf_hks,
                       Surf_cpks,
                       concXSurf,
                       kforSurf,
                       krevSurf,
                       ropForSurf,
                       ropRevSurf,
                       CoverageFactor,
                       // data from surface and gas phases
                       kmcd,
                       kmcdSurf,
                       pfrd);
  }
};

} // namespace Impl
} // namespace TChem

#endif
