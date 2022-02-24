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
#ifndef __TCHEM_IMPL_ISOTHERMAL_TRANSIENTCONTSTIRREDTANKREACTORRHS_HPP__
#define __TCHEM_IMPL_ISOTHERMAL_TRANSIENTCONTSTIRREDTANKREACTORRHS_HPP__

#include "TChem_Util.hpp"
#include "TChem_Impl_ReactionRates.hpp"
#include "TChem_NetProductionRatePerMass.hpp"
#include "TChem_Impl_ReactionRatesSurface.hpp"

namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct IsothermalTransientContStirredTankReactorRHS
{

  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;

  using ordinal_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;

  /// sacado is value type
  using value_type_0d_view_type = Tines::value_type_0d_view<value_type,device_type>;
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type      = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;
  using cstr_data_type = TransientContStirredTankReactorData<device_type>;

  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {
    return (6*kmcd.nSpec + 8*kmcd.nReac + 5*kmcdSurf.nSpec + 5*kmcdSurf.nReac + 1);
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec) mass fraction
    const value_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
    const value_type& density,
    const real_type& p,   // pressure
    // output
    const value_type_1d_view_type& dYs,  /// (kmcd.nSpec) // species
    const value_type_1d_view_type& dZs,  /// (kmcdSurf.nSpec) // surface species
    const value_type_0d_view_type& m_out,

    /// workspace
    const value_type_1d_view_type& omega,
    const value_type_1d_view_type& omegaSurfGas,
    const value_type_1d_view_type& omegaSurf,
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
    const cstr_data_type& cstr)
  {
    const ordinal_type idx = cstr.poisoning_species_idx;
    const value_type Acat = idx > 0 ?
                            cstr.Acat * (real_type(1) - Zs(idx)) : value_type(cstr.Acat);

    // std::cout << "Acat: " << Acat <<" cstr.Acat "<< cstr.Acat <<" poisoning_species_idx: "<< cstr.poisoning_species_idx<< '\n';

    using ReactionRatesSurface = Impl::ReactionRatesSurface<value_type,device_type>;
    using reducer_type = Tines::SumReducer<value_type>;

    ///  compute molar reaction rates
    ReactionRates<value_type,device_type>::team_invoke_detail(member,
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
                                       kmcd);

    // compute mix enthalpy
    member.team_barrier();
    /// 2. compute catalysis production rates
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

    /// 3. transform molar reaction rates to mass reaction rates
    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& k) {
                           omega(k) *= kmcd.sMass(k);        // kg/m3/2
                           omegaSurfGas(k) *= kmcd.sMass(k); // kg/m2/s
                         });

    member.team_barrier();

    typename reducer_type::value_type sumSkWk(0);
    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, typename reducer_type::value_type& update) {
        update += omegaSurfGas(k);
      },
      reducer_type(sumSkWk));

    member.team_barrier();

    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k) {
        dYs(k) = (omega(k) * cstr.Vol + // gas contribution from gas phase
                  omegaSurfGas(k) * Acat + // gas contribution from surface phase
                   cstr.mdotIn * (cstr.Yi(k) -  Ys(k)) - Ys(k) * sumSkWk * Acat ) /
                  (density * cstr.Vol);         // species equation
      });

    const real_type ten(10.0);
    const ordinal_type number_of_ode_surface = kmcdSurf.nSpec - cstr.number_of_algebraic_constraints;

    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, number_of_ode_surface),
      [&](const ordinal_type& k) {
        // units of sitedensity are mol/cm2 (ten)
        // printf("inside ode part k %d \n", k );
        dZs(k) = omegaSurf(k)/kmcdSurf.sitedensity/ten;  /// ; // surface species equation
    });

    if (cstr.number_of_algebraic_constraints > 0) {

      Kokkos::parallel_for(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, cstr.number_of_algebraic_constraints),
        [=](const ordinal_type& k) {
          // printf("inside constraint part k %d k + number_of_ode_surface %d \n", k, k + number_of_ode_surface  );
          const ordinal_type idx(k + number_of_ode_surface);
          dZs(idx) = omegaSurf(idx );  ///  surface species equation
      });

      typename reducer_type::value_type Zsum(0);
      Kokkos::parallel_reduce(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nSpec),
        [&](const ordinal_type& k, typename reducer_type::value_type& update) {
          update += dZs(k); // Units of omega (kg/m3/s).
        },reducer_type(Zsum));

      member.team_barrier();
      const value_type one(1);
      dZs(kmcdSurf.nSpec - 1) = one - Zsum;

    }

    member.team_barrier();


// #if !defined (TCHEM_CSTR_ALLOW_NEGATIVE_MASS_FLOW_OUTLET)
    //
    const value_type Wmix = MolarWeights<value_type, device_type>
    ::team_invoke(member, Ys, kmcd);

    typename reducer_type::value_type sum_dYkdt_wk(0);
    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, typename reducer_type::value_type& update) {
        update += dYs(k)/kmcd.sMass(k);
      }, reducer_type(sum_dYkdt_wk));

    member.team_barrier();


    m_out() = density * cstr.Vol * Wmix * sum_dYkdt_wk + cstr.mdotIn + sumSkWk * Acat;
    // m_out = ats<value_type>::value(m_out_value_type);
    // std::cout << "m_out_value_type " <<m_out_value_type<< '\n';

    // printf("Outlet mass flow rate %e\n", m_out);


    // if (m_out < 0 ){
// #if !defined(__CUDA_ARCH__)
//       std::cout << "sumSkWk " << sumSkWk << std::endl;
//       std::cout << "cstr.mdotIn  " << cstr.mdotIn  << std::endl;
//       std::cout << "m_out is negative " << m_out << std::endl;
// #endif

      // Kokkos::abort("outlet mass flow is negative ");

    // }

// #endif

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("IsothermalTransientContStirredTankReactorRHS.team_invoke.test.out", "a+");
      fprintf(fs, ":: IsothermalTransientContStirredTankReactorRHS::team_invoke\n");
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
           typename WorkViewType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type_1d_view_type& Ys, /// (kmcd.nSpec)
    const real_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p, // pressure
    /// output
    const real_type_1d_view_type& rhs, /// (kmcd.nSpec + 1)
    const real_type_0d_view_type& m_out, // outlet mass flow rate
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    // const input for plug flow reactor
    const cstr_data_type& cstr)
  {

    const ordinal_type len_rhs_w = getWorkSpaceSize(kmcd, kmcdSurf);
    if (len_rhs_w > ordinal_type(work.extent(0))) {
      Kokkos::abort("Error: workspace used is smaller than it "
                    "required::TChem_Impl_IsothermalTransientContStirredTankReactorRHS\n");
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


    using ordinal_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;

    auto iter = ordinal_type_1d_view_type((ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;

    // auto m_out_value_type = real_type_1d_view_type(w, 1);
    // w += 1;

    auto dYs = real_type_1d_view_type(rhs.data(), kmcd.nSpec);
    auto dZs = real_type_1d_view_type(rhs.data() + kmcd.nSpec, kmcdSurf.nSpec);

    team_invoke_detail(member,
                       t,
                       Ys,
                       Zs,
                       density,
                       p,
                       dYs,
                       dZs,
                       m_out,
                       /// workspace
                       omega,
                       omegaSurfGas,
                       omegaSurf,
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
                       // surface species
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
                       cstr);
  }

   //
   template<typename MemberType>
   KOKKOS_FORCEINLINE_FUNCTION static void team_invoke_sacado(
     const MemberType& member,
     /// input
     const real_type& t,
     const value_type_1d_view_type& Ys, /// (kmcd.nSpec)
     const value_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
     const value_type& density,
     const real_type& p, // pressure
     /// output
     const value_type_1d_view_type& rhs, /// (kmcd.nSpec + 1)
     const real_type_0d_view_type& m_out, // outlet mass flow rate
     /// workspace
     const real_type_1d_view_type& work,
     /// const input from kinetic model
     const kinetic_model_type& kmcd,
     const kinetic_surf_model_type& kmcdSurf,
     // const input for plug flow reactor
     const cstr_data_type& cstr)
     {
       auto w = (real_type*)work.data();
       const ordinal_type sacadoStorageDimension = ats<value_type>::sacadoStorageDimension(Ys(0));
       const ordinal_type len = value_type().length();

       auto omega = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
       w += kmcd.nSpec*len;
       auto omegaSurfGas = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
       w += kmcd.nSpec*len;
       auto omegaSurf = value_type_1d_view_type(w, kmcdSurf.nSpec, sacadoStorageDimension);
       w += kmcdSurf.nSpec*len;

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

       using ordinal_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;
       auto iter = ordinal_type_1d_view_type((ordinal_type*)w, kmcd.nReac * 2);
       w += kmcd.nReac * 2;

       auto m_out_value_type = value_type_0d_view_type(w, 1, sacadoStorageDimension);
       w += 1*len;

       using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
       const value_type_1d_view_type dYs  = Kokkos::subview(rhs,range_type(0, kmcd.nSpec));
       const value_type_1d_view_type dZs  = Kokkos::subview(rhs,range_type(kmcd.nSpec, kmcd.nSpec + kmcdSurf.nSpec ));

       team_invoke_detail(member,
                          t,
                          Ys,
                          Zs,
                          density,
                          p,
                          dYs,
                          dZs,
                          m_out_value_type,
                          /// workspace
                          omega,
                          omegaSurfGas,
                          omegaSurf,
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
                          // surface species
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
                          cstr);
      //
      // std::cout << "m_out2 " << m_out_value_type(0)<< '\n';
      m_out() = m_out_value_type().val();
      // printf("m_out %e\n", m_out );

     }
};

} // namespace Impl
} // namespace TChem

#endif
