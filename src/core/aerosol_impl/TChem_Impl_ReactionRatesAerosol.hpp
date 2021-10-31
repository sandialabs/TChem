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
#ifndef __TCHEM_IMPL_REACTION_RATES_AEROSOL_HPP__
#define __TCHEM_IMPL_REACTION_RATES_AEROSOL_HPP__

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Impl_KForward.hpp"
// #define TCHEM_ENABLE_SERIAL_TEST_OUTPUT
namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct ReactionRatesAerosol
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using kinetic_model_type= KineticModelNCAR_ConstData<device_type>;

  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd)
  {
    ordinal_type workspace_size = 2 * kmcd.nReac;
    if (kmcd.nConstSpec > 0 ) {
      workspace_size +=  kmcd.nSpec;
    }

    return workspace_size;
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input temperature
    const value_type& t,
    const value_type& p,
    //// inputs
    const value_type_1d_view_type& concX,
    /// output
    const value_type_1d_view_type& omega, /// (kmcd.nSpec)
    // work
    const value_type_1d_view_type& ropFor,
    const value_type_1d_view_type& kfor,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {

    // for (ordinal_type i = 0; i < kmcd.nSpec; i++) {
    //   printf("i %d x %e\n",i, concX(i) );
    // }

    using kForward_type = TChem::Impl::KForward<value_type, device_type >;

    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec), [&](const ordinal_type& i) {
      omega(i) = real_type(0);
    });
    member.team_barrier();
    // sources
    const ordinal_type n_const_sources = kmcd.EmissionCoef.extent(0);
    auto EmissionCoef = kmcd.EmissionCoef;
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, n_const_sources), [&](const ordinal_type& i) {
      auto emission_coef_at_i = EmissionCoef(i);
      omega(emission_coef_at_i._species_index) = emission_coef_at_i._emissition_rate;
    });
    member.team_barrier();

    // reactions constants
    kForward_type::team_invoke( member, t, p, kfor, kmcd);

    member.team_barrier();
    // rate of progress
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        value_type ropFor_at_i = kfor(i);

        for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j);
          const ordinal_type niup = ats<ordinal_type>::abs(kmcd.reacNuki(i, j));
          ropFor_at_i *= ats<value_type>::pow(concX(kspec), niup);
        }

        ropFor(i) = ropFor_at_i;
        // printf("reaction i %d ropFor %e \n", i, ropFor(i));
    });

    member.team_barrier();

    auto rop = ropFor;
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        const value_type rop_at_i = rop(i);
        for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j);
          const value_type val = kmcd.reacNuki(i, j) * rop_at_i;
          Kokkos::atomic_add(&omega(kspec), val);
        }
        const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
        for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          const value_type val = kmcd.reacNuki(i, j + joff) * rop_at_i;
          Kokkos::atomic_add(&omega(kspec), val);
        }
      });

    member.team_barrier();


  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_sacado(
    const MemberType& member,
    /// input temperature
    const value_type& t,
    const value_type& p,
    //// inputs
    const value_type_1d_view_type& X,
    const real_type_1d_view_type& const_X,
    /// output
    const value_type_1d_view_type& omega, /// (kmcd.nSpec)
    // work
    const real_type_1d_view_type& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
    {

      auto w = (real_type*)work.data();
      const ordinal_type len = value_type().length();
      const ordinal_type sacadoStorageDimension = ats<value_type>::sacadoStorageDimension(t);

      auto ropFor = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
      w += kmcd.nReac*len;
      auto kfor = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
      w += kmcd.nReac*len;
      auto concX = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
      w += kmcd.nSpec*len;

      const ordinal_type n_active_vars = kmcd.nSpec-kmcd.nConstSpec;

      Kokkos::parallel_for(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, n_active_vars ),
         [&](const ordinal_type& i) {
        concX(i) = X(i);
      });
      // constant variables
      Kokkos::parallel_for(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nConstSpec),
         [&](const ordinal_type& i) {
        concX(i + n_active_vars) = const_X(i);
      });
      member.team_barrier();

      team_invoke_detail(member,
                         t,
                         p,
                         concX,
                         omega,
                         ropFor,
                         kfor,
                         kmcd);

    }
  //
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_sacado(
    const MemberType& member,
    /// input temperature
    const value_type& t,
    const value_type& p,
    //// inputs
    const value_type_1d_view_type& X,
    /// output
    const value_type_1d_view_type& omega, /// (kmcd.nSpec)
    // work
    const real_type_1d_view_type& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
    {

      auto w = (real_type*)work.data();
      const ordinal_type len = value_type().length();
      const ordinal_type sacadoStorageDimension = ats<value_type>::sacadoStorageDimension(t);

      auto ropFor = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
      w += kmcd.nReac*len;
      auto kfor = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
      w += kmcd.nReac*len;
      member.team_barrier();

      team_invoke_detail(member,
                         t,
                         p,
                         X,
                         omega,
                         ropFor,
                         kfor,
                         kmcd);

    }
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input temperature
    const real_type& t,
    const real_type& p,
    //// inputs
    const real_type_1d_view_type& X, // molar concentration of active species
    //// inputs
    const real_type_1d_view_type& const_X, // molar concentration of constant species
    /// output
    const real_type_1d_view_type& omega, /// (kmcd.nSpec)
    // work
    const real_type_1d_view_type& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
    {

      auto w = (real_type*)work.data();
      auto ropFor = real_type_1d_view_type(w, kmcd.nReac);
      w += kmcd.nReac;
      auto kfor = real_type_1d_view_type(w, kmcd.nReac);
      w += kmcd.nReac;

      auto concX = real_type_1d_view_type(w, kmcd.nSpec);
      w += kmcd.nSpec;

      const ordinal_type n_active_vars = kmcd.nSpec-kmcd.nConstSpec;

      Kokkos::parallel_for(
          Tines::RangeFactory<value_type>::TeamVectorRange(member, n_active_vars ),
          [&](const ordinal_type& i) {
            concX(i) = X(i);
      });
      // constant variables
      Kokkos::parallel_for(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nConstSpec),
          [=](const ordinal_type& i) {
            const ordinal_type idx(i + n_active_vars);
            concX(idx) = const_X(i);
          });
      member.team_barrier();

      team_invoke_detail(member,
                         t,
                         p,
                         concX,
                         omega,
                         ropFor,
                         kfor,
                         kmcd);

    }

  // source term assuming all species are not constant
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input temperature
    const real_type& t,
    const real_type& p,
    //// inputs
    const real_type_1d_view_type& X, // molar concentration of active species
    /// output
    const real_type_1d_view_type& omega, /// (kmcd.nSpec)
    // work
    const real_type_1d_view_type& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
    {

      auto w = (real_type*)work.data();
      auto ropFor = real_type_1d_view_type(w, kmcd.nReac);
      w += kmcd.nReac;
      auto kfor = real_type_1d_view_type(w, kmcd.nReac);
      w += kmcd.nReac;

      team_invoke_detail(member,
                         t,
                         p,
                         X,
                         omega,
                         ropFor,
                         kfor,
                         kmcd);

    }
};

} // namespace Impl
} // namespace TChem

#endif
