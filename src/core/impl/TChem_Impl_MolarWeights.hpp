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
#ifndef __TCHEM_IMPL_MOLAR_WEIGHTS_HPP__
#define __TCHEM_IMPL_MOLAR_WEIGHTS_HPP__

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"

namespace TChem {

namespace Impl {
/// mean molecular weight
/// Computes molar weight based on species mass fraction (getMs2Wmix)
///    \f[ W_{mix}=\left(\sum_{k=1}^{N_{spec}}Y_k/W_k\right)^{-1} \f]
/// input mass fraction
template<typename ValueType, typename DeviceType>
struct MolarWeights
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;
  using real_type = scalar_type;

  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type = TChem::KineticModelConstData<device_type>;
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static value_type team_invoke(
    const MemberType& member,
    /// input
    const value_type_1d_view_type& Ys, /// mass fractions
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {

    using reducer_type = Tines::SumReducer<value_type>;

    typename reducer_type::value_type wmix(0);
    // mass fraction is the input for this function?
    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i, typename reducer_type::value_type& update) {
        update += Ys(i) / kmcd.sMass(i);
      },
      reducer_type(wmix));
    wmix = real_type(1.) / wmix;
    return wmix;
  }
};

/// mean molecular weight
/// Computes molar weight based on species mole fraction
template<typename ValueType, typename DeviceType>
struct MeanMolecularWeightXc
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;
  using real_type = scalar_type;

  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type = TChem::KineticModelConstData<device_type>;

  // computes mean Molecular weight
  // inputs mole fraction
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static value_type team_invoke(
    const MemberType& member,
    /// input
    const value_type_1d_view_type& Xc, /// mole fractions
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {

    using reducer_type = Tines::SumReducer<value_type>;

    typename reducer_type::value_type wmix(0);
    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i, typename reducer_type::value_type& update) {
        update += Xc(i) * kmcd.sMass(i);
      },
      reducer_type(wmix));
    return wmix;
  }
};

// converts species mass fraction  to species mole fraction
template<typename ValueType, typename DeviceType>
struct MassFractionToMoleFraction
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;
  using real_type = scalar_type;

  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type = TChem::KineticModelConstData<device_type>;

  // Computes Mole fraction from Mass fraction
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const value_type_1d_view_type& Ys, /// mass fractions
    /// const input from kinetic model
    const value_type_1d_view_type& Xc,
    const kinetic_model_type& kmcd)
  {

    const value_type Wmix = MolarWeights<value_type,device_type>::team_invoke(member, Ys, kmcd);

    // is mass fraction  the input for this function?
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i) { Xc(i) = Ys(i) * Wmix / kmcd.sMass(i); });
  }
};
//converts species mole fraction to species mass fraction
template<typename ValueType, typename DeviceType>
struct MoleFractionToMassFraction
{
  // Computes Mass fraction from Mole fraction
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;
  using real_type = scalar_type;

  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type = TChem::KineticModelConstData<device_type>;

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const value_type_1d_view_type& Xc, /// mole fraction
    /// const input from kinetic model
    const value_type_1d_view_type& Ys, // mass fraction
    const kinetic_model_type& kmcd)
  {

    const value_type Wmix = MeanMolecularWeightXc<value_type,device_type>::team_invoke(member, Xc, kmcd);
    // mass fraction is the input for this function?
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i) { Ys(i) = Xc(i) * kmcd.sMass(i) / Wmix; });
  }
};

} // namespace Impl
} // namespace TChem

#endif
