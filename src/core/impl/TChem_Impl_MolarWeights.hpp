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

namespace TChem {

namespace Impl {
/// mean molecular weight
/// Computes molar weight based on species mass fraction (getMs2Wmix)
///    \f[ W_{mix}=\left(\sum_{k=1}^{N_{spec}}Y_k/W_k\right)^{-1} \f]
/// input mass fraction
struct MolarWeights
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static real_type team_invoke(
    const MemberType& member,
    /// input
    const RealType1DViewType& Ys, /// mass fractions
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    real_type wmix(0);
    // mass fraction is the input for this function?
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i, real_type& update) {
        update += Ys(i) / kmcd.sMass(i);
      },
      wmix);
    wmix = real_type(1) / wmix;
    return wmix;
  }
};

struct MeanMolecularWeightXc
{
  // computes mean Molecular weight
  // inputs mole fraction
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static real_type team_invoke(
    const MemberType& member,
    /// input
    const RealType1DViewType& Xc, /// mole fractions
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    real_type wmix(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i, real_type& update) {
        update += Xc(i) * kmcd.sMass(i);
      },
      wmix);
    return wmix;
  }
};

struct MassFractionToMoleFraction
{
  // Computes Mole fraction from Mass fraction
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const RealType1DViewType& Ys, /// mass fractions
    /// const input from kinetic model
    const RealType1DViewType& Xc,
    const KineticModelConstDataType& kmcd)
  {

    const real_type Wmix = MolarWeights::team_invoke(member, Ys, kmcd);

    // mass fraction is the input for this function?
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i) { Xc(i) = Ys(i) * Wmix / kmcd.sMass(i); });
  }
};

struct MoleFractionToMassFraction
{
  // Computes Mass fraction from Mole fraction
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const RealType1DViewType& Xc, /// mole fraction
    /// const input from kinetic model
    const RealType1DViewType& Ys, // mass fraction
    const KineticModelConstDataType& kmcd)
  {

    const real_type Wmix = MeanMolecularWeightXc::team_invoke(member, Xc, kmcd);

    // mass fraction is the input for this function?
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i) { Ys(i) = Xc(i) * kmcd.sMass(i) / Wmix; });
  }
};

} // namespace Impl
} // namespace TChem

#endif
