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
#ifndef __TCHEM_IMPL_JACOBIAN_REDUCED_HPP__
#define __TCHEM_IMPL_JACOBIAN_REDUCED_HPP__

#include "TChem_Impl_Jacobian.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "TChem_Impl_RhoMixMs.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

struct JacobianReduced
{

  template<typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    using kmcd_type = KineticModelConstDataType;
    using device_type = typename kmcd_type::device_type;
    
    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<real_type,device_type>::getWorkSpaceSize(kmcd);

    const ordinal_type jac_dim_full = kmcd.nSpec + 3;
    const ordinal_type iter_size =
      (kmcd.nSpec > kmcd.nReac ? kmcd.nSpec : kmcd.nReac) * 2;
    const ordinal_type workspace_size =
      (jac_dim_full * jac_dim_full + 7 * kmcd.nSpec + 8 * kmcd.nReac +
       iter_size + 4 + work_kfor_rev_size);
    return workspace_size;
  }

  ///  \param t  : temperature [K]
  ///  \param p  : pressure []
  ///  \param Ys : array of \f$N_{spec}\f$ doubles \f$((XC_1,XC_2,...,XC_N)\f$:
  ///              molar concentrations XC \f$[kmol/m^3]\f$
  ///  \return jacobian : Jacobian matrix TYn
  ///
  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename OrdinalType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    /// output
    const RealType2DViewType& jac_reduced, /// (nSpec+1, nSpec+1),
    /// workspace
    const RealType2DViewType&
      jac_full, /// (jacDim, jacDim), jacDim = kmcd.nSpec+3
    const RealType1DViewType& omega,
    const RealType1DViewType& gk,
    const RealType1DViewType& gkp,
    const RealType1DViewType& hks,
    const RealType1DViewType& cpks,
    const RealType1DViewType& concX,
    const RealType1DViewType& concM,
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    const RealType1DViewType& crnd,
    const RealType1DViewType& ropFor,
    const RealType1DViewType& ropRev,
    const RealType1DViewType& kforp,
    const RealType1DViewType& krevp,
    const RealType1DViewType& CrndDer,
    const RealType1DViewType& PrDer,
    const RealType1DViewType& team_sum,
    const OrdinalType1DViewType& iter,
    const RealType1DViewType& work_kfor_rev,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    using kmcd_type = KineticModelConstDataType;
    using device_type = typename kmcd_type::device_type;
    using MolarWeights = MolarWeights<real_type,device_type>;
    using RhoMixMs = RhoMixMs<real_type,device_type>;
    // const real_type zero(0); //, one(1);
    // const real_type t_1 = one/t;
    const real_type tln = ats<real_type>::log(t);

    /// 0. compute full jac_full (nSpec+3)x(nSpec+3)
    const int dummy(0);
    Jacobian::team_invoke_detail(dummy,
                                 member,
                                 t,
                                 p,
                                 Ys,
                                 jac_full,
                                 omega,
                                 gk,
                                 gkp,
                                 hks,
                                 cpks,
                                 concX,
                                 concM,
                                 kfor,
                                 krev,
                                 crnd,
                                 ropFor,
                                 ropRev,
                                 kforp,
                                 krevp,
                                 CrndDer,
                                 PrDer,
                                 team_sum,
                                 iter,
                                 work_kfor_rev,
                                 kmcd);
    member.team_barrier();

    /// 1. compute rhomix and wmix
    const real_type rhomix = RhoMixMs::team_invoke(member, t, p, Ys, kmcd);
    const real_type wmix = MolarWeights::team_invoke(member, Ys, kmcd);

    /// 2. construct the reduced jacobian
    const ordinal_type jacDimReduced = kmcd.nSpec + 1;

    /// 2.1 modify the first column
    member.team_barrier();
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(member, jacDimReduced),
      [&](const ordinal_type& j) {
        if (j == 0) {
          /// modify the first column
          const real_type drhodT = -rhomix / t;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, jacDimReduced),
                               [&](const ordinal_type& i) {
                                 jac_reduced(i, 0) =
                                   jac_full(i + 2, 2) +
                                   jac_full(i + 2, 0) * drhodT;
                               });
        } else {
          /// modify the the rest of columns
          const real_type drhodY = -rhomix * wmix / kmcd.sMass(j - 1);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, jacDimReduced),
                               [&](const ordinal_type& i) {
                                 jac_reduced(i, j) =
                                   jac_full(i + 2, j + 2) +
                                   jac_full(i + 2, 0) * drhodY;
                               });
        }
      });
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    /// output
    const RealType2DViewType&
      jacobian, /// (jacDimReduced, jacDimRduced), jacDimReduced = kmcd.nSpec+1
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    static_assert(Kokkos::Impl::SpaceAccessibility<
                    typename RealType1DViewType::execution_space,
                    typename WorkViewType::memory_space>::accessible,
                  "RealType1DView is not accessible to workspace");
    static_assert(Kokkos::Impl::SpaceAccessibility<
                    typename RealType2DViewType::execution_space,
                    typename WorkViewType::memory_space>::accessible,
                  "RealType2DView is not accessible to workspace");

    ///
    /// workspace needed 6*nSpec + 10*nReac + 4;
    ///
    auto w = (real_type*)work.data();

    const ordinal_type jacDimFull = kmcd.nSpec + 3;
    const RealType2DViewType jac_full =
      RealType2DViewType(w, jacDimFull, jacDimFull);
    w += (jacDimFull * jacDimFull);
    const RealType1DViewType omega = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;

    const RealType1DViewType gk = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    const RealType1DViewType gkp = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    const RealType1DViewType hks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    const RealType1DViewType cpks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    const RealType1DViewType concX = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;

    const RealType1DViewType concM = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    const RealType1DViewType kfor = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    const RealType1DViewType krev = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    const RealType1DViewType crnd = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    const RealType1DViewType ropFor = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    const RealType1DViewType ropRev = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    const RealType1DViewType kforp = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    const RealType1DViewType krevp = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    const RealType1DViewType CrndDer = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    const RealType1DViewType PrDer = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    const RealType1DViewType team_sum = RealType1DViewType(w, 4);
    w += 4;

    /// iteration workspace
    const ordinal_type iter_size =
      (kmcd.nSpec > kmcd.nReac ? kmcd.nSpec : kmcd.nReac) * 2;
    const auto iter = Kokkos::View<ordinal_type*,
                                   Kokkos::LayoutRight,
                                   typename WorkViewType::memory_space>(
      (ordinal_type*)w, iter_size);
    w += iter_size;

    using kmcd_type = KineticModelConstDataType;
    using device_type = typename kmcd_type::device_type;

    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<real_type,device_type>::getWorkSpaceSize(kmcd) - iter_size;
    auto work_kfor_rev = RealType1DViewType(w, work_kfor_rev_size);
    w += work_kfor_rev_size;

    team_invoke_detail(member,
                       t,
                       p,
                       Ys,
                       jacobian,
                       jac_full,
                       omega,
                       gk,
                       gkp,
                       hks,
                       cpks,
                       concX,
                       concM,
                       kfor,
                       krev,
                       crnd,
                       ropFor,
                       ropRev,
                       kforp,
                       krevp,
                       CrndDer,
                       PrDer,
                       team_sum,
                       iter,
                       work_kfor_rev,
                       kmcd);
  }
};

} // namespace Impl
} // namespace TChem

#endif
