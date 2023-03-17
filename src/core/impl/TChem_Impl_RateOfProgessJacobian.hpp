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
#ifndef __TCHEM_IMPL_NET_PRODUCTION_RATE_JACOBIAN_HPP__
#define __TCHEM_IMPL_NET_PRODUCTION_RATE_JACOBIAN_HPP__

#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_Crnd.hpp"
#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_Gk.hpp"
#include "TChem_Impl_KForwardReverse.hpp"
#include "TChem_Impl_MolarConcentrations.hpp"
#include "TChem_Impl_RateOfProgress.hpp"
#include "TChem_Impl_RhoMixMs.hpp"
#include "TChem_Impl_ThirdBodyConcentrations.hpp"
#include "TChem_Util.hpp"

#include "TChem_Impl_ReactionRates.hpp"

namespace TChem {
namespace Impl {

// template<typename ControlType>
// KOKKOS_INLINE_FUNCTION bool
// JacobianEnableRealStoichiometricCoefficients(const ControlType& control)
// {
//   return true;
// }



struct RateOfProgessJacobian
{

  ///  \param t  : temperature [K]
  ///  \param p  : pressure []
  ///  \param Ys : array of \f$N_{spec}\f$ doubles \f$((XC_1,XC_2,...,XC_N)\f$:
  ///              molar concentrations XC \f$[kmol/m^3]\f$
  ///  \return jacobian : Jacobian matrix [stateVectorLength x
  ///  stateVectorLength]
  ///

  template<typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    using kmcd_type = KineticModelConstDataType;
    using device_type = typename kmcd_type::device_type;

    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<real_type,device_type>::getWorkSpaceSize(kmcd);

    const ordinal_type iter_size =
      (kmcd.nSpec > kmcd.nReac ? kmcd.nSpec : kmcd.nReac) * 2;
    const ordinal_type workspace_size =
      ( 7 * kmcd.nSpec + 8 * kmcd.nReac +
       iter_size + 4 + work_kfor_rev_size);
    return workspace_size;
  }

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
    const RealType2DViewType&
      jacobian, /// (jacDim, jacDim), jacDim = kmcd.nSpec+3
    /// workspace
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
    using Gk = GkFcn<real_type,device_type>;
    using EnthalpySpecMs = EnthalpySpecMsFcn<real_type,device_type>;
    using KForwardReverse= KForwardReverse<real_type,device_type>;
    using MolarConcentrations = MolarConcentrations<real_type,device_type>;
    using ThirdBodyConcentrations = ThirdBodyConcentrations<real_type,device_type>;
    using RateOfProgress = RateOfProgress<real_type,device_type>;
    using Crnd = Crnd<real_type,device_type>;
    using RhoMixMs = RhoMixMs<real_type,device_type>;
    using CpMixMs = CpMixMs<real_type, device_type>;

    const ordinal_type control(0);


    const real_type zero(0); //, one(1);
    // const real_type t_1 = one/t;
    const real_type tln = ats<real_type>::log(t);

    const real_type rhomix = RhoMixMs::team_invoke(member, t, p, Ys, kmcd);

    /// initialize and transform molar concentrations (kmol/m3) to (moles/cm3)
    {

      MolarConcentrations::team_invoke(member, t, p, rhomix, Ys, concX, kmcd);

      const real_type one_e_minus_three(1e-3);
      Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) {
                             omega(i) = zero;
                             concX(i) *= one_e_minus_three;
                             if (i < 4)
                               team_sum(i) = zero;
                           });
    }

    /// compute 3rd-body concentrations
    ThirdBodyConcentrations ::team_invoke(member,
                                          concX,
                                          concM, /// output
                                          kmcd);

    /// compute (-ln(T)+dS/R-dH/RT) for each species
    Gk::team_invoke(member, t, gk, hks, cpks,
                    kmcd); /// only need gk

    GkDerivative::team_invoke(member,
                              t,
                              gkp,
                              hks,
                              cpks,
                              kmcd); /// only need gkp

    /// compute forward and reverse rate constants
    KForwardReverse::team_invoke(member, t, p, gk, kfor, krev, iter, work_kfor_rev, kmcd);

    KForwardReverseDerivative::team_invoke(
      member, t, p, kforp, krevp, gkp, iter, kmcd);

    /// compute rate-of-progress
    RateOfProgress::team_invoke(
      member,  kfor, krev, concX, ropFor, ropRev, iter, kmcd);

    /// compute pressure dependent factors */
    Crnd::team_invoke(member, t, kfor, concX, concM, crnd, iter, kmcd);
    member.team_barrier();

    /// assemble reaction rates
    Kokkos::parallel_for(
      Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        const real_type rop_at_i = (ropFor(i) - ropRev(i)) * crnd(i);

        for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j);
          // omega(kspec) += kmcd.reacNuki(i,j)*rop_at_i;
          const real_type val = kmcd.reacNuki(i, j) * rop_at_i;
          Kokkos::atomic_fetch_add(&omega(kspec), val);
        }
        const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
        for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          // omega(kspec) += kmcd.reacNuki(i,j+joff)*rop_at_i;
          const real_type val = kmcd.reacNuki(i, j + joff) * rop_at_i;
          Kokkos::atomic_fetch_add(&omega(kspec), val);
        }
      });

    member.team_barrier();

    /// transform from mole/(cm3.s) to kg/(m3.s)
    {
      const real_type one_e_3(1e3);
      Kokkos::parallel_for(
        Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i) { omega(i) *= one_e_3 * kmcd.sMass(i); });
    }

    /// assemble jacobian terms, F[3+i,2...]
    {
      Kokkos::parallel_for(
        Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.jacDim * kmcd.jacDim),
        [&](const ordinal_type& i) {
          const int k0 = i / kmcd.jacDim, k1 = i % kmcd.jacDim;
          jacobian(k0, k1) = zero;
        });
    }
    member.team_barrier();

    {
      ordinal_type arbord(0);
      for (ordinal_type j = 0, iord = 0, itbdy = 0, ipfal = 0; j < kmcd.nReac;
           ++j, iord += arbord) {
        real_type CrndDt;
        /// compute {\partial Crnd_j}/{\partial T} and {\partial
        /// Crnd_j}/{\partial Y1,Y2,...,YN}
        CrndDerivative::team_invoke(member,
                                    j,
                                    t,
                                    kfor,
                                    concX,
                                    concM,
                                    itbdy,
                                    ipfal,
                                    CrndDt,
                                    CrndDer,
                                    PrDer,
                                    kmcd);

        arbord = ((iord < kmcd.nOrdReac) && (kmcd.reacAOrd(iord) == j));
        /// const j variables
        const real_type crnd_at_j = crnd(j), ropFor_at_j = ropFor(j),
                        ropRev_at_j = ropRev(j),
                        rop_at_j = ropFor_at_j - ropRev_at_j,
                        kforp_at_j = kforp(j), krevp_at_j = krevp(j);

        //
        // 2  is temperature
        const real_type dqidT = (CrndDt * rop_at_j + crnd_at_j *
        (ropFor_at_j * kforp_at_j - ropRev_at_j * krevp_at_j) )*1e3;

        jacobian(j,1) = dqidT;
        printf("CrndDt %e kforp_at_j %e  krevp_at_j %e term %e \n", CrndDt , kforp_at_j, krevp_at_j, crnd_at_j *
        (ropFor_at_j * kforp_at_j - ropRev_at_j * krevp_at_j));

        {

          if (kmcd.reacScoef(j) == -1) {
            /// reaction has integer stoichiometric coefficients
            Kokkos::parallel_for(
              Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
              [&](const ordinal_type& i) {

                const real_type nuij(kmcd.NuIJ(j, i));
                jacobian(j, i + 2) = CrndDer(i) * rop_at_j;
                if (nuij < 0 ) {
                      real_type qfor(-1), qrev(0);
                      RateOfProgressDerivative::serial_invoke(
                        control, j, i, kfor, krev, concX, qfor, qrev, kmcd);
                      jacobian(j, i + 2) +=  crnd_at_j * qfor;
                 }

                 if (nuij > 0 ){
                      real_type qfor(0), qrev(-1);
                      RateOfProgressDerivative::serial_invoke(
                        control, j, i, kfor, krev, concX, qfor, qrev, kmcd);
                      jacobian(j,  i + 2) -=  crnd_at_j * qrev;
                      // printf("reac %d sp %d qrev %e\n", j, i, qrev );
                  }

                  jacobian(j,  i + 2) *= rhomix/kmcd.sMass(i);

            }); /* done loop over species (counter i) for integer */
          }
        }
    }

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("Jacobian.team_invoke.test.out", "a+");
      fprintf(fs, ":: Jacobian::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, t %e, p %e\n",
              kmcd.nSpec,
              kmcd.nReac,
              t,
              p);
      for (int i = 0; i < int(Ys.extent(0)); ++i)
        fprintf(fs, "     i %3d, Ys %e\n", i, Ys(i));
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(jacobian.extent(0)); ++i) {
        fprintf(fs, "     i %3d ", i);
        for (int j = 0; j < int(jacobian.extent(1)); ++j)
          fprintf(fs, " % 3.2e", jacobian(i, j));
        fprintf(fs, "\n");
      }
    }
#endif
  }

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
      jacobian, /// (jacDim, jacDim), jacDim = kmcd.nSpec+3
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    using kmcd_type = KineticModelConstDataType;
    using device_type = typename kmcd_type::device_type;

    static_assert(Kokkos::SpaceAccessibility<
                    typename RealType1DViewType::execution_space,
                    typename WorkViewType::memory_space>::accessible,
                  "RealType1DView is not accessible to workspace");
    static_assert(Kokkos::SpaceAccessibility<
                    typename RealType2DViewType::execution_space,
                    typename WorkViewType::memory_space>::accessible,
                  "RealType2DView is not accessible to workspace");

    ///
    /// workspace needed 6*nSpec + 10*nReac + 4;
    ///
    auto w = (real_type*)work.data();
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

    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<real_type,device_type>::getWorkSpaceSize(kmcd);
    auto work_kfor_rev = RealType1DViewType(w, work_kfor_rev_size);
    w += work_kfor_rev_size;

    team_invoke_detail(member,
                       t,
                       p,
                       Ys,
                       jacobian,
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
