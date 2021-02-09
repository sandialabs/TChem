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


#ifndef __TCHEM_IMPL_SURFACE_HPP__
#define __TCHEM_IMPL_SURFACE_HPP__

#include "TChem_Impl_ReactionRatesSurface.hpp"
#include "TChem_Util.hpp"
namespace TChem {
namespace Impl {

struct SurfaceRHS
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    const ordinal_type per_team_extent =
      ReactionRatesSurface::getWorkSpaceSize(kmcd, kmcdSurf);

    return (per_team_extent + kmcd.nSpec + kmcdSurf.nSpec);
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec) mass fraction
    const RealType1DViewType& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& p,           // pressure

    /// output
    const RealType1DViewType& dZs, /// (kmcdSurf.nSpec) // surface species
    const RealType1DViewType& omegaSurfGas,
    const RealType1DViewType& omegaSurf,
    const WorkViewType& work,
    /// workspace

    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    const real_type ten(10.0);
    /// compute catalysis production rates
    ReactionRatesSurface ::team_invoke(
      member, t, p, Ys, Zs, omegaSurfGas, omegaSurf, work, kmcd, kmcdSurf);

    member.team_barrier();

    // Kokkos::parallel_for
    //   (Kokkos::TeamVectorRange(member, kmcdSurf.nSpec),
    //   [&](const ordinal_type &k) {
    //   dZs(k) = omegaSurf(k)/kmcdSurf.sitedensity; // surface species equation
    // });
    const real_type one(1);

    real_type Zsum(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcdSurf.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        dZs(k) =
          omegaSurf(k) / kmcdSurf.sitedensity / ten; // surface species equation
        update += Zs(k);                       // Units of omega (kg/m3/s).
      },
      Zsum);

    member.team_barrier();
    dZs(kmcdSurf.nSpec - 1) = one - Zsum;

    member.team_barrier();

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("SurfaceRHS.team_invoke.test.out", "a+");
      fprintf(fs, ":: SurfaceRHS::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, site density %e\n",
              kmcdSurf.nSpec,
              kmcdSurf.nReac,
              kmcdSurf.sitedensity);
      fprintf(fs, "  t %e, p %e \n", t, p);
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "   i %3d,  Ys %e, \n", i, Ys(i));
      for (int i = 0; i < kmcdSurf.nSpec; ++i)
        fprintf(fs, "   i %3d,  Zs %e, \n", i, Zs(i));

      fprintf(fs, ":::: output\n");
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "   i %3d,  omegaSurfGas %e, \n", i, omegaSurfGas(i));
      for (int i = 0; i < kmcdSurf.nSpec; ++i)
        fprintf(fs, "   i %3d,  omegaSurf %e, \n", i, omegaSurf(i));
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
           typename KineticSurfModelConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    const RealType1DViewType& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& p,           // pressure
    /// output
    const RealType1DViewType& rhs, /// (kmcdSurf.nSpec )
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {

    auto w = (real_type*)work.data();

    auto omegaSurfGas = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto omegaSurf = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;

    auto work_surf = WorkViewType(w, work.extent(0) - (w - work.data()));
    team_invoke_detail(member,
                       t,
                       Ys,
                       Zs,
                       p,   // y variables
                       rhs, // rhs dydz
                       omegaSurfGas,
                       omegaSurf,
                       /// workspace
                       work_surf,
                       // data from surface and gas phases
                       kmcd,
                       kmcdSurf);
  }
};

// Jac only for surface phase
struct SurfaceNumJacobian
{
  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    const ordinal_type per_team_extent =
      SurfaceRHS::getWorkSpaceSize(kmcd, kmcdSurf);

    return (per_team_extent + 3 * kmcdSurf.nSpec);
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec) mass fraction
    const RealType1DViewType& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& p,           // pressure
    /// output
    const RealType2DViewType& Jac,
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    using kmcd_type = KineticModelConstDataType;
    using real_type_1d_view_type = typename kmcd_type::real_type_1d_view_type;

    const real_type small(1e-23);
    const real_type reltol(kmcdSurf.TChem_reltol);
    const real_type abstol(kmcdSurf.TChem_abstol);

    auto wptr = work.data();
    const ordinal_type Nspec = kmcdSurf.nSpec;

    /// trbdf workspace
    auto f = real_type_1d_view_type(wptr, Nspec);
    wptr += Nspec;
    auto fp = real_type_1d_view_type(wptr, Nspec);
    wptr += Nspec;
    auto Zp = real_type_1d_view_type(wptr, Nspec);
    wptr += Nspec;

    const ordinal_type workspace_used(wptr - work.data()),
      workspace_extent(work.extent(0));
    auto work_rhs = WorkViewType(wptr, workspace_extent - workspace_used);

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, Nspec),
                         [&](const ordinal_type& i) { Zp(i) = Zs(i); });

    Impl::SurfaceRHS::team_invoke(
      member, t, Ys, Zs, p, f, work_rhs, kmcd, kmcdSurf);

    member.team_barrier();

    // site fraction
    for (size_t j = 0; j < Nspec; j++) {

      const real_type perturb = (ats<real_type>::abs(reltol * Zs(j)) == 0
                                   ? abstol
                                   : ats<real_type>::abs(reltol * Zs(j)));

      Zp(j) = Zs(j) + perturb; // add perturb

      SurfaceRHS::team_invoke(
        member, t, Ys, Zp, p, fp, work_rhs, kmcd, kmcdSurf);

      member.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, Nspec),
                           [&](const ordinal_type& i) {
                             Jac(i, j) = (fp(i) - f(i)) / perturb + small;
                           });

      Zp(j) = Zs(j); // remove perturb
      member.team_barrier();
    }
  }
};

} // namespace Impl
} // namespace TChem

#endif
