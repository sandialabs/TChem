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

#ifndef __TCHEM_IMPL_SURFACE_HPP__
#define __TCHEM_IMPL_SURFACE_HPP__

#include "TChem_Impl_ReactionRatesSurface.hpp"
#include "TChem_Impl_RhoMixMs.hpp"
#include "TChem_Util.hpp"
namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct SurfaceRHS
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

  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {
    using ReactionRatesSurface = Impl::ReactionRatesSurface<real_type,device_type>;

    const ordinal_type per_team_extent =
      ReactionRatesSurface::getWorkSpaceSize(kmcd, kmcdSurf);

    return (per_team_extent + kmcd.nSpec + kmcdSurf.nSpec);
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type_1d_view_type& Ys, /// (kmcd.nSpec) mass fraction
    const real_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& p,           // pressure

    /// output
    const real_type_1d_view_type& dZs, /// (kmcdSurf.nSpec) // surface species
    const real_type_1d_view_type& omegaSurfGas,
    const real_type_1d_view_type& omegaSurf,
    const real_type_1d_view_type& work,
    /// workspace

    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {

    using ReactionRatesSurface = Impl::ReactionRatesSurface<real_type,device_type>;
    using RhoMixMs = RhoMixMs<real_type,device_type>;
    const real_type rhomix = RhoMixMs::team_invoke(member, t, p, Ys, kmcd);
    member.team_barrier();

    const real_type ten(10.0);
    /// compute surface production rates
    ReactionRatesSurface ::team_invoke(
      member, t, p, rhomix, Ys, Zs, omegaSurfGas, omegaSurf, work, kmcd, kmcdSurf);

    member.team_barrier();

    Kokkos::parallel_for
      (Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nSpec),
      [&](const ordinal_type &k) {
      dZs(k) = omegaSurf(k)/kmcdSurf.sitedensity / ten; // surface species equation
    });

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
           typename WorkViewType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type_1d_view_type& Ys, /// (kmcd.nSpec)
    const real_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& p,           // pressure
    /// output
    const real_type_1d_view_type& rhs, /// (kmcdSurf.nSpec )
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {

    auto w = (real_type*)work.data();

    auto omegaSurfGas = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto omegaSurf = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;

    auto work_surf = real_type_1d_view_type(w, work.extent(0) - (w - work.data()));
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


} // namespace Impl
} // namespace TChem

#endif
