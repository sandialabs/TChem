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
#ifndef __TCHEM_IMPL_SURFACE_COVERAGE_MODIFICATION__
#define __TCHEM_IMPL_SURFACE_COVERAGE_MODIFICATION__

#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct SurfaceCoverageModification
{

  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;
  using real_type = scalar_type;

  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;

  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
  const MemberType& member,
  /// input
  const value_type& t,
  const value_type_1d_view_type& concX,
  const value_type_1d_view_type& concXSurf,
  //outputs
  const value_type_1d_view_type& CoverageFactor,
  const kinetic_surf_model_type& kmcdSurf)
  {
    const real_type ten(10), one(1);

    const ordinal_type NumofConvFac = kmcdSurf.coverageFactor.extent(0);
    // printf("Numer of coverage Factors %d \n", NumofConvFac);
    coverage_modification_type cov;

    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nReac),
      [&](const ordinal_type& i) {
       CoverageFactor(i) = one;
     });

    member.team_barrier();

    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, NumofConvFac),
      [&](const ordinal_type& i) {

          cov = kmcdSurf.coverageFactor(i);
          value_type act;
          if (cov._isgas){
            act = concX(cov._species_index); // molar concentration
            // printf("I am a gas species \n" );
          } else{
            act = concXSurf(cov._species_index)/kmcdSurf.sitedensity; // site fraction
            // printf("I am a surface species \n" );
          }
          // printf("eta %e, mu %e, epsilon %e reac indx %d spec indx %d \n",
          // cov._eta, cov._mu, cov._epsilon, cov._reaction_index, cov._species_index  );

          const value_type prod = ats<value_type>::pow(ten, cov._eta * act)*
                                 ats<value_type>::pow(act, cov._mu) *
                                 ats<value_type>::exp(-cov._epsilon * act/t);

        CoverageFactor(cov._reaction_index) *= prod;

      });

  }
};

} // namespace Impl
} // namespace TChem

#endif
