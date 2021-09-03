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
#ifndef __TCHEM_SIMPLESURFACE_HPP__
#define __TCHEM_SIMPLESURFACE_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

#include "TChem_Impl_SimpleSurface.hpp"

namespace TChem {

struct SimpleSurface
{

  using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
  using device_type      = typename Tines::UseThisDevice<exec_space>::type;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
  using kinetic_model_type = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;

  template< typename DeviceType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstData<DeviceType>& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf)
  {
    return Impl::SimpleSurface<real_type,DeviceType>::getWorkSpaceSize(kmcd, kmcdSurf);
  }


  static void runDeviceBatch( /// input
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_1d_view_type& tol_newton,
    const real_type_2d_view_type& tol_time,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_1d_view_type& t_out,
    const real_type_1d_view_type& dt_out,
    const real_type_2d_view_type& site_fraction_out,
    const real_type_2d_view_type& fac, // jac comput
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf);

  //
  static void runDeviceBatch( /// input
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_1d_view_type& tol_newton,
    const real_type_2d_view_type& tol_time,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_1d_view_type& t_out,
    const real_type_1d_view_type& dt_out,
    const real_type_2d_view_type& site_fraction_out,
    const real_type_2d_view_type& fac, // jac comput
    /// const data from kinetic model
    const Kokkos::View<kinetic_model_type*,device_type>& kmcds,
    const Kokkos::View<kinetic_surf_model_type*,device_type>& kmcdSurfs);
};

} // namespace TChem

#endif
