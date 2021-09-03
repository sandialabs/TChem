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
#ifndef __TCHEM_ENTROPYMASS_HPP__
#define __TCHEM_ENTROPYMASS_HPP__

#include "TChem_Impl_Entropy0SpecMl.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct EntropyMass
{
  using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
  using device_type      = typename Tines::UseThisDevice<exec_space>::type;

  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  using real_type_1d_view_host_type = Tines::value_type_1d_view<real_type,host_device_type>;
  using real_type_2d_view_host_type = Tines::value_type_2d_view<real_type,host_device_type>;

  using kinetic_model_type = KineticModelConstData<device_type>;
  using kinetic_model_host_type = KineticModelConstData<host_device_type>;

  template<typename DeviceType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstData<DeviceType>& kmcd)
  {
    return kmcd.nSpec; // does not need work size
  }

  static void runDeviceBatch( /// thread block size
    const exec_space& exec_space_instance,
    const ordinal_type& team_size,
    const ordinal_type& vector_size,
    /// input
    const ordinal_type& nBatch,
    const real_type_2d_view_type& state,
    /// output
    const real_type_2d_view_type& EntropyMass,
    const real_type_1d_view_type& EntropyMixMass,

    /// const data from kinetic model
    const kinetic_model_type& kmcd);
  //
  static void runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view_type& state,
    /// output
    const real_type_2d_view_type& EntropyMass,
    const real_type_1d_view_type& EntropyMixMass,

    /// const data from kinetic model
    const kinetic_model_type& kmcd);
  //
  static void runHostBatch( /// thread block size
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    const real_type_2d_view_host_type& state,
    /// output
    const real_type_2d_view_host_type& EntropyMass,
    const real_type_1d_view_host_type& EntropyMixMass,
    /// const data from kinetic model
    const kinetic_model_host_type& kmcd);
};



} // namespace TChem

#endif
