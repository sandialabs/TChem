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


#ifndef __TCHEM_SpecificHeatCapacityPerMass_HPP__
#define __TCHEM_SpecificHeatCapacityPerMass_HPP__

#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct SpecificHeatCapacityPerMass
{
  static inline ordinal_type getWorkSpaceSize()
  {
    return 0; // does not need work size
  }

  static void runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view& state,
    /// output
    const real_type_2d_view& CpMass,
    const real_type_1d_view& CpMixMass,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd);

  //
  static void runHostBatch( /// thread block size
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    const real_type_2d_view_host& state,
    /// output
    const real_type_2d_view_host& CpMass,
    const real_type_1d_view_host& CpMixMass,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd);
};

} // namespace TChem

#endif
