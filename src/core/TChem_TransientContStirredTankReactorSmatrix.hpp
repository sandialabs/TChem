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
#ifndef __TCHEM_TRANSIENTCONTSTIRREDTANKREACTORSMATRIX_HPP__
#define __TCHEM_TRANSIENTCONTSTIRREDTANKREACTORSMATRIX_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
#include "TChem_TransientContStirredTankReactor.hpp"
#include "TChem_Impl_TransientContStirredTankReactorSmatrix.hpp"

namespace TChem {


struct TransientContStirredTankReactorSmatrix
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    return 2 * kmcd.nSpec + kmcdSurf.nReac;
  }
  //policy is define inside the interface
  static void runDeviceBatch( /// thread block size
    const ordinal_type team_size,
    const ordinal_type vector_size,
    const ordinal_type nBatch,
    const real_type_2d_view& state,
    const real_type_2d_view& zSurf,
    /// output
    const real_type_3d_view& Smat,
    const real_type_3d_view& Ssmat,
    const real_type_2d_view& Sconv,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const cstr_data_type& cstr);

  // policy is define outside of the interface
  static void runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view& state,
    const real_type_2d_view& zSurf,
    /// output
    const real_type_3d_view& Smat,
    const real_type_3d_view& Ssmat,
    const real_type_2d_view& Sconv,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const cstr_data_type& cstr);

  //
  static void runHostBatch( /// thread block size
    const ordinal_type team_size,
    const ordinal_type vector_size,
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    const real_type_2d_view_host& zSurf,
    /// output
    const real_type_3d_view_host& Smat,
    const real_type_3d_view_host& Ssmat,
    const real_type_2d_view_host& Sconv,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const cstr_data_type& cstr);
  //
  static void runHostBatch( /// thread block size
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    const real_type_2d_view_host& state,
    const real_type_2d_view_host& zSurf,
    /// output
    const real_type_3d_view_host& Smat,
    const real_type_3d_view_host& Ssmat,
    const real_type_2d_view_host& Sconv,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const cstr_data_type& cstr);

};

} // namespace TChem

#endif
