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
#ifndef __TCHEM_SMATRIX_HPP__
#define __TCHEM_SMATRIX_HPP__

#include "TChem_Impl_Smatrix.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct Smatrix
{

  template<typename KineticModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    return Impl::Smatrix::getWorkSpaceSize(kmcd);
  }

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view& state,
    /// output
    const real_type_3d_view& Smatrix,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd);
  //
  static void runDeviceBatch( /// input
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view& state,
    /// output
    const real_type_3d_view& Smatrix,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd);
  //
  static void runHostBatch( /// input
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    const real_type_2d_view_host& state,
    /// output
    const real_type_3d_view_host& Smatrix,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd);

  //
  static void runHostBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// output
    const real_type_3d_view_host& Smatrix,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd);

};

} // namespace TChem

#endif
