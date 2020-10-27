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
#ifndef __TCHEM_PlugFlowReactorRHS_HPP__
#define __TCHEM_PlugFlowReactorRHS_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

template<typename SpT>
struct PlugFlowReactorConstData
{
public:
  enum
  {
    is_device = std::is_same<SpT, exec_space>::value,
    is_host = std::is_same<SpT, host_exec_space>::value
  };
  // static_assert(!is_device && !is_host, "SpT is wrong" );
  using kmcd_ordinal_type_1d_view =
    ConstUnmanaged<typename std::conditional<is_device,
                                             ordinal_type_1d_view,
                                             ordinal_type_1d_view_host>::type>;
  using kmcd_ordinal_type_2d_view =
    ConstUnmanaged<typename std::conditional<is_device,
                                             ordinal_type_2d_view,
                                             ordinal_type_2d_view_host>::type>;

  using kmcd_real_type_1d_view =
    ConstUnmanaged<typename std::conditional<is_device,
                                             real_type_1d_view,
                                             real_type_1d_view_host>::type>;
  using kmcd_real_type_2d_view =
    ConstUnmanaged<typename std::conditional<is_device,
                                             real_type_2d_view,
                                             real_type_2d_view_host>::type>;
  using kmcd_real_type_3d_view =
    ConstUnmanaged<typename std::conditional<is_device,
                                             real_type_3d_view,
                                             real_type_3d_view_host>::type>;

  real_type Area;
  real_type Pcat;
};

using PlugFlowReactorConstDataHost = PlugFlowReactorConstData<host_exec_space>;
using PlugFlowReactorConstDataDevice = PlugFlowReactorConstData<exec_space>;

struct PlugFlowReactorRHS
{

  template<typename SpT>
  static inline PlugFlowReactorConstData<SpT> createConstData()
  {
    PlugFlowReactorConstData<SpT> data;
    data.Area = 0.00053;              // m2
    data.Pcat = 0.025977239243415308; //
    return data;
  }

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    return (7 * kmcd.nSpec + 8 * kmcd.nReac + 5 * kmcdSurf.nSpec +
            6 * kmcdSurf.nReac);
  }

  static void runHostBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// input
    const real_type_2d_view_host& zSurf,
    // prf aditional variable
    const real_type_1d_view_host& velocity,
    /// output
    const real_type_2d_view_host& rhs,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataHost& kmcdSurf);

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    /// input gas state
    const real_type_2d_view& state,
    /// surface state
    const real_type_2d_view& zSurf,
    // prf aditional variable
    const real_type_1d_view& velocity,
    /// output
    const real_type_2d_view& rhs,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataDevice& kmcdSurf);
};

} // namespace TChem

#endif
