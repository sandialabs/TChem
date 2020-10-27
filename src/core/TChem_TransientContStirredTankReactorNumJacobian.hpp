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
#ifndef __TCHEM_TRANSIENTCONTSTIRREDTANKREACTORNUMJAC_HPP__
#define __TCHEM_TRANSIENTCONTSTIRREDTANKREACTORNUMJAC_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
#include "TChem_TransientContStirredTankReactor.hpp"
#include "TChem_Impl_TransientContStirredTankReactorNumJacobian.hpp"

namespace TChem {


struct TransientContStirredTankReactorNumJacobian
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename TransientContStirredTankReactorConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const TransientContStirredTankReactorConstDataType& cstr)
  {
    return Impl::TransientContStirredTankReactorNumJacobian
               ::getWorkSpaceSize(kmcd, kmcdSurf, cstr);
  }

  static void runDeviceBatch( /// thread block size
    const ordinal_type nBatch,
    const real_type_2d_view& state,
    const real_type_2d_view& zSurf,
    /// output
    const real_type_3d_view& jac,
    const real_type_2d_view& fac,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const cstr_data_type& cstr);

};

} // namespace TChem

#endif
