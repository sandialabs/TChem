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


#ifndef __TCHEM_JACOBIAN_HPP__
#define __TCHEM_JACOBIAN_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

///      _                 _     _
///     | | __ _  ___ ___ | |__ (_) __ _ _ __  ___
///  _  | |/ _` |/ __/ _ \| '_ \| |/ _` | '_ \/ __|
/// | |_| | (_| | (_| (_) | |_) | | (_| | | | \__ \
  ///  \___/ \__,_|\___\___/|_.__/|_|\__,_|_| |_|___/
///
struct Jacobian
{
  template<typename KineticModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    const ordinal_type iter_size =
      (kmcd.nSpec > kmcd.nReac ? kmcd.nSpec : kmcd.nReac) * 2;
    return (6 * kmcd.nSpec + 10 * kmcd.nReac + 4 + iter_size);
  }

  static void runHostBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// output
    const real_type_3d_view_host& jac,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd);

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view& state,
    /// output
    const real_type_3d_view& jac,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd);
};

} // namespace TChem

#endif
