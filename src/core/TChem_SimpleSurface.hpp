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

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    return Impl::SimpleSurface::getWorkSpaceSize(kmcd, kmcdSurf);
          // +
           /// temporary tol view, later this tol also be given from users
           // 2 *
           //   (Impl::SimpleSurface_Problem<
           //      KineticModelConstDataType,
           //      KineticSurfModelConstDataType>::getNumberOfEquations(kmcdSurf) +
           //    1);
  }


  static void runDeviceBatch( /// input
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_1d_view& tol_newton,
    const real_type_2d_view& tol_time,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view& state,
    const real_type_2d_view& zSurf,
    /// output
    const real_type_1d_view& t_out,
    const real_type_1d_view& dt_out,
    const real_type_2d_view& Z_out,
    const real_type_2d_view& fac, // jac comput
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    const KineticSurfModelConstDataDevice& kmcdSurf);
};

} // namespace TChem

#endif
