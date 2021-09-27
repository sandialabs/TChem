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
#ifndef __TCHEM_TEST_SURF_KINETICMODEL_HPP__
#define __TCHEM_TEST_SURF_KINETICMODEL_HPP__

#include "TChem_KineticModelData.hpp"

TEST( KineticModelData, constructor ) {
  // gas phase
  /// check echo file

  EXPECT_TRUE(TChem::Test::compareFiles("kmod.echo",_prefixPath+"kmod.ref.echo"));
  EXPECT_TRUE(TChem::Test::compareFiles("kmod.list",_prefixPath+"kmod.ref.list"));
  EXPECT_TRUE(TChem::Test::compareFiles("kmod.reactions",_prefixPath+"kmod.ref.reactions"));

  //surface phase

  EXPECT_TRUE(TChem::Test::compareFiles("kmodSurf.echo",_prefixPath+"kmodSurf.ref.echo"));
  EXPECT_TRUE(TChem::Test::compareFiles("kmodSurf.list",_prefixPath+"kmodSurf.ref.list"));
  EXPECT_TRUE(TChem::Test::compareFiles("kmodSurf.reactions",_prefixPath+"kmodSurf.ref.reactions"));

  using device_type = typename Tines::UseThisDevice<TChem::exec_space>::type;
  using host_device_type = typename Tines::UseThisDevice<TChem::host_exec_space>::type;

  /// check const data operations on gas phase
  const auto kmcd_device = TChem::createGasKineticModelConstData<device_type>(kmd);
  const auto kmcd_host = TChem::createGasKineticModelConstData<host_device_type>(kmd);

  EXPECT_EQ(kmcd_device.nSpec, kmcd_host.nSpec);

  /// check const data operations on surface phase
  const auto kmcdSurf_device =
    TChem::createSurfaceKineticModelConstData<device_type>(kmd);
  const auto kmcdSurf_host =
    TChem::createSurfaceKineticModelConstData<host_device_type>(kmd);

  EXPECT_EQ(kmcdSurf_device.nSpec, kmcdSurf_host.nSpec);

}

#endif
