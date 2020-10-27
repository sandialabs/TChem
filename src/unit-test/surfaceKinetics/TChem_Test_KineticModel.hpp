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
  std::ifstream f1("kmod.echo"), f2(_prefixPath + "kmod.ref.echo");
  std::ifstream f3("kmod.list"), f4(_prefixPath + "kmod.ref.list");

  std::string s1((std::istreambuf_iterator<char>(f1)), std::istreambuf_iterator<char>());
  std::string s2((std::istreambuf_iterator<char>(f2)), std::istreambuf_iterator<char>());
  std::string s3((std::istreambuf_iterator<char>(f3)), std::istreambuf_iterator<char>());
  std::string s4((std::istreambuf_iterator<char>(f4)), std::istreambuf_iterator<char>());

  EXPECT_TRUE(s3.compare(s4) ==0);
  EXPECT_TRUE(s1.compare(s2) ==0);
  //surface phase

  std::ifstream f5("kmodSurf.echo"), f6(_prefixPath + "kmodSurf.ref.echo");
  std::ifstream f7("kmodSurf.list"), f8(_prefixPath + "kmodSurf.ref.list");

  std::string s5((std::istreambuf_iterator<char>(f5)), std::istreambuf_iterator<char>());
  std::string s6((std::istreambuf_iterator<char>(f6)), std::istreambuf_iterator<char>());
  std::string s7((std::istreambuf_iterator<char>(f7)), std::istreambuf_iterator<char>());
  std::string s8((std::istreambuf_iterator<char>(f8)), std::istreambuf_iterator<char>());

  EXPECT_TRUE(s5.compare(s6) ==0);
  EXPECT_TRUE(s7.compare(s8) ==0);

  /// check const data operations on gas phase
  const auto kmcd_device = kmd.createConstData<TChem::     exec_space>();
  const auto kmcd_host   = kmd.createConstData<TChem::host_exec_space>();

  EXPECT_EQ(kmcd_device.nSpec, kmcd_host.nSpec);

  /// check const data operations on surface phase
  const auto kmcdSurf_device = kmd.createConstSurfData<TChem::     exec_space>();
  const auto kmcdSurf_host   = kmd.createConstSurfData<TChem::host_exec_space>();

  EXPECT_EQ(kmcdSurf_device.nSpec, kmcdSurf_host.nSpec);

}

#endif
