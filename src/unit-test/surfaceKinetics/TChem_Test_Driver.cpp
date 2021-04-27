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
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

#include "TChem_KineticModelData.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;

TChem::KineticModelData kmd;
std::string _prefixPath;

#include "TChem_Test_KineticModel.hpp"
#include "TChem_Test_ReactionRates.hpp"
#include "TChem_Test_Thermo.hpp"
#include "TChem_NetProductionRateSurfacePerMass.hpp"

int main (int argc, char *argv[]) {
  int r_val(0);
  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::     exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    _prefixPath = "input/PT/";
    std::string chemFile(_prefixPath + "chem.inp");
    std::string thermFile(_prefixPath + "therm.dat");
    std::string chemSurfFile(_prefixPath + "chemSurf.inp");
    std::string thermSurfFile(_prefixPath + "thermSurf.dat");

    /// construct kmd and use the view for testing
    kmd = TChem::KineticModelData(chemFile, thermFile,
                                  chemSurfFile, thermSurfFile);

    ::testing::InitGoogleTest(&argc, argv);
    r_val = RUN_ALL_TESTS();

    /// deallocate internal views
    kmd = TChem::KineticModelData();
  }
  Kokkos::finalize();

  return r_val;
}
