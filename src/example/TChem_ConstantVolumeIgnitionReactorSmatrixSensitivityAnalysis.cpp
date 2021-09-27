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
#include "TChem_CommandLineParser.hpp"

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Impl_ConstantVolumeIgnitionReactorSmatrixSensitivityAnalysis.hpp"

int
main(int argc, char* argv[])
{
  ///
  /// 1. Command line input parser
  ///

  /// default input filename
  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string inputFile("sample.dat");

  /// parse command line arguments --chemfile=user-chemfile.inp --thermfile=user-thermfile.dat
  /// with --help, the code list the available options.
  TChem::CommandLineParser opts("This example computes reaction rates with a given state vector");
  opts.set_option<std::string>("chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>("thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>
  ("samplefile", "Input state file name e.g.,input.dat", &inputFile);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  ///
  /// 2. Kokkos initialization
  ///

  /// Kokkos initialization requires local scoping to properly deallocate global variables.
  Kokkos::initialize(argc, argv);
  {
    ///
    /// 3. Type definitions
    ///

    /// scalar type and ordinal type
    using real_type = double;
    using ordinal_type = int;

    /// Kokkos environments - host device type and multi dimensional arrays
    /// note that the 2d view use row major layout while most matrix format uses column major layout.
    /// to make the 2d view compatible with other codes, we need to transpose it.
    using host_device_type = typename Tines::UseThisDevice<TChem::host_exec_space>::type;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type,host_device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type,host_device_type>;

    ///
    /// 4. Construction of a kinetic model
    ///

    /// construct TChem's kinect model and read reaction mechanism
    TChem::KineticModelData kmd(chemFile, thermFile);

    /// construct const (read-only) data and move the data to device.
    /// for host device, it is soft copy (pointer assignmnet).
    auto kmcd = TChem::createGasKineticModelConstData<host_device_type>(kmd);

    ///
    /// 5. Problem setup
    ///

    /// use problem object for ignition zero D problem providing interface to source term and jacobian
    /// other available problem objects in TChem: PFR, CSTR, and CV ignition
    using Smatrix_type = TChem::Impl::ConstantVolumeIgnitionReactorSmatrixSensitivityAnalysis<real_type, host_device_type>;

    ///
    /// TChem does not allocate any workspace internally.
    /// workspace should be explicitly given from users.
    /// you can create the work space using NVector, std::vector or real_type_1d_view_type (Kokkos view)
    /// here we use kokkos view
    const ordinal_type problem_workspace_size = Smatrix_type::getWorkSpaceSize(kmcd);
    real_type_1d_view_type work("workspace", problem_workspace_size);
    ordinal_type nBatch(0);

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);


    real_type_2d_view_type state_host;
    const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
    Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);
    {
      // get species molecular weigths
      const auto SpeciesMolecularWeights =
        Kokkos::create_mirror_view(kmcd.sMass);
      Kokkos::deep_copy(SpeciesMolecularWeights, kmcd.sMass);

      TChem::Test::readSample(inputFile,
                              speciesNamesHost,
                              SpeciesMolecularWeights,
                              kmcd.nSpec,
                              stateVecDim,
                              state_host,
                              nBatch);
    }


    /// create a fake team member
    const auto member = Tines::HostSerialTeamMember();


    const real_type_1d_view_type state_at_i = Kokkos::subview(state_host, 0, Kokkos::ALL());
    Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i);

    const auto t = sv_at_i.Temperature();
    const auto p = sv_at_i.Pressure();
    const auto Ys = sv_at_i.MassFractions();
    const auto density = sv_at_i.Density();


    real_type_2d_view_type Smat_rev("Smat_rev",kmcd.nSpec + 1, kmcd.nReac);
    real_type_2d_view_type Smat_fwd("Smat_fwd",kmcd.nSpec + 1, kmcd.nReac);
    real_type_1d_view_type ak("Smat_fwd",kmcd.nSpec);

    Smatrix_type::team_invoke(member,
                              /// input
                              t, p, density, Ys, /// (kmcd.nSpec)
                              /// output
                              Smat_rev, /// (1)
                              Smat_fwd, /// (1)
                              ak, /// (1)
                              /// workspace
                              work,
    /// const input from kinetic model
                              kmcd);

    ///
    Tines::showMatrix("Smat_rev", Smat_rev);
    Tines::showMatrix("Smat_fwd", Smat_fwd);

    printf(" ak  \n" );
    for (ordinal_type i = 0; i < ak.extent(0); i++) {
      printf("i %d ak %e\n",i, ak(i) );
    }


    using Smatrix_source_term_type = TChem::Impl::ConstantVolumeIgnitionReactorTLASourceTerm<real_type, host_device_type>;

    const ordinal_type work2_len = Smatrix_source_term_type::getWorkSpaceSize(kmcd);
    const ordinal_type ns(kmcd.nSpec + 1);
    const real_type alpha(1000);

    real_type_1d_view_type fac("fac", ns );

    real_type_1d_view_type Zn("Zn",ns * ns);
    real_type_1d_view_type source("source",ns * ns);
    real_type_1d_view_type work2("workTLA", work2_len );
    Smatrix_source_term_type::team_invoke(
                                       member,
                                       Zn, /// column-major order
      /// outputs
                                       source, // column-major order
                                       t,
                                       p,
                                       density,
                                       Ys, /// (kmcd.nSpec)
                                       alpha,
                                       fac,
                                       work2,
                                       kmcd);

  //
  printf(" source  \n" );
  for (ordinal_type i = 0; i < ns; i++) {
    for (size_t k = 0; k < ns; k++) {
      printf(" %e ", source(k*ns + i ) );
    }
    printf("\n" );
  }

#if 0
  printf("stoiCoefMatrix\n" );

  for (ordinal_type i = 0; i < kmcd.nReac; i++) {
    printf("reaction No %d \n", i );
    for (ordinal_type k = 0; k < kmcd.nSpec; k++) {
      if ( std::abs(kmcd.stoiCoefMatrix(k,i)) > 0  ) {
        printf("%s : %d ", &kmcd.speciesNames(k, 0), ordinal_type(kmcd.stoiCoefMatrix(k,i)) );
      }
    }
    printf("\n");
  }
#endif






    /// all Kokkos varialbes are reference counted objects. they are deallocated
    /// within this local scope.
  }

  /// Kokkos finalize checks any memory leak that are not properly deallocated.
  Kokkos::finalize();

  return 0;
}
