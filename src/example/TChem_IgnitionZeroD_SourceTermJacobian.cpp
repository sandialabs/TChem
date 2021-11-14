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
#include "TChem_Impl_IgnitionZeroD_Problem.hpp" // here is where Ignition Zero D problem is implemented

int
main(int argc, char* argv[])
{
  ///
  /// 1. Command line input parser
  ///

  /// default input filename
  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");

  /// parse command line arguments --chemfile=user-chemfile.inp --thermfile=user-thermfile.dat
  /// with --help, the code list the available options.
  TChem::CommandLineParser opts("This example computes reaction rates with a given state vector");
  opts.set_option<std::string>("chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>("thermfile", "Therm file name e.g., therm.dat", &thermFile);

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
    using TChem::real_type;
    using TChem::ordinal_type;
    using value_type = Sacado::Fad::SLFad<real_type,100>;

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
    using problem_type = TChem::Impl::IgnitionZeroD_Problem<value_type, host_device_type >;

    /// state vector - Temperature, Y_0, Y_1, ... Y_{n-1}), n is # of species.
    const ordinal_type number_of_equations = problem_type::getNumberOfEquations(kmcd);

    ///
    /// TChem does not allocate any workspace internally.
    /// workspace should be explicitly given from users.
    /// you can create the work space using NVector, std::vector or real_type_1d_view_type (Kokkos view)
    /// here we use kokkos view
    const ordinal_type problem_workspace_size = problem_type::getWorkSpaceSize(kmcd);
    real_type_1d_view_type work("workspace", problem_workspace_size);

    ///
    /// 6. Input state vector
    ///

    /// set input vector
    const real_type pressure(101325);

    /// we use std vector mimicing users' interface
    std::vector<real_type> x_std(number_of_equations, real_type(0));
    x_std[0] = 1200; // temperature
    x_std[1] = 0.1;  // mass fraction species 1
    x_std[2] = 0.9;  // mass fraction species 2

    /// output rhs vector and J matrix
    std::vector<real_type> rhs_std(number_of_equations);
    std::vector<real_type> J_std(number_of_equations * number_of_equations);

    /// get points for std vectors
    real_type * ptr_x = x_std.data();
    real_type * ptr_rhs = rhs_std.data();
    real_type * ptr_J = J_std.data();

    ///
    /// 7. Compute right hand side vector and Jacobian
    ///
    {
      problem_type problem;

      /// initialize problem
      problem._p = pressure; // pressure
      problem._work = work;  // problem workspace array
      problem._kmcd = kmcd;  // kinetic model

      /// create view wrapping the pointers
      real_type_1d_view_type x(ptr_x,   number_of_equations);
      real_type_1d_view_type f(ptr_rhs, number_of_equations);
      real_type_2d_view_type J(ptr_J,   number_of_equations, number_of_equations);

      /// create a fake team member
      const auto member = Tines::HostSerialTeamMember();

      /// compute rhs and Jacobian
      problem.computeFunction(member, x, f);
      problem.computeJacobian(member, x, J);

      /// change the layout from row major to column major
      for (ordinal_type j=0;j<number_of_equations;++j)
	for (ordinal_type i=0;i<j;++i)
	  std::swap(J(i,j), J(j,i));
    }

    ///
    /// 8. Check the rhs and Jacobian
    ///

    printf("RHS std vector \n" );
    for (ordinal_type i=0;i<number_of_equations;++i) {
      printf("%e\n", rhs_std[i]);
    }

    printf("Jacobian std vector \n" );
    for (ordinal_type i=0;i<number_of_equations;++i) {
      for (ordinal_type j=0;j<number_of_equations;++j) {
        printf("%e ", J_std[i+j*number_of_equations] );
      }
      printf("\n" );
    }

    /// all Kokkos varialbes are reference counted objects. they are deallocated
    /// within this local scope.
  }

  /// Kokkos finalize checks any memory leak that are not properly deallocated.
  Kokkos::finalize();

  return 0;
}
