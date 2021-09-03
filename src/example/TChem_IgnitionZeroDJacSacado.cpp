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
#include "TChem_KineticModelData.hpp"
#include "TChem_SourceTerm.hpp"
#include "TChem_Util.hpp"
#include "Sacado.hpp"
#include "TChem_Impl_CpMixMs.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("data/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "sample.dat");
  // std::string outputFile(prefixPath + "CpMixMass.dat");
  bool verbose(true);

  int nBatch(1), team_size(-1), vector_size(-1);
  ;

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<std::string>(
    "chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>(
    "thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>(
    "inputfile", "Input state file name e.g., input.dat", &inputFile);
  // opts.set_option<std::string>("outputfile", "Output omega file name e.g.,
  // omega.dat", &outputFile);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first omega values", &verbose);
  opts.set_option<int>("vector-size", "User defined vector size", &vector_size);
  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    const auto exec_space_instance = TChem::host_exec_space();

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);
    using host_device_type      = typename Tines::UseThisDevice<TChem::host_exec_space>::type;

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const auto kmcd = kmd.createConstData<host_device_type>();

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);
    real_type_2d_view_host state_host;

    {
      // get species molecular weigths
      TChem::Test::readSample(inputFile,
                              kmcd.speciesNames,
                              kmcd.sMass,
                              kmcd.nSpec,
                              stateVecDim,
                              state_host,
                              nBatch);
    }

    // output:: Source
    real_type_2d_view_host sourceTerm("source term ignition zero D", nBatch, stateVecDim);

    // compute heat capacity in mass base
    Kokkos::Impl::Timer timer;

    timer.reset();

    const real_type t_deepcopy = timer.seconds();

    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::host_exec_space>::type;

    const ordinal_type level = 1;

    timer.reset();

    {

      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      const ordinal_type per_team_extent =
        TChem::SourceTerm::getWorkSpaceSize(kmcd);
      ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view_host>::shmem_size(per_team_extent);
      policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

      TChem::SourceTerm::runHostBatch(policy,
                                      state_host,
                                      sourceTerm,
                                      kmcd);
    }


    Kokkos::fence(); /// timing purpose
    const real_type t_device_batch = timer.seconds();

    /// show time
    printf("Batch size %d, chemfile %s, thermfile %s, statefile %s\n",
           nBatch,
           chemFile.c_str(),
           thermFile.c_str(),
           inputFile.c_str());
    printf("---------------------------------------------------\n");
    printf("Time deep copy      %e [sec] %e [sec/sample]\n",
           t_deepcopy,
           t_deepcopy / real_type(nBatch));
    printf("Time reaction rates %e [sec] %e [sec/sample]\n",
           t_device_batch,
           t_device_batch / real_type(nBatch));



    constexpr int FadDimUpperBound = 100;
    using FadType = Sacado::Fad::SLFad<double,FadDimUpperBound>;
    using fad_type_1d_view =
      Kokkos::View<FadType*, Kokkos::LayoutRight, Kokkos::HostSpace>;

    //
    const auto state_at_i = Kokkos::subview(state_host, 0, Kokkos::ALL());
    const Impl::StateVector<real_type_1d_view_host> sv_at_i(kmcd.nSpec,
                                                        state_at_i);
    //
    const auto t = sv_at_i.Temperature();
    const auto p = sv_at_i.Pressure();
    const auto density = sv_at_i.Density();
    const auto Ys = sv_at_i.MassFractions();

    // thermal propierties only depend on temperature
    {

      int m = 1;//kmcd.nSpec +
      int n = kmcd.nSpec;

      fad_type_1d_view x ("x", m, m+1);
      fad_type_1d_view f ("f", n, n+1);
      fad_type_1d_view f2 ("f2", n, n+1);
      fad_type_1d_view f3 ("f3", n, n+1);

      real_type_2d_view_host df("df", m, n);
      real_type_2d_view_host df2("df2", m, n);
      real_type_2d_view_host df3("df3", m, n);

      x(0) = FadType(m, 0, t);

      const FadType t2 = x(0);

      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      Kokkos::parallel_for(
        "Test jacobian ",
        policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
          const ordinal_type i = member.league_rank();

         // auto cp_mix = TChem::Impl::CpMixMs::team_invoke(member, t2, Ys2, f, kmcd);
         // TChem::Impl::CpSpecMl::team_invoke(member, t2, f, kmcd);
         // TChem::Impl::EnthalpySpecMl::team_invoke(member, t2, f, f2, kmcd);
         // TChem::Impl::CpSpecMs::team_invoke(member, t2, f, kmcd);
         // TChem::Impl::EnthalpySpecMs::team_invoke(member, t2, f, f2, kmcd);
         // TChem::Impl::Entropy0SpecMl::team_invoke(member, t2, f, f2, kmcd);
         TChem::Impl::Gk::team_invoke(member, t2, f, f2, f3, kmcd);

          });


          for (int i=0;i<m;++i)
            for (int j=0;j<n;++j)
  	          df(i,j) = f(i).fastAccessDx(j);
          //
          printf("J = \n");
          for (int i=0;i<m;++i) {
            for (int j=0;j<n;++j)
  	           printf(" %e ",  df(i,j));
          }
          printf("\n");

          for (int i=0;i<m;++i)
            for (int j=0;j<n;++j)
  	          df2(i,j) = f2(i).fastAccessDx(j);
          //
          printf("J = \n");
          for (int i=0;i<m;++i) {
            for (int j=0;j<n;++j)
  	           printf(" %e ",  df2(i,j));
          }
          printf("\n");


    }

    // function with depends on mass fraction and temperature
    {
      int m = 2;// temperature and pressure
      int n = kmcd.nSpec;

      fad_type_1d_view x ("x", m, m+1);

      fad_type_1d_view gk("gk", n, n+1);
      fad_type_1d_view hks("hks", n, n+1);
      fad_type_1d_view workGk("workGk", n, n+1);

      fad_type_1d_view kfor("kfor", n, n+1 );
      fad_type_1d_view krev("krev", n, n+1 );

      ordinal_type_1d_view_host work("work", 2*kmcd.nReac );

      real_type_2d_view_host dgk("df", m, n);
      real_type_2d_view_host dkfor("kfor", m, n);

      x(0) = FadType(m, 0, t);
      x(1) = FadType(m, 1, p);

      //
      const FadType t2 = x(0);
      const FadType p2 = x(1);

      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      Kokkos::parallel_for(
        "Test jacobian ",
        policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
          const ordinal_type i = member.league_rank();

          TChem::Impl::Gk::team_invoke(member, t2, gk, hks, workGk, kmcd );

          member.team_barrier();

          TChem::Impl::KForwardReverse::
          team_invoke(member, t2, p2, gk, kfor, krev, work, kmcd );

          });

      //
      for (int i=0;i<m;++i)
        for (int j=0;j<n;++j)
          dgk(i,j) = gk(i).fastAccessDx(j);
      //
      printf("J (gk) = \n");
      for (int i=0;i<m;++i) {
        for (int j=0;j<n;++j)
           printf(" %e ",  dgk(i,j));
      }
      printf("\n");

      for (int i=0;i<m;++i)
        for (int j=0;j<n;++j)
          dkfor(i,j) = kfor(i).fastAccessDx(j);
      //
      printf("J (kfor) = \n");
      for (int i=0;i<m;++i) {
        printf("i %d:",i);
        for (int j=0;j<n;++j)
           printf(" %e ",  dkfor(i,j));
      }
      printf("\n");
    }

  }
  Kokkos::finalize();

  return 0;
}
