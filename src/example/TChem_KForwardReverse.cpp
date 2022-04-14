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
#include "TChem_Impl_Gk.hpp"
#include "TChem_Impl_KForwardReverse.hpp"
#include "TChem_Util.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;

using real_type_0d_view = TChem::real_type_0d_view;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;


// #define TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT

int
main(int argc, char* argv[])
{
  /// default inputs
  std::string prefixPath("data/ignition-zero-d/CO/");
  int nBatch(1), team_size(-1), vector_size(-1);
  bool verbose(true);

  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string inputFile("sample.dat");
  std::string inputFileParamModifiers("ParameterModifiers.dat");

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes the solution of an ignition problem");
  opts.set_option<std::string>(
    "inputsPath", "path to input files e.g., data/inputs", &prefixPath);
  //
  opts.set_option<std::string>
  ("chemfile", "Chem file name e.g., chem.inp",&chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file namee.g., therm.dat", &thermFile);
  opts.set_option<std::string>
  ("samplefile", "Input state file name e.g.,input.dat", &inputFile);
  opts.set_option<std::string>
  ("inputFileParamModifiers", "Input state file name e.g.,input.dat", &inputFileParamModifiers);

  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);
  //
  opts.set_option<int>("team-size", "User defined team size", &team_size);
  opts.set_option<int>("vector-size", "User defined vector size", &vector_size);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return


  Kokkos::initialize(argc, argv);
  {

    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);
    using device_type      = typename Tines::UseThisDevice<exec_space>::type;

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const TChem::KineticModelConstData<device_type> kmcd =
      TChem::createGasKineticModelConstData<device_type>(kmd);


    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    printf("Number of Species %d \n", kmcd.nSpec);
    printf("Number of Reactions %d \n", kmcd.nReac);

    /// input from a file; this is not necessary as the input is created
    /// by other applications.
    // real_type_2d_view state;
    real_type_2d_view_host state_host;
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

    real_type_2d_view state("StateVector Devices", nBatch, stateVecDim);
    Kokkos::deep_copy(state, state_host);


    real_type_2d_view kFor("kfor gas ", nBatch, kmcd.nReac);
    real_type_2d_view kRev("krev gas" , nBatch, kmcd.nReac);

    /// different sample would have a different kinetic model
    auto kmds = kmd.clone(nBatch);
    constexpr ordinal_type gas(0);
    TChem::modifyArrheniusForwardParameter(kmds, gas, inputFileParamModifiers);
    auto kmcds = TChem::createGasKineticModelConstData<device_type>(kmds);

    if (verbose) {
      auto reacArhenFor_host_orginal = Kokkos::create_mirror_view(kmcd.reacArhenFor);
      Kokkos::deep_copy(reacArhenFor_host_orginal, kmcd.reacArhenFor);

      ordinal_type ireac(2);
      printf("-- Testing Modification --\n");

      ordinal_type NprintSample(3);
      if  (nBatch < NprintSample) NprintSample=nBatch;

      for (ordinal_type isample = 0; isample < NprintSample; isample++) {
        auto kmcd_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
      						       Kokkos::subview(kmcds, isample));
        printf("sample No %d Reaction Index %d \n", isample, ireac );
        printf("Before Modification : A %e B %f E/R %f \n", reacArhenFor_host_orginal(ireac,0),
          reacArhenFor_host_orginal(ireac,1), reacArhenFor_host_orginal(ireac,2) );
        printf("After  Modification : A %e B %f E/R %f \n", kmcd_host().reacArhenFor(ireac,0),
          kmcd_host().reacArhenFor(ireac,1), kmcd_host().reacArhenFor(ireac,2) );
        printf("       Modification : A %f B %f E/R %f \n", kmcd_host().reacArhenFor(ireac,0)/reacArhenFor_host_orginal(ireac,0),
          kmcd_host().reacArhenFor(ireac,1)/reacArhenFor_host_orginal(ireac,1),kmcd_host().reacArhenFor(ireac,2)/reacArhenFor_host_orginal(ireac,2));

      }

    }

    const auto exec_space_instance = TChem::exec_space();

    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    /// team policy
    policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());
    const ordinal_type level = 1;

    if (team_size > 0 && vector_size > 0) {
      policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
    }
    const ordinal_type work_kfor_rev_size =
    TChem::Impl::KForwardReverse<real_type,device_type>::getWorkSpaceSize(kmcd);
    const ordinal_type per_team_extent = 3 * kmcd.nSpec + 2 * kmcd.nReac +  work_kfor_rev_size;

    const ordinal_type per_team_scratch =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    Kokkos::parallel_for(
      "TChem::KForwardReverse::example model variation",
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));

        const real_type_1d_view state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        const real_type_1d_view kfor_at_i =
          Kokkos::subview(kFor, i, Kokkos::ALL());
        const real_type_1d_view krev_at_i =
          Kokkos::subview(kRev, i, Kokkos::ALL());

        Scratch<real_type_1d_view> work(member.team_scratch(level),
                                        per_team_extent);

        const Impl::StateVector<real_type_1d_view> sv_at_i(kmcd_at_i.nSpec,
                                                           state_at_i);
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");
        {
          const real_type t = sv_at_i.Temperature();
          const real_type p = sv_at_i.Pressure();
          const real_type density = sv_at_i.Density();
          const real_type_1d_view Ys = sv_at_i.MassFractions();

          printf("temp %e pressure %e density %e\n", t, p, density );


          auto w = (real_type*)work.data();

          auto gk = real_type_1d_view(w, kmcd_at_i.nSpec);
          w += kmcd_at_i.nSpec;
          auto hks = real_type_1d_view(w, kmcd_at_i.nSpec);
          w += kmcd_at_i.nSpec;
          auto cpks = real_type_1d_view(w, kmcd_at_i.nSpec);
          w += kmcd_at_i.nSpec;
          auto work_kfor_rev = real_type_1d_view(w, work_kfor_rev_size);
          w += work_kfor_rev_size;
          auto iter_real_type = real_type_1d_view(w, kmcd_at_i.nReac * 2);
          w += kmcd_at_i.nReac * 2;
          auto ww =  (ordinal_type*)iter_real_type.data();
          auto iter = ordinal_type_1d_view(ww,iter_real_type.span());


          Impl::GkFcn<real_type, device_type>::team_invoke(member,
                           t, /// input
                           gk,
                           hks,  /// output
                           cpks, /// workspace
                           kmcd_at_i);
          member.team_barrier();


          Impl::KForwardReverse<real_type, device_type> ::team_invoke(
            member, t, p, gk, kfor_at_i, krev_at_i, iter, work_kfor_rev, kmcd_at_i);
        }
      });
    Kokkos::Profiling::popRegion();


    auto kFor_host = Kokkos::create_mirror_view(kFor);
    Kokkos::deep_copy(kFor_host, kFor);
    for (ordinal_type i = 0; i < kFor.extent(1); i++) {
      printf(" Kfor: Reac No %d,  wo mod. %e, w mod. %e, ratio %e\n", i , kFor_host(0,i), kFor_host(1,i), kFor_host(1,i)/kFor_host(0,i));
    }







  }
  Kokkos::finalize();

  return 0;

}
