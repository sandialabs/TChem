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
#include "TChem.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
/// if we do not want to profile the code undefine the following
/// #define TCHEM_ENABLE_CUDA_PROFILE
#endif
#if defined(TCHEM_ENABLE_CUDA_PROFILE)
#include "cuda_profiler_api.h"
#define TCHEM_PROFILER_REGION_BEGIN { CUDA_SAFE_CALL(cudaProfilerStart()); }
#define TCHEM_PROFILER_REGION_END   { CUDA_SAFE_CALL(cudaProfilerStop()); }
#else
#define TCHEM_PROFILER_REGION_BEGIN {}
#define TCHEM_PROFILER_REGION_END  {}
#endif

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int
main(int argc, char* argv[])
{
  TCHEM_PROFILER_REGION_END;

  /// default inputs
  std::string prefixPath("");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");
  std::string outputFile(prefixPath + "omega.dat");
  std::string outputFileTimes(prefixPath + "wall_times.dat");

  int nBatch(1), league_size(-1), team_size(-1), vector_size(-1);
  bool verbose(true), use_sample_format(false), use_shared_workspace(true);

  bool run_source_term(true), run_analytic_Jacobian(true) ,
    run_numerical_Jacobian(true), run_numerical_Jacobian_fwd(true),
    run_sacado_Jacobian(true);


  /// parse command line arguments
  TChem::CommandLineParser opts(
                                "This example computes reaction rates with a given state vector");
  opts.set_option<std::string>(
                               "chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>(
                               "thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>(
                               "inputfile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<std::string>(
                               "outputfile", "Output omega file name e.g., omega.dat", &outputFile);
  //
  opts.set_option<std::string>(
                               "output-file-times", "Wal times file name e.g., times.dat", &outputFileTimes);
  opts.set_option<int>(
                       "league_size", "league size ", &league_size);
  opts.set_option<int>(
                       "team_thread_size", "time thread size ", &team_size);
  //
  opts.set_option<int>(
                       "vector_thread_size", "vector thread size ", &vector_size);
  opts.set_option<int>(
                       "batchsize",
                       "Batchsize the same state vector described in statefile is cloned",
                       &nBatch);
  //
  opts.set_option<bool>(
                        "use-sample-format", "If true, input file does not have header or format", &use_sample_format);
  opts.set_option<bool>(
                        "use-shared-workspace", "If true, it use team policy shared scratch memory for workspace", &use_shared_workspace);
  opts.set_option<bool>(
                        "verbose", "If true, printout the first omega values", &verbose);
  opts.set_option<bool>(
                        "run_source_term", "If true, runs source term computation", &run_source_term);
  opts.set_option<bool>(
                        "run_analytic_Jacobian", "If true, runs analytic Jacobian computation", &run_analytic_Jacobian);
  opts.set_option<bool>(
                        "run_numerical_Jacobian", "If true, runs numerical(Richardson Extrapolation) Jacobian computation", &run_numerical_Jacobian);
  //
  opts.set_option<bool>(
                        "run_numerical_Jacobian_fwd", "If true, runs numerical (Forward Difference) Jacobian computation", &run_numerical_Jacobian_fwd);
  opts.set_option<bool>(
                        "run_sacado_Jacobian", "If true, runs analytic, sacado,  Jacobian computation", &run_sacado_Jacobian);


  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    using device_type      = typename Tines::UseThisDevice<exec_space>::type;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);
    const auto exec_space_instance = TChem::exec_space();

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const auto kmcd = kmd.createConstData<device_type>();

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    real_type_2d_view_host state_host;

    if (use_sample_format){
      // read gas
      const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
      Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);

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

    } else{
      state_host = real_type_2d_view_host("state vector host", nBatch, stateVecDim);
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_0);
      TChem::Test::cloneView(state_host);

    }

    real_type_2d_view state("StateVector Devices", nBatch, stateVecDim);

    using policy_type = typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
    Kokkos::Impl::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    const real_type t_deepcopy = timer.seconds();

    const ordinal_type level = 1;

    FILE* fout_times = fopen(outputFileTimes.c_str(), "w");
    fprintf(fout_times," Computation type, total time [sec], time per sample [sec/sample]\n ");

    if (run_source_term) {
      printf("Running source-term computations using: team_size %d vector_size %d... \n", team_size, vector_size);
      real_type_2d_view source_term("source term", nBatch, kmcd.nSpec + 1);

      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      if (league_size > 0 && team_size > 0 && vector_size > 0) {
        policy = policy_type(exec_space_instance, league_size, team_size, vector_size);
      }

      const ordinal_type per_team_extent = SourceTerm::getWorkSpaceSize(kmcd);
      ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      printf("source term per_team extent %d\n", per_team_extent);
      real_type_2d_view workspace;
      if (use_shared_workspace) {
        policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
      } else {
        workspace = real_type_2d_view("workspace", policy.league_size(), per_team_extent);
      }
      Kokkos::fence();
      
      /// skip the first run
      SourceTerm::runDeviceBatch( policy, state, source_term, workspace, kmcd);
      exec_space_instance.fence();
      timer.reset();
      TCHEM_PROFILER_REGION_BEGIN;
      SourceTerm::runDeviceBatch( policy, state, source_term, workspace, kmcd);
      TCHEM_PROFILER_REGION_END;
      exec_space_instance.fence();
      const real_type t_device_batch = timer.seconds();
      fprintf(fout_times,"Source Term,%15.10e,%15.10e\n ", t_device_batch, t_device_batch / real_type(nBatch));

      if (verbose) {
        auto source_term_host = Kokkos::create_mirror_view(source_term);
        Kokkos::deep_copy(source_term_host, source_term);

        {
          auto outputFile_source_term =  "source_term_" + outputFile ;
          auto source_term_host_at_0 = Kokkos::subview(source_term_host, 0, Kokkos::ALL());
          TChem::Test::writeReactionRates(outputFile_source_term, kmcd.nSpec+1, source_term_host_at_0);
        }
      }
    }

    // hand derived jacobian
    if (run_analytic_Jacobian)  {
      printf("Running analytic-Jacobian computations using: team_size %d vector_size %d... \n", team_size, vector_size);

      real_type_3d_view analytic_Jacobian("analytic_Jacobian", nBatch, kmcd.nSpec + 1, kmcd.nSpec + 1);

      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      if (league_size > 0 && team_size > 0 && vector_size > 0) {
        policy = policy_type(exec_space_instance, league_size, team_size, vector_size);
      }

      const ordinal_type per_team_extent = JacobianReduced::getWorkSpaceSize(kmcd);
      ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      
      real_type_2d_view workspace;
      if (use_shared_workspace) {
        policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
      } else {
        workspace = real_type_2d_view("workspace", policy.league_size(), per_team_scratch);
      }
      Kokkos::fence();
      TChem::JacobianReduced::runDeviceBatch(policy, state, analytic_Jacobian, workspace, kmcd);
      exec_space_instance.fence();
      timer.reset();
      TCHEM_PROFILER_REGION_BEGIN;
      TChem::JacobianReduced::runDeviceBatch(policy, state, analytic_Jacobian, workspace, kmcd);
      TCHEM_PROFILER_REGION_END;
      exec_space_instance.fence();
      const real_type t_device_batch = timer.seconds();
      fprintf(fout_times,"Analytic Jacobian,%15.10e,%15.10e\n ", t_device_batch, t_device_batch / real_type(nBatch));

      if (verbose) {
        auto analytic_Jacobian_host = Kokkos::create_mirror_view(analytic_Jacobian);
        Kokkos::deep_copy(analytic_Jacobian_host, analytic_Jacobian);

        /// all values are same (print only the first one)
        {
          auto outputFile_analytic_Jacobian =  "analytic_Jacobian_" + outputFile ;
          auto analytic_Jacobian_host_at_0 = Kokkos::subview(analytic_Jacobian_host, 0, Kokkos::ALL(), Kokkos::ALL());
          TChem::Test::writeJacobian(outputFile_analytic_Jacobian, kmcd.nSpec, analytic_Jacobian_host_at_0);
        }
      }
    }

    //numerical jacobian
    if (run_numerical_Jacobian)  {
      printf("Running numerical-Jacobian (Richardson Extrapolation) computations using: team_size %d vector_size %d... \n", team_size, vector_size);

      real_type_3d_view numerical_Jacobian("numerical_Jacobian", nBatch, kmcd.nSpec + 1, kmcd.nSpec + 1);
      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      if (league_size > 0 && team_size > 0 && vector_size > 0) {
        policy = policy_type(exec_space_instance, league_size, team_size, vector_size);
      }

      const ordinal_type per_team_extent = IgnitionZeroDNumJacobian::getWorkSpaceSize(kmcd);
      ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      real_type_2d_view workspace;
      if (use_shared_workspace) {
        policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
      } else {
        workspace = real_type_2d_view("workspace", policy.league_size(), per_team_scratch);
      }

      real_type_2d_view fac("fac", nBatch, kmcd.nSpec + 1);
      Kokkos::fence();

      IgnitionZeroDNumJacobian::runDeviceBatch( policy, state, numerical_Jacobian, fac, workspace, kmcd);
      exec_space_instance.fence();
      timer.reset();
      TCHEM_PROFILER_REGION_BEGIN;
      IgnitionZeroDNumJacobian::runDeviceBatch( policy, state, numerical_Jacobian, fac, workspace, kmcd);
      TCHEM_PROFILER_REGION_END;
      exec_space_instance.fence();
      const real_type t_device_batch = timer.seconds();
      fprintf(fout_times,"Numerical Jacobian,%15.10e,%15.10e\n ", t_device_batch, t_device_batch / real_type(nBatch));

      if (verbose) {
        auto numerical_Jacobian_host = Kokkos::create_mirror_view(numerical_Jacobian);
        Kokkos::deep_copy(numerical_Jacobian_host, numerical_Jacobian);

        /// all values are same (print only the first one)
        {
          auto outputFile_numerical_Jacobian = "numerical_Jacobian_" + outputFile;
          auto numerical_Jacobian_host_at_0 = Kokkos::subview(numerical_Jacobian_host, 0, Kokkos::ALL(), Kokkos::ALL());
          TChem::Test::writeJacobian(outputFile_numerical_Jacobian, kmcd.nSpec, numerical_Jacobian_host_at_0);
        }
      }

    }

    //numerical jacobian
    if (run_numerical_Jacobian_fwd)  {
      printf("Running numerical-Jacobian (Forward Difference) computations using: team_size %d vector_size %d... \n", team_size, vector_size);

      real_type_3d_view numerical_Jacobian_fwd("numerical_Jacobian_fwd", nBatch, kmcd.nSpec + 1, kmcd.nSpec + 1);

      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      if (league_size > 0 && team_size > 0 && vector_size > 0) {
        policy = policy_type(exec_space_instance, league_size, team_size, vector_size);
      }

      const ordinal_type per_team_extent = IgnitionZeroDNumJacobianFwd::getWorkSpaceSize(kmcd);
      ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      real_type_2d_view workspace;
      if (use_shared_workspace) {
        policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
      } else {
        workspace = real_type_2d_view("workspace", policy.league_size(), per_team_extent);
      }

      real_type_2d_view fac("fac", nBatch, kmcd.nSpec + 1);
      Kokkos::fence();

      IgnitionZeroDNumJacobianFwd::runDeviceBatch( policy, state, numerical_Jacobian_fwd, fac, workspace, kmcd);
      exec_space_instance.fence();
      timer.reset();
      IgnitionZeroDNumJacobianFwd::runDeviceBatch( policy, state, numerical_Jacobian_fwd, fac, workspace, kmcd);
      exec_space_instance.fence();
      const real_type t_device_batch = timer.seconds();
      fprintf(fout_times,"Numerical Jacobian Fwd,%15.10e,%15.10e\n ", t_device_batch, t_device_batch / real_type(nBatch));

      if (verbose) {
        /// all values are same (print only the first one)
        auto numerical_Jacobian_fwd_host = Kokkos::create_mirror_view(numerical_Jacobian_fwd);
        Kokkos::deep_copy(numerical_Jacobian_fwd_host, numerical_Jacobian_fwd);
        {
          auto outputFile_numerical_Jacobian_fwd = "numerical_Jacobian_fwd_" + outputFile;
          auto numerical_Jacobian_fwd_host_at_0 = Kokkos::subview(numerical_Jacobian_fwd_host, 0, Kokkos::ALL(), Kokkos::ALL());
          TChem::Test::writeJacobian(outputFile_numerical_Jacobian_fwd, kmcd.nSpec, numerical_Jacobian_fwd_host_at_0);
        }
      }
    }


    // sacado jacobian
    if (run_sacado_Jacobian)  {
      printf("Running analytical-Jacobian (sacado) computations using: team_size %d vector_size %d... \n", team_size, vector_size);
      const ordinal_type m = kmcd.nSpec+1;
      printf("nBatch %d, m %d\n", nBatch, m);
      real_type_3d_view sacado_Jacobian("sacado_Jacobian", nBatch, m, m);

      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      if (league_size > 0 && team_size > 0 && vector_size > 0) {
        policy = policy_type(exec_space_instance, league_size, team_size, vector_size);
      }

      const ordinal_type per_team_extent = IgnitionZeroD_SacadoJacobian::getWorkSpaceSize(kmcd);
      ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      real_type_2d_view workspace;
      if (use_shared_workspace) {
        policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
      } else {
        workspace = real_type_2d_view("workspace", policy.league_size(), per_team_extent);
      }
      Kokkos::fence();
      IgnitionZeroD_SacadoJacobian::runDeviceBatch( policy, state, sacado_Jacobian, workspace, kmcd);
      exec_space_instance.fence();
      timer.reset();
      TCHEM_PROFILER_REGION_BEGIN;
      IgnitionZeroD_SacadoJacobian::runDeviceBatch( policy, state, sacado_Jacobian, workspace, kmcd);
      TCHEM_PROFILER_REGION_END;
      exec_space_instance.fence();
      const real_type t_device_batch = timer.seconds();
      fprintf(fout_times,"Sacado Jacobian,%15.10e,%15.10e\n ", t_device_batch, t_device_batch / real_type(nBatch));

      if (verbose) {
        auto sacado_Jacobian_host = Kokkos::create_mirror_view(sacado_Jacobian);
        Kokkos::deep_copy(sacado_Jacobian_host, sacado_Jacobian);

        /// all values are same (print only the first one)
        {
          auto outputFile_sacado_Jacobian = "sacado_Jacobian_" +  outputFile;
          auto sacado_Jacobian_host_at_0 = Kokkos::subview(sacado_Jacobian_host, 0, Kokkos::ALL(), Kokkos::ALL());
          TChem::Test::writeJacobian(outputFile_sacado_Jacobian, kmcd.nSpec, sacado_Jacobian_host_at_0);
        }
      }
    }

    fclose(fout_times);
    /// show time
    printf("---------------------------------------------------\n");
    printf("Testing Arguments: \n batch size %d\n jac dimension %d\n chemfile %s\n thermfile %s\n inputfile %s\n outputfile %s\n verbose %s\n",
           nBatch,
           (kmcd.nSpec+1),
           chemFile.c_str(),
           thermFile.c_str(),
           inputFile.c_str(),
           outputFile.c_str(),
           verbose ? "true" : "false");


  }
  Kokkos::finalize();

  return 0;
}
