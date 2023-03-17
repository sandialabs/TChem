#include "TChem_CommandLineParser.hpp"
#include "TChem_Util.hpp"
#include "Sacado.hpp"
#include "Tines.hpp"

#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_EnthalpyMass.hpp"
#include "TChem_Impl_TransientContStirredTankReactor_Problem.hpp"

int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("runs/T-CSTR/CH4_PT_Quinceno2006/inputs/");
  std::string chemFile(prefixPath + "chemgri30.inp");
  std::string thermFile(prefixPath + "thermgri30.dat");
  std::string inputFile(prefixPath + "sample_phi1.dat");
  std::string chemSurfFile(prefixPath +"chemSurf.inp");
  std::string thermSurfFile(prefixPath +"thermSurf.dat");
  std::string inputFileSurf( prefixPath +"inputSurf.dat");

  bool verbose(true);

  int nBatch(1);

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<std::string>(
    "chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>(
    "thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>(
    "inputfile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<std::string>
  ("chemSurffile","Chem file name e.g., chemSurf.inp",
   &chemSurfFile);
  opts.set_option<std::string>
  ("thermSurffile", "Therm file name e.g.,thermSurf.dat",
  &thermSurfFile);
  opts.set_option<std::string>
  ("inputSurffile", "Input state file name e.g., inputSurfGas.dat", &inputFileSurf);

  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first omega values", &verbose);
  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    using TChem::real_type;
    using TChem::ordinal_type;
    using fad_type = Sacado::Fad::SLFad<real_type,100>;

    using ats = Tines::ats<real_type>;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space,host_memory_space>;

    using problem_type = TChem::Impl::TransientContStirredTankReactor_Problem<fad_type, host_device_type>;

    using real_type_1d_view_type = typename problem_type::real_type_1d_view_type;
    using real_type_2d_view_type = typename problem_type::real_type_2d_view_type;

    using value_type_1d_view_type = typename problem_type::value_type_1d_view_type;

    using cstr_data_type = TransientContStirredTankReactorData<host_device_type>;

    const bool detail = false;

    TChem::exec_space().print_configuration(std::cout, detail);
    TChem::host_exec_space().print_configuration(std::cout, detail);

    TChem::KineticModelData kmd(
      chemFile, thermFile, chemSurfFile, thermSurfFile);
    const auto kmcd = TChem::createGasKineticModelConstData<host_device_type>(kmd);
    const auto kmcdSurf = TChem::createSurfaceKineticModelConstData<host_device_type>(kmd);

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    real_type_2d_view_type state;
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
                              state,
                              nBatch);
    }

    real_type_2d_view_type siteFraction;
    const auto SurfSpeciesNamesHost =
      Kokkos::create_mirror_view(kmcdSurf.speciesNames);
    Kokkos::deep_copy(SurfSpeciesNamesHost, kmcdSurf.speciesNames);

    {
      TChem::Test::readSurfaceSample(inputFileSurf,
                                     SurfSpeciesNamesHost,
                                     kmcdSurf.nSpec,
                                     siteFraction,
                                     nBatch);

    }

    real_type mdotIn(1e-1);
    real_type Vol(0.00013470);
    real_type Acat (0.0013074);


    //setting up cstr reactor
    printf("Setting up CSTR reactor\n");
    cstr_data_type cstr;
    cstr.mdotIn = mdotIn; // inlet mass flow kg/s
    cstr.Vol    = Vol; // volumen of reactor m3
    cstr.Acat   = Acat; // Catalytic area m2: chemical active area
    cstr.pressure = state(0, 1);


    cstr.Yi = real_type_1d_view_type("Mass fraction at inlet", kmcd.nSpec);
    printf("Reactor residence time [s] %e\n", state(0, 0)*cstr.Vol/cstr.mdotIn);

    // work batch = 1
    Kokkos::parallel_for(
      Kokkos::RangePolicy<TChem::host_exec_space>(0, nBatch),
      KOKKOS_LAMBDA(const ordinal_type& i) {
        //mass fraction
        for (ordinal_type k = 0; k < kmcd.nSpec; k++) {
          cstr.Yi(k) = state(i,k+3);
        }
    });

    {
      real_type_2d_view_type EnthalpyMass("EnthalpyMass", 1, kmcd.nSpec);
      real_type_1d_view_type EnthalpyMixMass("EnthalpyMass Mixture", 1);
      using policy_type =
        typename TChem::UseThisTeamPolicy<TChem::host_exec_space>::type;
      const ordinal_type level = 1;
      const ordinal_type per_team_extent =

      TChem::EnthalpyMass::getWorkSpaceSize(kmcd); ///
      const ordinal_type per_team_scratch =
          Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);
      policy_type policy(TChem::host_exec_space(), nBatch, Kokkos::AUTO());
      policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

      TChem::EnthalpyMass::runHostBatch(policy,
                                          state,
                                          EnthalpyMass,
                                          EnthalpyMixMass,
                                          kmcd);
      cstr.EnthalpyIn = EnthalpyMixMass(0);
    }

    problem_type problem;
    problem._kmcd = kmcd;
    problem._kmcdSurf = kmcdSurf;
    problem._cstr = cstr;

    const int m = problem.getNumberOfEquations();
    const int wlen = problem.getWorkSpaceSize();
    real_type_1d_view_type work("work", wlen);
    problem.setWorkspace(work);

    const int n = problem.getNumberOfEquations();

    real_type_1d_view_type x("x", n);
    x(0) = state(0,2); // temperature
    for (ordinal_type k = 0; k < kmcd.nSpec; k++) {
      x(k+1) = state(0,k+3); // mass fraction
      printf("mass fraction k %d %f\n", k, x(k+1) );
    }

    for (ordinal_type k = 0; k < kmcdSurf.nSpec; k++) {
      x(k + kmcd.nSpec+1) = siteFraction(0,k); // site fraction
      printf("site fraction k %d %e\n", k, x(k + kmcd.nSpec+1) );
    }

    printf(" temp %e\n", x(0) );

    real_type_2d_view_type J_s("J_sacado",  m, n );
    const auto member = Tines::HostSerialTeamMember();

    problem.computeSacadoJacobian(member, x, J_s);

    Tines::showMatrix("SacadoJacobian", J_s);

    /// numeric tests
    auto compareJacobian = [m](const std::string &label, auto &A, auto &B) {
      real_type err(0), norm(0);
      real_type max_diff(0);
      int max_i(0);
      int max_j(0);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
          const real_type diff = ats::abs(A(i, j) - B(i, j));
          const real_type val = ats::abs(A(i, j));
          if (max_diff < diff)
          {
            max_diff = diff;
            max_i = i;
            max_j = j;
          }
          norm += val * val;
          err += diff * diff;
        }
      const real_type rel_err = ats::sqrt(err / norm);

      const real_type max_rel_err = (A(max_i, max_j)-B(max_i, max_j))/A(max_i, max_j);

      // Tines::showMatrix(label, B);
      const real_type margin = 1e2, threshold = ats::epsilon() * margin;
      if (rel_err < threshold)
        std::cout << "PASS ";
      else
        std::cout << "FAIL ";
      std::cout << label << " relative error : " << rel_err
                << " within threshold : " << threshold <<"\n\n";

      std::cout  << " row idx : " << max_i   << " column idx : " << max_j
                 << " max absolute error : " << max_diff
                 << " Ja : "      << A(max_i, max_j) << " Js : "    << B(max_i, max_j)
                 << " max relavite error :"  << max_rel_err <<"\n\n";
    };

    real_type_1d_view_type fac(
      "fac", problem_type::getNumberOfEquations(kmcd, kmcdSurf));

    problem._fac = fac;

    real_type_2d_view_type J_n("J_numerical",  m, n);
    problem.computeNumericalJacobian(member, x, J_n);
    Tines::showMatrix("NumericalJacobian", J_n);

    compareJacobian(std::string("SacadovsNumerical"), J_n, J_s);




  }
  Kokkos::finalize();

  return 0;
}
