#include "TChem_CommandLineParser.hpp"
#include "TChem_Util.hpp"
#include "Sacado.hpp"
#include "Tines.hpp"
#include "TChem_Impl_IgnitionZeroD_Problem.hpp"


int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("data/H2/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "sample.dat");
  // std::string outputFile(prefixPath + "CpMixMass.dat");
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
  // opts.set_option<std::string>("outputfile", "Output omega file name e.g.,
  // omega.dat", &outputFile);
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
    using fad_type = Sacado::Fad::SLFad<real_type,1000>;
    // using fad_type = Sacado::Fad::DFad<real_type>;

    using ats = Tines::ats<real_type>;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space,host_memory_space>;

    using problem_type = TChem::Impl::IgnitionZeroD_Problem<fad_type, host_device_type>;

    using real_type_1d_view_type = typename problem_type::real_type_1d_view_type;
    using real_type_2d_view_type = typename problem_type::real_type_2d_view_type;

    using value_type_1d_view_type = typename problem_type::value_type_1d_view_type;

    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const auto kmcd = TChem::createGasKineticModelConstData<host_device_type>(kmd);

    const ordinal_type stateVecDim = TChem::Impl::getStateVectorSize(kmcd.nSpec);

    real_type_2d_view_type state_host;
    const auto speciesNamesHost = kmcd.speciesNames;
    {
      // get species molecular weigths
      const auto SpeciesMolecularWeights =kmcd.sMass;
      TChem::Test::readSample(inputFile,
                              speciesNamesHost,
                              SpeciesMolecularWeights,
                              kmcd.nSpec,
                              stateVecDim,
                              state_host,
                              nBatch);
    }

    real_type_2d_view_type state("StateVector Devices", nBatch, stateVecDim);
    Kokkos::deep_copy(state, state_host);
    // output:: CpMass
    real_type_2d_view_type CpMass("CpMass", nBatch, kmcd.nSpec);
    real_type_1d_view_type CpMixMass("CpMass Mixture", nBatch);
    value_type_1d_view_type CpW("CpMass", kmcd.nSpec, kmcd.nSpec+1);

    problem_type problem;
    problem._kmcd = kmcd;
    const int m = problem.getNumberOfEquations();
    int wlen = problem.getWorkSpaceSize(kmcd);
    real_type_1d_view_type work("work", wlen);
    problem.setWorkspace(work);

    const int n = problem.getNumberOfEquations();

    real_type_1d_view_type x("x", n);
    x(0) = state(0,2);
    for (ordinal_type k = 0; k < kmcd.nSpec; k++) {
      x(k+1) = state(0,k+3);
      printf("mass fraction k %d %e\n", k, x(k+1) );
    }

    problem._p = state(0,1);//fad_type(state(0,1));
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

    {
      real_type_1d_view_type fac(
        "fac", problem_type::getNumberOfEquations(kmcd));
      //
      real_type_2d_view_type J_a("J_analytical",  m, n);

      /// constant values of the problem
      /// initialize problem
      problem._fac = fac;    // fac for numerical jacobian

      problem.computeAnalyticalJacobian(member, x, J_a);

      Tines::showMatrix("AnalyticalJacobian", J_a);

      compareJacobian(std::string("AnalyticSacado"), J_a, J_s);

      member.team_barrier();

      real_type_2d_view_type J_n("J_numerical",  m, n);

      problem.computeNumericalJacobian(member, x, J_n);

      Tines::showMatrix("NumericalJacobian", J_n);







    }





  }
  Kokkos::finalize();

  return 0;
}
