#include "TChem_CommandLineParser.hpp"
#include "TChem_Util.hpp"
#include "Sacado.hpp"
#include "Tines.hpp"
#include "TChem_Impl_PlugFlowReactor_Problem.hpp"
#include "TChem_Impl_CpMixMs.hpp"
int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("data/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "sample.dat");
  std::string chemSurfFile("chemSurf.inp");
  std::string thermSurfFile("thermSurf.dat");
  std::string inputFileSurf( "inputSurf.dat");
  std::string inputFilevelocity( "inputVelocity.dat");

  bool verbose(true);

  int nBatch(1), team_size(-1), vector_size(-1);

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

  opts.set_option<std::string>
  ("inputVelocityfile", "Input state file name e.g., inputVelocity.dat", &inputFilevelocity);

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
    using real_type = TChem::real_type;
    using ordinal_type = TChem::ordinal_type;
    using fad_type = Sacado::Fad::SLFad<real_type,100>;

    using ats = Tines::ats<real_type>;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space,host_memory_space>;

    using problem_type = TChem::Impl::PlugFlowReactor_Problem<fad_type, host_device_type>;

    using real_type_1d_view_type = typename problem_type::real_type_1d_view_type;
    using real_type_2d_view_type = typename problem_type::real_type_2d_view_type;

    using value_type_1d_view_type = typename problem_type::value_type_1d_view_type;

    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

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
    real_type_1d_view_type velocity;
    const auto SurfSpeciesNamesHost =
      Kokkos::create_mirror_view(kmcdSurf.speciesNames);
    Kokkos::deep_copy(SurfSpeciesNamesHost, kmcdSurf.speciesNames);

    {
      TChem::Test::readSurfaceSample(inputFileSurf,
                                     SurfSpeciesNamesHost,
                                     kmcdSurf.nSpec,
                                     siteFraction,
                                     nBatch);

      // read velocity (velocity)
      velocity = real_type_1d_view_type("Velocity host", nBatch);
      TChem::Test::read1DVector(inputFilevelocity, nBatch, velocity);

    }

    real_type Area(0.00053);
    real_type Pcat(0.025977239243415308);

    PlugFlowReactorData pfrd;
    pfrd.Area = Area; // m2
    pfrd.Pcat = Pcat; //

    problem_type problem;
    problem._kmcd = kmcd;
    problem._kmcdSurf = kmcdSurf;
    problem._pfrd = pfrd;

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
    x(kmcd.nSpec+1) = state(0,0); //density
    x(kmcd.nSpec+2) = velocity(0); //velocity

    for (ordinal_type k = 0; k < kmcdSurf.nSpec; k++) {
      x(k + kmcd.nSpec+3) = siteFraction(0,k); // site fraction
      printf("site fraction k %d %f\n", k, x(k + kmcd.nSpec+3) );
    }

    printf(" temp %e\n", x(0) );
    printf(" density %e\n", x(kmcd.nSpec+1) );
    printf(" vel %e\n", x(kmcd.nSpec+2) );

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


    {
      real_type_1d_view_type cpk("dcp", kmcd.nSpec);
      const auto state_at_i = Kokkos::subview(state, 0, Kokkos::ALL());
      const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i);
      const real_type t = sv_at_i.Temperature();
      const real_type p = sv_at_i.Pressure();
      const real_type_1d_view_type Ys = sv_at_i.MassFractions();
      auto dCpmix = Impl::CpMixMsDerivative::team_invoke(member, t, Ys, cpk, kmcd );
      printf("%e \t",dCpmix );

      auto Cpmix = Impl::CpMixMs<real_type, host_device_type>::team_invoke(member, t, Ys, cpk, kmcd );
      for (size_t k = 0; k < kmcd.nSpec; k++) {
        printf("%e \t" , cpk(k) );
      }
      printf("\n");

    }



  }
  Kokkos::finalize();

  return 0;
}
