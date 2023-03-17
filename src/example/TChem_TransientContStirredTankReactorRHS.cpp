
#include "TChem_Util.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_EnthalpyMass.hpp"
#include "TChem_TransientContStirredTankReactorRHS.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;


int
main(int argc, char* argv[])
{

  /// default inputs
  // std::string prefixPath("data/plug-flow-reactor/T/");
  std::string prefixPath("data/plug-flow-reactor/X/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string chemSurfFile(prefixPath + "chemSurf.inp");
  std::string thermSurfFile(prefixPath + "thermSurf.dat");
  std::string inputFile(prefixPath + "inputGas.dat");
  std::string inputFileSurf(prefixPath + "inputSurfGas.dat");
  std::string inputFilevelocity(prefixPath + "inputVelocity.dat");
  std::string outputFile(prefixPath + "CSTR_RHS.dat");

  int nBatch(1);
  bool verbose(true);

  real_type mdotIn(3.596978981250784e-06);
  real_type Vol(0.00013470);
  real_type Acat (0.0013074);
  bool isothermal(false);

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<std::string>(
    "chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>(
    "thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>(
    "chemSurffile", "Chem file name e.g., chem.inp", &chemSurfFile);
  opts.set_option<std::string>(
    "thermSurffile", "Therm file name e.g., therm.dat", &thermSurfFile);
  opts.set_option<std::string>(
    "inputFileGas", "Input state file name e.g., inputGas.dat", &inputFile);
  opts.set_option<std::string>("inputFileSurface",
                               "Input state file name e.g., inputSurfGas.dat",
                               &inputFileSurf);
  opts.set_option<std::string>(
    "outputFile",
    "Output rhs file name e.g., CSTR_RHS.dat",
    &outputFile);
  //
  opts.set_option<real_type>("Acat", "Catalytic area [m2]", &Acat);
  opts.set_option<real_type>("Vol", "Reactor Volumen [m3]", &Vol);
  opts.set_option<real_type>("mdotIn", "Inlet mass flow rate [kg/s]", &mdotIn);
  opts.set_option<bool>("isothermal", "if True, reaction is isotermic", &isothermal);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first omega values", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space().print_configuration(std::cout, detail);
    TChem::host_exec_space().print_configuration(std::cout, detail);

    using device_type      = typename Tines::UseThisDevice<exec_space>::type;

    TChem::KineticModelData kmd(
      chemFile, thermFile, chemSurfFile, thermSurfFile);
    const auto kmcd =
      TChem::createGasKineticModelConstData<device_type>(kmd); // data struc with gas phase info
    const auto kmcdSurf =
      TChem::createSurfaceKineticModelConstData<device_type>(kmd); // data struc with
                                                        // surface phase info

    /// input: state vectors: temperature, pressure and mass fraction
    real_type_2d_view state(
      "StateVector", nBatch, TChem::Impl::getStateVectorSize(kmcd.nSpec));

    // input :: surface fraction vector, zk
    real_type_2d_view siteFraction("SiteFraction", nBatch, kmcdSurf.nSpec);
    /// output: rhs for Plug Flow rector with surface reactions
    const auto nEquations = kmcd.nSpec + kmcdSurf.nSpec + 1;
    real_type_2d_view rhs("CSTR_RHS", nBatch, nEquations);
    // density, temperature, gas.nSpe, velocity, surface.nSpe

    /// create a mirror view to store input from a file
    auto state_host = Kokkos::create_mirror_view(state);

    /// create a mirror view to store input from a file
    auto siteFraction_host = Kokkos::create_mirror_view(siteFraction);

    //



    /// input from a file; this is not necessary as the input is created
    /// by other applications.
    {
      // read gas
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_0);
      TChem::Test::cloneView(state_host);
      // read surface
      auto siteFraction_host_at_0 =
        Kokkos::subview(siteFraction_host, 0, Kokkos::ALL());
      TChem::Test::readSiteFraction(
        inputFileSurf, kmcdSurf.nSpec, siteFraction_host_at_0);
      TChem::Test::cloneView(siteFraction_host);
    }

    //setting up cstr reactor
    printf("Setting up CSTR reactor\n");
    TransientContStirredTankReactorData<device_type> cstr;
    cstr.mdotIn = mdotIn; // inlet mass flow kg/s
    cstr.Vol    = Vol; // volumen of reactor m3
    cstr.Acat   = Acat; // Catalytic area m2: chemical active area
    cstr.pressure = state_host(0, 1);
    cstr.isothermal = 1;
    if (isothermal) cstr.isothermal = 0; // 0 constant temperature

    cstr.Yi = real_type_1d_view("Mass fraction at inlet", kmcd.nSpec);
    printf("Reactor residence time [s] %e\n", state_host(0, 0)*cstr.Vol/cstr.mdotIn);

    // work batch = 1
    Kokkos::parallel_for(
      Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
      KOKKOS_LAMBDA(const ordinal_type& i) {
        //mass fraction
        for (ordinal_type k = 0; k < kmcd.nSpec; k++) {
          cstr.Yi(k) = state(i,k+3);
        }
    });

    {
      real_type_2d_view EnthalpyMass("EnthalpyMass", 1, kmcd.nSpec);
      real_type_1d_view EnthalpyMixMass("EnthalpyMass Mixture", 1);

      const auto exec_space_instance = TChem::exec_space();
      TChem::EnthalpyMass::runDeviceBatch(exec_space_instance,
                                          -1,
                                          -1,
                                          1,
                                          state,
                                          EnthalpyMass,
                                          EnthalpyMixMass,
                                          kmcd);
      cstr.EnthalpyIn = EnthalpyMixMass(0);
    }

    printf("Done Setting up CSTR reactor\n" );



    Kokkos::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    Kokkos::deep_copy(siteFraction, siteFraction_host);
    const real_type t_deepcopy = timer.seconds();

    timer.reset();

    TChem::TransientContStirredTankReactorRHS::runDeviceBatch(nBatch,
                                              // inputs
                                              state,        // gas
                                              siteFraction, // surface
                                              // ouputs
                                              rhs,
                                              // data
                                              kmcd,
                                              kmcdSurf,
                                              cstr);

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

    //  create a mirror view of rhs (output) to export a file
    if (verbose) {
      auto rhs_host = Kokkos::create_mirror_view(rhs);
      Kokkos::deep_copy(rhs_host, rhs);

      /// all values are same (print only the first one)
      {
        auto rhs_host_at_0 = Kokkos::subview(rhs_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(outputFile, nEquations, rhs_host_at_0);
      }
    }

    printf("Done ...\n");
  }
  Kokkos::finalize();

  return 0;
}
