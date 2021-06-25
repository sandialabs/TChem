#ifndef __TCHEM_PYTCHEM__
#define __TCHEM_PYTCHEM__


#include "TChem.hpp"
#include "TChem_Driver.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

using namespace TChem;
namespace py = pybind11;

template <typename... Args>
using py_overload_cast = py::detail::overload_cast_impl<Args...>;

struct TChemKokkosInterface {
  static void initialize() {
    if (!Kokkos::is_initialized())
      Kokkos::initialize();
  }
  static void finalize() {
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  }
  static void fence() {
    Kokkos::fence();
  }
};

struct TChemDriver {
private:
  TChem::Driver * _obj;

public:
  TChemDriver() {
    _obj = new TChem::Driver();
  }
  ~TChemDriver() {
    delete _obj;
  }

  ///
  /// copy utils
  ///
  void copyToNonConstView(py::array_t<real_type> a, typename TChem::Driver::real_type_1d_view_host b) {
    py::buffer_info buf = a.request();
    TCHEM_CHECK_ERROR(buf.ndim != 1, "Error: input state is not rank 1 array");
    const real_type * src = (const real_type*)buf.ptr;
    const int ss = buf.strides[0]/sizeof(real_type);
    auto tgt = b;
    TCHEM_CHECK_ERROR(tgt.extent(0) != buf.shape[0], "Error: vector length does not match");

    /// this is a small vector
    const int m = tgt.extent(0);
    for (int ii=0;ii<m;++ii)
      tgt(ii) = src[ii*ss];
  }

  void copyFromConstView(typename TChem::Driver::real_type_1d_const_view_host a, py::array_t<real_type> b) {
    py::buffer_info buf = b.request();
    TCHEM_CHECK_ERROR(buf.ndim != 1, "Error: input state is not rank 1 array");
    auto src = a;
    real_type * tgt = (real_type*)buf.ptr;
    const int ts = buf.strides[0]/sizeof(real_type);
    TCHEM_CHECK_ERROR(buf.shape[0] != src.extent(0), "Error: vector length does not match");

    /// this is a small vector
    const int m = src.extent(0);
    for (int ii=0;ii<m;++ii)
      tgt[ii*ts] = src(ii);
  }

  void copyToNonConstView(py::array_t<real_type> a, typename TChem::Driver::real_type_2d_view_host b) {
    py::buffer_info buf = a.request();
    TCHEM_CHECK_ERROR(buf.ndim != 2, "Error: input state is not rank 2 array");

    const real_type * src = (const real_type*)buf.ptr;
    const int ss0 = buf.strides[0]/sizeof(real_type), ss1 = buf.strides[1]/sizeof(real_type);
    auto tgt = b;
    TCHEM_CHECK_ERROR(tgt.extent(0) != buf.shape[0] ||
		      tgt.extent(1) != buf.shape[1] , "Error: matrix length does not match");

    /// this is a small vector
    const int m = tgt.extent(0), n = tgt.extent(1);
    const Kokkos::RangePolicy<host_exec_space> policy(0, m*n);
    Kokkos::parallel_for(policy, [=](const int ij) {
      const int ii = ij/n, jj = ij%n;
      tgt(ii,jj) = src[ii*ss0+jj*ss1];
    });
  }

  void copyFromConstView(typename TChem::Driver::real_type_2d_const_view_host a, py::array_t<real_type> b) {
    py::buffer_info buf = b.request();
    TCHEM_CHECK_ERROR(buf.ndim != 2, "Error: input state is not rank 2 array");

    auto src = a;
    real_type * tgt = (real_type*)buf.ptr;
    const int ts0 = buf.strides[0]/sizeof(real_type), ts1 = buf.strides[1]/sizeof(real_type);
    TCHEM_CHECK_ERROR(buf.shape[0] != src.extent(0) ||
		      buf.shape[1] != src.extent(1) , "Error: matrix length does not match");

    /// this is a small vector
    const int m = src.extent(0), n = src.extent(1);
    const Kokkos::RangePolicy<host_exec_space> policy(0, m*n);
    Kokkos::parallel_for(policy, [=](const int ij) {
      const int ii = ij/n, jj = ij%n;
      tgt[ii*ts0+jj*ts1] = src(ii,jj);
    });
  }

  ///
  /// kinetic model interface
  /// - when a kinetic model recreate, all view objects are freed.
  void createKineticModel(const std::string& chem_file, const std::string& therm_file) {
    _obj->createKineticModel(chem_file, therm_file);
  }

  int getNumberOfSpecies() const {
    return _obj->getNumberOfSpecies();
  }

  int getNumberOfReactions() const {
    return _obj->getNumberOfReactions();
  }

  ///
  /// batch parallel
  /// - when a new batch size is set, all view objects are freed.
  void setNumberOfSamples(const int n_sample) {
    _obj->setNumberOfSamples(n_sample);
  }

  int getNumberOfSamples() const {
    return _obj->getNumberOfSamples();
  }

  ///
  /// state vector setup helper
  ///
  int getLengthOfStateVector() const {
    return _obj->getLengthOfStateVector();
  }

  int getSpeciesIndex(const std::string& species_name) const {
    return _obj->getSpeciesIndex(species_name);
  }

  int getStateVariableIndex(const std::string& var_name) {
    const int varIndex = _obj->getStateVariableIndex(var_name);
    if (varIndex == -1) {
      printf("!!! WARNING variable with name %s does not exit in state vector\n",(var_name).c_str() );
    }
    return varIndex;
  }

  ///
  /// state vector
  ///
  void createStateVector() {
    _obj->createStateVector();
  }

  void setStateVectorSingle(const int i, py::array_t<real_type> src) {
    TCHEM_CHECK_ERROR(!_obj->isStateVectorCreated(), "Error: state vector is not created");

    typename TChem::Driver::real_type_1d_view_host tgt;
    _obj->getStateVectorNonConstHost(i, tgt);

    copyToNonConstView(src, tgt);
  }

  py::array_t<real_type> getStateVectorSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isStateVectorCreated(), "Error: state vector is not created");
    typename TChem::Driver::real_type_1d_const_view_host src;
    _obj->getStateVectorHost(i, src);

    auto tgt = py::array_t<real_type>(src.extent(0));

    copyFromConstView(src, tgt);
    return tgt;
  }

  void setStateVectorAll(py::array_t<real_type> src) {
    TCHEM_CHECK_ERROR(!_obj->isStateVectorCreated(), "Error: state vector is not created");

    typename TChem::Driver::real_type_2d_view_host tgt;
    _obj->getStateVectorNonConstHost(tgt);

    copyToNonConstView(src, tgt);
  }

  py::array_t<real_type> getStateVectorAll() {
    TCHEM_CHECK_ERROR(!_obj->isStateVectorCreated(), "Error: state vector is not created");
    typename TChem::Driver::real_type_2d_const_view_host src;
    _obj->getStateVectorHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0),src.extent(1)});
    copyFromConstView(src, tgt);
    return tgt;
  }

  ///
  /// net production rate
  ///
  void createNetProductionRatePerMass() {
    _obj->createNetProductionRatePerMass();
  }

  py::array_t<real_type> getNetProductionRatePerMassSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isNetProductionRatePerMassCreated(), "Error: net production rate per mass is not created");
    typename TChem::Driver::real_type_1d_const_view_host src;
    _obj->getNetProductionRatePerMassHost(i, src);

    auto tgt = py::array_t<real_type>(src.extent(0));

    copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getNetProductionRatePerMassAll() {
    TCHEM_CHECK_ERROR(!_obj->isNetProductionRatePerMassCreated(), "Error: net production rate per mass is not created");
    typename TChem::Driver::real_type_2d_const_view_host src;
    _obj->getNetProductionRatePerMassHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1)});
    copyFromConstView(src, tgt);
    return tgt;
  }

  void computeNetProductionRatePerMass() {
    _obj->computeNetProductionRatePerMassDevice();
  }

  // jacobian

  py::array_t<real_type> getJacobianHomogeneousGasReactor() {
    TCHEM_CHECK_ERROR(!_obj->isJacobianHomogeneousGasReactorCreated(), "Error: net production rate per mass is not created");
    typename TChem::Driver::real_type_3d_const_view_host src;
    _obj->getJacobianHomogeneousGasReactorHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1), src.extent(2)});

    // copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getJacobianHomogeneousGasReactorSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isJacobianHomogeneousGasReactorCreated(), "Error: net production rate per mass is not created");
    typename TChem::Driver::real_type_2d_const_view_host src;
    _obj->getJacobianHomogeneousGasReactorHost(i, src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1)});

    copyFromConstView(src, tgt);
    return tgt;
  }

  void createJacobianHomogeneousGasReactor() {
    _obj->createJacobianHomogeneousGasReactor();
  }

  void computeJacobianHomogeneousGasReactor() {
    _obj->computeJacobianHomogeneousGasReactorDevice();
  }
  // rhs

  py::array_t<real_type> getRHS_HomogeneousGasReactor() {
    TCHEM_CHECK_ERROR(!_obj->isRHS_HomogeneousGasReactorCreated(), "Error: rhs of homogeneous gas reactor is not created");
    typename TChem::Driver::real_type_2d_const_view_host src;
    _obj->getRHS_HomogeneousGasReactorHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1)});

    copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getRHS_HomogeneousGasReactorSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isRHS_HomogeneousGasReactorCreated(), "Error: rhs of homogeneous gas reactor is not created");
    typename TChem::Driver::real_type_1d_const_view_host src;
    _obj->getRHS_HomogeneousGasReactorHost(i, src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0)});

    copyFromConstView(src, tgt);
    return tgt;
  }

  void createRHS_HomogeneousGasReactor() {
    _obj->createRHS_HomogeneousGasReactor();
  }

  void computeRHS_HomogeneousGasReactor() {
    _obj->computeRHS_HomogeneousGasReactorDevice();
  }

  ///
  /// homogeneous gas reactor
  ///
  void setTimeAdvanceHomogeneousGasReactor(const real_type & tbeg,
					   const real_type & tend,
					   const real_type & dtmin,
					   const real_type & dtmax,
					   const real_type & max_num_newton_iterations,
					   const real_type & num_time_iterations_per_interval,
					   const real_type& atol_newton, const real_type&rtol_newton,
					   const real_type& atol_time, const real_type& rtol_time) {
    _obj->setTimeAdvanceHomogeneousGasReactor(tbeg, tend, dtmin, dtmax,
					      max_num_newton_iterations, num_time_iterations_per_interval,
					      atol_newton, rtol_newton,
					      atol_time, rtol_time);
  }

  real_type computeTimeAdvanceHomogeneousGasReactor() const {
    return  _obj->computeTimeAdvanceHomogeneousGasReactorDevice();
  }

  ///
  /// t and dt accessor
  ///
  void unsetTimeAdvance() {
    _obj->unsetTimeAdvance();
  }

  py::array_t<real_type> getTimeStep() {
    typename TChem::Driver::real_type_1d_const_view_host src;
    _obj->getTimeStepHost(src);

    auto tgt = py::array_t<real_type>(src.extent(0));
    copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getTimeStepSize() {
    typename TChem::Driver::real_type_1d_const_view_host src;
    _obj->getTimeStepSizeHost(src);

    auto tgt = py::array_t<real_type>(src.extent(0));
    copyFromConstView(src, tgt);
    return tgt;
  }

  ///
  /// view utils
  ///
  void createAllViews() {
    _obj->createAllViews();
  }
  void freeAllViews() {
    _obj->freeAllViews();
  }
  void showViewStatus() {
    _obj->showViews("TChemDriver: View Status");
  }
};

PYBIND11_MODULE(pytchem, m) {
    m.doc() = R"pbdoc(
        TChem for Python
        ----------------
        .. currentmodule:: pytchem
        .. autosummary::
           :toctree: _generate
        KokkosHelper
        TChemDriver
    )pbdoc";
    m.def("initialize", &TChemKokkosInterface::initialize, "Initialize Kokkos");
    m.def("finalize", &TChemKokkosInterface::finalize, "Finalize Kokkos");
    m.def("fence", &TChemKokkosInterface::fence, "Execute Kokkos::fence");
    py::class_<TChemDriver>
      (m,
       "TChemDriver",
       R"pbdoc(A class to manage data movement between numpy to kokkos views in TChem::Driver object)pbdoc")
      .def(py::init<>())
      //// kinetic model interface
      .def("createKineticModel",
	   py_overload_cast<const std::string&,const std::string&>()(&TChemDriver::createKineticModel),
	   py::arg("chemkin_input"), py::arg("thermo_data"),
	   "Create a kinetic model from CHEMKIN input files")
      .def("getNumberOfSpecies",
	   (&TChemDriver::getNumberOfSpecies),
	   "Get the number of species registered in the kinetic model")
      .def("getNumberOfReactions",
	   (&TChemDriver::getNumberOfReactions),
	   "Get the number of reactions registered in the kinetic model")
      //// batch parallel interface
      .def("setNumberOfSamples",
	   (&TChemDriver::setNumberOfSamples), py::arg("number_of_samples"),
	   "Set the number of samples; this is used for Kokkos view allocation")
      .def("getNumberOfSamples",
	   (&TChemDriver::getNumberOfSamples),
	   "Get the number of samples which is currently used in the driver")
      /// state vector helpers
      .def("getLengthOfStateVector",
	   (&TChemDriver::getLengthOfStateVector),
	   "Get the size of state vector i.e., rho, P, T, Y_{0-Nspec-1}")
      .def("getSpeciesIndex",
           (&TChemDriver::getSpeciesIndex),py::arg("species_name"),
           "Get species index" )
      .def("getStateVariableIndex",
           (&TChemDriver::getStateVariableIndex), py::arg("var_name"), "Get state variable index " )
      /// state vector
      .def("createStateVector",
	   (&TChemDriver::createStateVector),
	   "Allocate memory for state vector (# samples, state vector length)")
      .def("setStateVector",
	   (&TChemDriver::setStateVectorSingle), py::arg("sample_index"), py::arg("1d_state_vector"),
	   "Overwrite state vector for a single sample")
      .def("setStateVector",
	   (&TChemDriver::setStateVectorAll), py::arg("2d_state_vector"),
	   "Overwrite state vector for all samples")
      .def("getStateVector",
	   (&TChemDriver::getStateVectorSingle), py::arg("sample_index"), py::return_value_policy::take_ownership,
	   "Retrieve state vector for a single sample")
      .def("getStateVector",
	   (&TChemDriver::getStateVectorAll), py::return_value_policy::take_ownership,
	   "Retrieve state vector for all samples")
      /// net production rate
      .def("createNetProductionRatePerMass",
	   (&TChemDriver::createNetProductionRatePerMass),
	   "Allocate memory for net production rate per mass (# samples, # species)")
      .def("getNetProductionRatePerMass",
	   (&TChemDriver::getNetProductionRatePerMassSingle), py::arg("sample_index"), py::return_value_policy::take_ownership,
	   "Retrive net production rate for a single sample")
      .def("getNetProductionRatePerMass",
	   (&TChemDriver::getNetProductionRatePerMassAll), py::return_value_policy::take_ownership,
	   "Retrieve net production rate for all samples")
      .def("computeNetProductionRatePerMass",
	   (&TChemDriver::computeNetProductionRatePerMass), "Compute net production rate")
     // jacobian homogeneous gas reactor
     .def("createJacobianHomogeneousGasReactor",
    (&TChemDriver::createJacobianHomogeneousGasReactor),
    "Allocate memory for homogeneous-gas-reactor Jacobian  (# samples, # species + 1  # species + 1)")
     .def("getJacobianHomogeneousGasReactor",
    (&TChemDriver::getJacobianHomogeneousGasReactorSingle), py::arg("sample_index"), py::return_value_policy::take_ownership,
    "Retrive homogeneous-gas-reactor Jacobian for a single sample")
    //  .def("getJacobianHomogeneousGasReactor",
    // (&TChemDriver::getJacobianHomogeneousGasReactor), py::return_value_policy::take_ownership,
    // "Retrieve homogeneous-gas-reactor Jacobian for all samples")
     .def("computeJacobianHomogeneousGasReactor",
    (&TChemDriver::computeJacobianHomogeneousGasReactor), "Compute Jacobian matrix for homogeneous gas reactor")
    // rhs homogeneous gas reactor
    .def("createRHS_HomogeneousGasReactor",
   (&TChemDriver::createRHS_HomogeneousGasReactor),
   "Allocate memory for homogeneous-gas-reactor RHS  (# samples, # species + 1 )")
    .def("getRHS_HomogeneousGasReactor",
   (&TChemDriver::getRHS_HomogeneousGasReactorSingle), py::arg("sample_index"), py::return_value_policy::take_ownership,
   "Retrive homogeneous-gas-reactor RHS for a single sample")
    .def("getRHS_HomogeneousGasReactor",
   (&TChemDriver::getRHS_HomogeneousGasReactor), py::return_value_policy::take_ownership,
   "Retrieve homogeneous-gas-reactor RHS_ for all samples")
    .def("computeRHS_HomogeneousGasReactor",
   (&TChemDriver::computeRHS_HomogeneousGasReactor), "Compute RHS for homogeneous gas reactor")

      /// homogeneous gas reactor
      .def("setTimeAdvanceHomogeneousGasReactor",(&TChemDriver::setTimeAdvanceHomogeneousGasReactor),
           py::arg("tbeg"),  py::arg("tend"), py::arg("dtmin"), py::arg("dtmax"),
           py::arg("max_num_newton_iterations"), py::arg("num_time_iterations_per_interval"),
           py::arg("atol_newton"), py::arg("rtol_newton"),
           py::arg("atol_time"), py::arg("rtol_time"),
           "Set time advance object for homogeneous gas reactor")
      .def("computeTimeAdvanceHomogeneousGasReactor",
           (&TChemDriver::computeTimeAdvanceHomogeneousGasReactor),
           "Compute Time Advance for a Homogeneous-Gas Reactor ")
      /// time step accessors
      .def("getTimeStep",
	   (&TChemDriver::getTimeStep), py::return_value_policy::take_ownership,
	   "Retrieve time line of all samples")
      .def("getTimeStepSize",
	   (&TChemDriver::getTimeStepSize), py::return_value_policy::take_ownership,
	   "Retrieve time step sizes of all samples")
      /// view utils
      .def("createAllViews",
	   (&TChemDriver::createAllViews),
	   "Allocate all necessary workspace for this driver")
      .def("freeAllViews",
	   (&TChemDriver::freeAllViews),
	   "Free all necessary workspace for this driver")
      .def("showViewStatus",
	   (&TChemDriver::showViewStatus),
	   "Print member variable view status")
      ;

    m.attr("__version__") = "development";
}

#endif
