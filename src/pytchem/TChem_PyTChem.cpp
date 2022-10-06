#ifndef __TCHEM_PYTCHEM__
#define __TCHEM_PYTCHEM__

#include "TChem.hpp"
#include "TChem_Driver.hpp"

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

using namespace TChem;
namespace py = pybind11;

template <typename... Args> using py_overload_cast = py::detail::overload_cast_impl<Args...>;

struct TChemKokkosInterface {
  static void initialize() {
    if (!Kokkos::is_initialized())
      Kokkos::initialize();
  }
  static void finalize() {
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  }
  static void fence() { Kokkos::fence(); }
};

struct TChemDriver {
private:
  TChem::Driver *_obj;

public:
  TChemDriver() { _obj = new TChem::Driver(); }
  ~TChemDriver() { delete _obj; }

  ///
  /// copy utils
  ///
  void copyToNonConstView(py::array_t<ordinal_type> a, TChem::ordinal_type_1d_view_host b) {
    py::buffer_info buf = a.request();
    TCHEM_CHECK_ERROR(buf.ndim != 1, "Error: input state is not rank 1 array");
    const ordinal_type *src = (const ordinal_type *)buf.ptr;
    const int ss = buf.strides[0] / sizeof(ordinal_type);
    auto tgt = b;
    TCHEM_CHECK_ERROR(tgt.extent(0) != buf.shape[0], "Error: vector length does not match");

    /// this is a small vector
    const int m = tgt.extent(0);
    for (int ii = 0; ii < m; ++ii)
      tgt(ii) = src[ii * ss];
  }

  void copyToNonConstView(py::array_t<real_type> a, TChem::real_type_1d_view_host b) {
    py::buffer_info buf = a.request();
    TCHEM_CHECK_ERROR(buf.ndim != 1, "Error: input state is not rank 1 array");
    const real_type *src = (const real_type *)buf.ptr;
    const int ss = buf.strides[0] / sizeof(real_type);
    auto tgt = b;
    TCHEM_CHECK_ERROR(tgt.extent(0) != buf.shape[0], "Error: vector length does not match");

    /// this is a small vector
    const int m = tgt.extent(0);
    for (int ii = 0; ii < m; ++ii)
      tgt(ii) = src[ii * ss];
  }

  void copyFromConstView(TChem::real_type_1d_const_view_host a, py::array_t<real_type> b) {
    py::buffer_info buf = b.request();
    TCHEM_CHECK_ERROR(buf.ndim != 1, "Error: input state is not rank 1 array");
    auto src = a;
    real_type *tgt = (real_type *)buf.ptr;
    const int ts = buf.strides[0] / sizeof(real_type);
    TCHEM_CHECK_ERROR(buf.shape[0] != src.extent(0), "Error: vector length does not match");

    /// this is a small vector
    const int m = src.extent(0);
    for (int ii = 0; ii < m; ++ii)
      tgt[ii * ts] = src(ii);
  }

  void copyToNonConstView(py::array_t<real_type> a, TChem::real_type_2d_view_host b) {
    py::buffer_info buf = a.request();
    TCHEM_CHECK_ERROR(buf.ndim != 2, "Error: input state is not rank 2 array");

    const real_type *src = (const real_type *)buf.ptr;
    const int ss0 = buf.strides[0] / sizeof(real_type), ss1 = buf.strides[1] / sizeof(real_type);
    auto tgt = b;
    TCHEM_CHECK_ERROR(tgt.extent(0) != buf.shape[0] || tgt.extent(1) != buf.shape[1],
                      "Error: matrix length does not match");

    /// this is a small vector
    const int m = tgt.extent(0), n = tgt.extent(1);
    const Kokkos::RangePolicy<host_exec_space> policy(0, m * n);
    Kokkos::parallel_for(policy, [=](const int ij) {
      const int ii = ij / n, jj = ij % n;
      tgt(ii, jj) = src[ii * ss0 + jj * ss1];
    });
  }

  void copyFromConstView(TChem::real_type_2d_const_view_host a, py::array_t<real_type> b) {
    py::buffer_info buf = b.request();
    TCHEM_CHECK_ERROR(buf.ndim != 2, "Error: input state is not rank 2 array");

    auto src = a;
    real_type *tgt = (real_type *)buf.ptr;
    const int ts0 = buf.strides[0] / sizeof(real_type), ts1 = buf.strides[1] / sizeof(real_type);
    TCHEM_CHECK_ERROR(buf.shape[0] != src.extent(0) || buf.shape[1] != src.extent(1),
                      "Error: matrix length does not match");

    /// this is a small vector
    const int m = src.extent(0), n = src.extent(1);
    const Kokkos::RangePolicy<host_exec_space> policy(0, m * n);
    Kokkos::parallel_for(policy, [=](const int ij) {
      const int ii = ij / n, jj = ij % n;
      tgt[ii * ts0 + jj * ts1] = src(ii, jj);
    });
  }

  void copyToNonConstView(py::array_t<real_type> a, TChem::real_type_3d_view_host b) {
    py::buffer_info buf = a.request();
    TCHEM_CHECK_ERROR(buf.ndim != 3, "Error: input state is not rank 2 array");

    const real_type *src = (const real_type *)buf.ptr;
    const int ss0 = buf.strides[0] / sizeof(real_type), ss1 = buf.strides[1] / sizeof(real_type),
              ss2 = buf.strides[2] / sizeof(real_type);

    auto tgt = b;
    TCHEM_CHECK_ERROR(tgt.extent(0) != buf.shape[0] || tgt.extent(1) != buf.shape[1] || tgt.extent(2) != buf.shape[2],
                      "Error: matrix length does not match");

    /// this is a small vector
    const int m = tgt.extent(0), n = tgt.extent(1), l = tgt.extent(2), nl = n * l;

    const Kokkos::RangePolicy<host_exec_space> policy(0, m * n * l);
    Kokkos::parallel_for(policy, [=](const int ijk) {
      const int ii = ijk / (nl);
      const int ij = ijk - (ii * nl);
      const int jj = ij / l;
      const int kk = ij % l;
      tgt(ii, jj, kk) = src[ii * ss0 + jj * ss1 + kk * ss2];
    });
  }

  ///
  /// batch parallel
  /// - when a new batch size is set, all view objects are freed.
  void setNumberOfSamples(const int n_sample) { _obj->setNumberOfSamples(n_sample); }

  int getNumberOfSamples() const { return _obj->getNumberOfSamples(); }

  ///
  /// kinetic model interface
  /// - when a kinetic model recreate, all view objects are freed.
  void createGasKineticModel(const std::string &chem_file, const std::string &therm_file) {
    _obj->createGasKineticModel(chem_file, therm_file);
  }
  // using a yaml input file from cantera
  void createGasKineticModel(const std::string &chem_file) { _obj->createGasKineticModel(chem_file); }

  void cloneGasKineticModel() { _obj->cloneGasKineticModel(); }

  /// make factors unit
  void modifyGasArrheniusForwardParameters(py::array_t<ordinal_type> reac_indices, py::array_t<real_type> factors) {
    TCHEM_CHECK_ERROR(reac_indices.ndim() != 1, "Error: input reac indices is not rank 1 array");
    TCHEM_CHECK_ERROR(factors.ndim() != 3, "Error: input reac factors is not rank 3 array");

    TChem::ordinal_type_1d_view_host reac_indices_view(TChem::do_not_init_tag("reac indices"), reac_indices.shape(0));

    TChem::real_type_3d_view_host factors_view(TChem::do_not_init_tag("factors"), factors.shape(0), factors.shape(1),
                                               factors.shape(2));

    /// when input array is sliced (non contiguous array), it needs to be repacked
    copyToNonConstView(reac_indices, reac_indices_view);
    copyToNonConstView(factors, factors_view);
    _obj->modifyGasArrheniusForwardParameters(reac_indices_view, factors_view);
  }

  void createGasKineticModelConstData() { _obj->createGasKineticModelConstData(); }

  void createGasKineticModelConstDataWithArreniusForwardParameters(py::array_t<ordinal_type> reac_indices,
                                                                   py::array_t<real_type> factors) {
    // 1. make copies of kinetic model
    _obj->cloneGasKineticModel();
    // 2. modify arrhenius parameters
    modifyGasArrheniusForwardParameters(reac_indices, factors);
    // 3. create constant gas model
    _obj->createGasKineticModelConstData();
  }

  int getNumberOfSpecies() const { return _obj->getNumberOfSpecies(); }

  int getNumberOfReactions() const { return _obj->getNumberOfReactions(); }

  ///
  /// state vector setup helper
  ///
  int getLengthOfStateVector() const { return _obj->getLengthOfStateVector(); }

  int getSpeciesIndex(const std::string &species_name) const { return _obj->getSpeciesIndex(species_name); }

  int getStateVariableIndex(const std::string &var_name) {
    const int varIndex = _obj->getStateVariableIndex(var_name);
    if (varIndex == -1) {
      printf("!!! WARNING variable with name %s does not exit in state vector\n", (var_name).c_str());
    }
    return varIndex;
  }

  ///
  /// state vector
  ///
  void createStateVector() { _obj->createStateVector(); }

  void setStateVectorSingle(const int i, py::array_t<real_type> src) {
    TCHEM_CHECK_ERROR(!_obj->isStateVectorCreated(), "Error: state vector is not created");

    TChem::real_type_1d_view_host tgt;
    _obj->getStateVectorNonConstHost(i, tgt);

    copyToNonConstView(src, tgt);
  }

  py::array_t<real_type> getStateVectorSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isStateVectorCreated(), "Error: state vector is not created");
    TChem::real_type_1d_const_view_host src;
    _obj->getStateVectorHost(i, src);

    auto tgt = py::array_t<real_type>(src.extent(0));

    copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getGasArrheniusForwardParameterModel(const int imodel, py::array_t<ordinal_type> reac_indices,
                                                              const int param_index) {
    py::buffer_info reac_buf = reac_indices.request();
    TCHEM_CHECK_ERROR(reac_buf.ndim != 1, "Error: input reac indices is not rank 1 array");

    const ordinal_type *reac_ptr = (const ordinal_type *)reac_buf.ptr;
    const int rs = reac_buf.strides[0] / sizeof(ordinal_type);
    const int n_reac_buf = reac_buf.shape[0];

    auto tgt = py::array_t<real_type>(n_reac_buf);

    py::buffer_info tgt_buf = tgt.request();
    real_type *tgt_ptr = (real_type *)tgt_buf.ptr;
    const int ts = tgt_buf.strides[0] / sizeof(real_type);
    for (int i = 0; i < n_reac_buf; ++i) {
      const int reac_index = reac_ptr[i * rs];
      tgt_ptr[i * ts] = _obj->getGasArrheniusForwardParameter(imodel, reac_index, param_index);
    }
    return tgt;
  }

  py::array_t<real_type> getGasArrheniusForwardParameterReference(py::array_t<ordinal_type> reac_indices,
                                                                  const int param_index) {
    py::buffer_info reac_buf = reac_indices.request();
    TCHEM_CHECK_ERROR(reac_buf.ndim != 1, "Error: input reac indices is not rank 1 array");

    const ordinal_type *reac_ptr = (const ordinal_type *)reac_buf.ptr;
    const int rs = reac_buf.strides[0] / sizeof(ordinal_type);
    const int n_reac_buf = reac_buf.shape[0];

    auto tgt = py::array_t<real_type>(n_reac_buf);
    py::buffer_info tgt_buf = tgt.request();
    real_type *tgt_ptr = (real_type *)tgt_buf.ptr;
    const int ts = tgt_buf.strides[0] / sizeof(real_type);

    for (int i = 0; i < n_reac_buf; ++i) {
      const int reac_index = reac_ptr[i * rs];
      tgt_ptr[i * ts] = _obj->getGasArrheniusForwardParameter(reac_index, param_index);
    }
    return tgt;
  }

  void setStateVectorAll(py::array_t<real_type> src) {
    TCHEM_CHECK_ERROR(!_obj->isStateVectorCreated(), "Error: state vector is not created");

    TChem::real_type_2d_view_host tgt;
    _obj->getStateVectorNonConstHost(tgt);

    copyToNonConstView(src, tgt);
  }

  py::array_t<real_type> getStateVectorAll() {
    TCHEM_CHECK_ERROR(!_obj->isStateVectorCreated(), "Error: state vector is not created");
    TChem::real_type_2d_const_view_host src;
    _obj->getStateVectorHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1)});
    copyFromConstView(src, tgt);
    return tgt;
  }

  ///
  /// net production rate
  ///
  void createGasNetProductionRatePerMass() { _obj->createGasNetProductionRatePerMass(); }

  py::array_t<real_type> getGasNetProductionRatePerMassSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isGasNetProductionRatePerMassCreated(),
                      "Error: net production rate per mass is not created");
    TChem::real_type_1d_const_view_host src;
    _obj->getGasNetProductionRatePerMassHost(i, src);

    auto tgt = py::array_t<real_type>(src.extent(0));

    copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getGasNetProductionRatePerMassAll() {
    TCHEM_CHECK_ERROR(!_obj->isGasNetProductionRatePerMassCreated(),
                      "Error: net production rate per mass is not created");
    TChem::real_type_2d_const_view_host src;
    _obj->getGasNetProductionRatePerMassHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1)});
    copyFromConstView(src, tgt);
    return tgt;
  }

  void computeGasNetProductionRatePerMass() { _obj->computeGasNetProductionRatePerMassDevice(); }

  // jacobian

  py::array_t<real_type> getJacobianHomogeneousGasReactor() {
    TCHEM_CHECK_ERROR(!_obj->isJacobianHomogeneousGasReactorCreated(),
                      "Error: net production rate per mass is not created");
    TChem::real_type_3d_const_view_host src;
    _obj->getJacobianHomogeneousGasReactorHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1), src.extent(2)});

    // copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getJacobianHomogeneousGasReactorSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isJacobianHomogeneousGasReactorCreated(),
                      "Error: net production rate per mass is not created");
    TChem::real_type_2d_const_view_host src;
    _obj->getJacobianHomogeneousGasReactorHost(i, src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1)});

    copyFromConstView(src, tgt);
    return tgt;
  }

  void createJacobianHomogeneousGasReactor() { _obj->createJacobianHomogeneousGasReactor(); }

  void computeJacobianHomogeneousGasReactor() { _obj->computeJacobianHomogeneousGasReactorDevice(); }
  // rhs homogeneous gas reactor

  py::array_t<real_type> getRHS_HomogeneousGasReactor() {
    TCHEM_CHECK_ERROR(!_obj->isRHS_HomogeneousGasReactorCreated(),
                      "Error: rhs of homogeneous gas reactor is not created");
    TChem::real_type_2d_const_view_host src;
    _obj->getRHS_HomogeneousGasReactorHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1)});

    copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getRHS_HomogeneousGasReactorSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isRHS_HomogeneousGasReactorCreated(),
                      "Error: rhs of homogeneous gas reactor is not created");
    TChem::real_type_1d_const_view_host src;
    _obj->getRHS_HomogeneousGasReactorHost(i, src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0)});

    copyFromConstView(src, tgt);
    return tgt;
  }

  void createRHS_HomogeneousGasReactor() { _obj->createRHS_HomogeneousGasReactor(); }

  void computeRHS_HomogeneousGasReactor() { _obj->computeRHS_HomogeneousGasReactorDevice(); }

  // kforward and reverse

  py::array_t<real_type> getGasForwardReactionRateConstants() {
    TCHEM_CHECK_ERROR(!_obj->isGasReactionRateConstantsCreated(), "Error: reaction rate constants are not created");
    TChem::real_type_2d_const_view_host kfor;
    TChem::real_type_2d_const_view_host krev;
    _obj->getGasReactionRateConstantsHost(kfor, krev);

    auto tgt = py::array_t<real_type>(kfor.span());
    tgt.resize({kfor.extent(0), kfor.extent(1)});

    copyFromConstView(kfor, tgt);
    return tgt;
  }

  py::array_t<real_type> getGasReverseReactionRateConstants() {
    TCHEM_CHECK_ERROR(!_obj->isGasReactionRateConstantsCreated(), "Error: reaction rate constants are not created");
    TChem::real_type_2d_const_view_host kfor;
    TChem::real_type_2d_const_view_host krev;
    _obj->getGasReactionRateConstantsHost(kfor, krev);

    auto tgt = py::array_t<real_type>(krev.span());
    tgt.resize({krev.extent(0), krev.extent(1)});

    copyFromConstView(krev, tgt);
    return tgt;
  }

  py::array_t<real_type> getGasForwardReactionRateConstantsSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isGasReactionRateConstantsCreated(), "Error: reaction rate constants are not created");
    TChem::real_type_1d_const_view_host kfor;
    TChem::real_type_1d_const_view_host krev;
    _obj->getGasReactionRateConstantsHost(i, kfor, krev);

    auto tgt = py::array_t<real_type>(kfor.span());
    tgt.resize({kfor.extent(0)});

    copyFromConstView(kfor, tgt);
    return tgt;
  }

  py::array_t<real_type> getGasReverseReactionRateConstantsSingle(const int i) {
    TCHEM_CHECK_ERROR(!_obj->isGasReactionRateConstantsCreated(), "Error: reaction rate constants are not created");
    TChem::real_type_1d_const_view_host kfor;
    TChem::real_type_1d_const_view_host krev;
    _obj->getGasReactionRateConstantsHost(i, kfor, krev);

    auto tgt = py::array_t<real_type>(krev.span());
    tgt.resize({krev.extent(0)});

    copyFromConstView(krev, tgt);
    return tgt;
  }

  void createGasReactionRateConstants() { _obj->createGasReactionRateConstants(); }

  void computeGasReactionRateConstants() { _obj->computeGasReactionRateConstantsDevice(); }

  // enthalpy mix

  py::array_t<real_type> getGasEnthapyMass() {
    TCHEM_CHECK_ERROR(!_obj->isGasEnthapyMassCreated(), "Error: enthalpy is not created");
    TChem::real_type_2d_const_view_host src;
    _obj->getGasEnthapyMassHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0), src.extent(1)});

    copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getGasEnthapyMixMass() {
    TCHEM_CHECK_ERROR(!_obj->isGasEnthapyMassCreated(), "Error: enthalpy is not created");
    TChem::real_type_1d_const_view_host src;
    _obj->getGasEnthapyMixMassHost(src);

    auto tgt = py::array_t<real_type>(src.span());
    tgt.resize({src.extent(0)});

    copyFromConstView(src, tgt);
    return tgt;
  }

  void createGasEnthapyMass() { _obj->createGasEnthapyMass(); }

  void computeGasEnthapyMass() { _obj->computeGasEnthapyMassDevice(); }

  /// homogeneous gas reactor
  ///
  void setTimeAdvanceHomogeneousGasReactor(const real_type &tbeg, const real_type &tend, const real_type &dtmin,
                                           const real_type &dtmax, const int &jacobian_interval,
                                           const int &max_num_newton_iterations,
                                           const int &num_time_iterations_per_interval, const real_type &atol_newton,
                                           const real_type &rtol_newton, const real_type &atol_time,
                                           const real_type &rtol_time) {
    _obj->setTimeAdvanceHomogeneousGasReactor(tbeg, tend, dtmin, dtmax, jacobian_interval, max_num_newton_iterations,
                                              num_time_iterations_per_interval, atol_newton, rtol_newton, atol_time,
                                              rtol_time);
  }

  real_type computeTimeAdvanceHomogeneousGasReactor() const {
    return _obj->computeTimeAdvanceHomogeneousGasReactorDevice();
  }

  ///
  /// t and dt accessor
  ///
  py::array_t<real_type> getTimeStep() {
    TChem::real_type_1d_const_view_host src;
    _obj->getTimeStepHost(src);

    auto tgt = py::array_t<real_type>(src.extent(0));
    copyFromConstView(src, tgt);
    return tgt;
  }

  py::array_t<real_type> getTimeStepSize() {
    TChem::real_type_1d_const_view_host src;
    _obj->getTimeStepSizeHost(src);

    auto tgt = py::array_t<real_type>(src.extent(0));
    copyFromConstView(src, tgt);
    return tgt;
  }

  ///
  /// view utils
  ///
  void createAllViews() { _obj->createAllViews(); }
  void freeAllViews() { _obj->freeAllViews(); }
  void showViewStatus() { _obj->showViews("TChemDriver: View Status"); }
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
  py::class_<TChemDriver>(
      m, "TChemDriver",
      R"pbdoc(A class to manage data movement between numpy to kokkos views in TChem::Driver object)pbdoc")
      .def(py::init<>())
      /// kinetic model interface using chemkin files
      .def("createGasKineticModel",
           py_overload_cast<const std::string &, const std::string &>()(&TChemDriver::createGasKineticModel),
           py::arg("chemkin_input"), py::arg("thermo_data"), "Create a kinetic model from CHEMKIN input files")
      // using yaml cantera files
      .def("createGasKineticModel", py_overload_cast<const std::string &>()(&TChemDriver::createGasKineticModel),
           py::arg("chemkin_input"), "Create a kinetic model from yaml input files")
      /// batch parallel interface
      .def("setNumberOfSamples", (&TChemDriver::setNumberOfSamples), py::arg("number_of_samples"),
           "Set the number of samples; this is used for Kokkos view allocation")
      .def("getNumberOfSamples", (&TChemDriver::getNumberOfSamples),
           "Get the number of samples which is currently used in the driver")
      /// modify kinetic model
      .def("cloneGasKineticModel", (&TChemDriver::cloneGasKineticModel),
           "Internally create clones of the kinetic model")
      .def("modifyGasArrheniusForwardParameters", (&TChemDriver::modifyGasArrheniusForwardParameters),
           py::arg("reac_indices"), py::arg("factors"), "Modify the cloned kinetic models Arrhenius parameters")
      .def("createGasKineticModelConstData", (&TChemDriver::createGasKineticModelConstData),
           "Internally construct const object of the kinetic model and load them to device")
      .def("createGasKineticModelConstDataWithArreniusForwardParameters",
           (&TChemDriver::createGasKineticModelConstDataWithArreniusForwardParameters), py::arg("reac_indices"),
           py::arg("factors"),
           "Creates clones of the kinetic model; modifies the Arrhenius forward parameters of the clones;"
           "and creates a const object of the kinetics models."
           "factors is a 3d array of size: number of samples, number of reactions to be modified, and 3 kinetic "
           "parameters."
           " kinetic parameters: pre exponential (0), temperature coefficient(1), and activation energy(2)")
      // arrhenius parameter accessor
      .def("getGasArrheniusForwardParameter", (&TChemDriver::getGasArrheniusForwardParameterReference),
           py::arg("reac_indices"), py::arg("param_index"), py::return_value_policy::take_ownership,
           "Retrive pre exponential for reactions listed by reaction_indices")
      .def("getGasArrheniusForwardParameter", (&TChemDriver::getGasArrheniusForwardParameterModel), py::arg("imodel"),
           py::arg("reac_indices"), py::arg("param_index"), py::return_value_policy::take_ownership,
           "Retrive pre exponential for reactions listed by reaction_indices")
      /// const object interface
      .def("getNumberOfSpecies", (&TChemDriver::getNumberOfSpecies),
           "Get the number of species registered in the kinetic model")
      .def("getNumberOfReactions", (&TChemDriver::getNumberOfReactions),
           "Get the number of reactions registered in the kinetic model")
      /// state vector helpers
      .def("getLengthOfStateVector", (&TChemDriver::getLengthOfStateVector),
           "Get the size of state vector i.e., rho, P, T, Y_{0-Nspec-1}")
      .def("getSpeciesIndex", (&TChemDriver::getSpeciesIndex), py::arg("species_name"), "Get species index")
      .def("getStateVariableIndex", (&TChemDriver::getStateVariableIndex), py::arg("var_name"),
           "Get state variable index ")
      /// state vector
      .def("createStateVector", (&TChemDriver::createStateVector),
           "Allocate memory for state vector (# samples, state vector length)")
      .def("setStateVector", (&TChemDriver::setStateVectorSingle), py::arg("sample_index"), py::arg("1d_state_vector"),
           "Overwrite state vector for a single sample")
      .def("setStateVector", (&TChemDriver::setStateVectorAll), py::arg("2d_state_vector"),
           "Overwrite state vector for all samples")
      .def("getStateVector", (&TChemDriver::getStateVectorSingle), py::arg("sample_index"),
           py::return_value_policy::take_ownership, "Retrieve state vector for a single sample")
      .def("getStateVector", (&TChemDriver::getStateVectorAll), py::return_value_policy::take_ownership,
           "Retrieve state vector for all samples")
      /// net production rate
      .def("createGasNetProductionRatePerMass", (&TChemDriver::createGasNetProductionRatePerMass),
           "Allocate memory for net production rate per mass (# samples, # species)")
      .def("getGasNetProductionRatePerMass", (&TChemDriver::getGasNetProductionRatePerMassSingle),
           py::arg("sample_index"), py::return_value_policy::take_ownership,
           "Retrive net production rate for a single sample")
      .def("getGasNetProductionRatePerMass", (&TChemDriver::getGasNetProductionRatePerMassAll),
           py::return_value_policy::take_ownership, "Retrieve net production rate for all samples")
      .def("computeGasNetProductionRatePerMass", (&TChemDriver::computeGasNetProductionRatePerMass),
           "Compute net production rate")
      // jacobian homogeneous gas reactor
      .def("createJacobianHomogeneousGasReactor", (&TChemDriver::createJacobianHomogeneousGasReactor),
           "Allocate memory for homogeneous-gas-reactor Jacobian  (# samples, # species + 1  # species + 1)")
      .def("getJacobianHomogeneousGasReactor", (&TChemDriver::getJacobianHomogeneousGasReactorSingle),
           py::arg("sample_index"), py::return_value_policy::take_ownership,
           "Retrive homogeneous-gas-reactor Jacobian for a single sample")
      //  .def("getJacobianHomogeneousGasReactor",
      // (&TChemDriver::getJacobianHomogeneousGasReactor), py::return_value_policy::take_ownership,
      // "Retrieve homogeneous-gas-reactor Jacobian for all samples")
      .def("computeJacobianHomogeneousGasReactor", (&TChemDriver::computeJacobianHomogeneousGasReactor),
           "Compute Jacobian matrix for homogeneous gas reactor")
      // rhs homogeneous gas reactor
      .def("createRHS_HomogeneousGasReactor", (&TChemDriver::createRHS_HomogeneousGasReactor),
           "Allocate memory for homogeneous-gas-reactor RHS  (# samples, # species + 1 )")
      .def("getRHS_HomogeneousGasReactor", (&TChemDriver::getRHS_HomogeneousGasReactorSingle), py::arg("sample_index"),
           py::return_value_policy::take_ownership, "Retrive homogeneous-gas-reactor RHS for a single sample")
      .def("getRHS_HomogeneousGasReactor", (&TChemDriver::getRHS_HomogeneousGasReactor),
           py::return_value_policy::take_ownership, "Retrieve homogeneous-gas-reactor RHS_ for all samples")
      .def("computeRHS_HomogeneousGasReactor", (&TChemDriver::computeRHS_HomogeneousGasReactor),
           "Compute RHS for homogeneous gas reactor")
      // kforwar and reverse
      .def("createGasReactionRateConstants", (&TChemDriver::createGasReactionRateConstants),
           "Allocate memory for forward/reverse rate constants  (# samples, # reactions )")
      .def("getGasForwardReactionRateConstants", (&TChemDriver::getGasForwardReactionRateConstantsSingle),
           py::arg("sample_index"), py::return_value_policy::take_ownership,
           "Retrive forward rate constants for a single sample")
      .def("getGasReverseReactionRateConstants", (&TChemDriver::getGasReverseReactionRateConstantsSingle),
           py::arg("sample_index"), py::return_value_policy::take_ownership,
           "Retrive reverse rate constants for a single sample")
      .def("getGasForwardReactionRateConstants", (&TChemDriver::getGasForwardReactionRateConstants),
           py::return_value_policy::take_ownership, "Retrieve forward rate constants  for all samples")
      .def("getGasReverseReactionRateConstants", (&TChemDriver::getGasReverseReactionRateConstants),
           py::return_value_policy::take_ownership, "Retrieve reverse rate constants  for all samples")
      .def("computeGasReactionRateConstants", (&TChemDriver::computeGasReactionRateConstants),
           "Compute forward/reverse rate constant")
      // ehthalpy mix
      .def("createGasEnthapyMass", (&TChemDriver::createGasEnthapyMass),
           "Allocate memory for enthalpy mass  (# samples, # species )")
      .def("getGasEnthapyMass", (&TChemDriver::getGasEnthapyMass), py::return_value_policy::take_ownership,
           "Retrive enthalpy mass per species for all samples")
      .def("getGasEnthapyMixMass", (&TChemDriver::getGasEnthapyMixMass), py::return_value_policy::take_ownership,
           "Retrieve mixture enthalpy for all samples")
      .def("computeGasEnthapyMass", (&TChemDriver::computeGasEnthapyMass), "Compute enthalpy mass and mixture enthalpy")

      /// homogeneous gas reactor
      .def("setTimeAdvanceHomogeneousGasReactor", (&TChemDriver::setTimeAdvanceHomogeneousGasReactor), py::arg("tbeg"),
           py::arg("tend"), py::arg("dtmin"), py::arg("dtmax"), py::arg("jacobian_interval"),
           py::arg("max_num_newton_iterations"), py::arg("num_time_iterations_per_interval"), py::arg("atol_newton"),
           py::arg("rtol_newton"), py::arg("atol_time"), py::arg("rtol_time"),
           "Set time advance object for homogeneous gas reactor")
      .def("computeTimeAdvanceHomogeneousGasReactor", (&TChemDriver::computeTimeAdvanceHomogeneousGasReactor),
           "Compute Time Advance for a Homogeneous-Gas Reactor ")
      /// time step accessors
      .def("getTimeStep", (&TChemDriver::getTimeStep), py::return_value_policy::take_ownership,
           "Retrieve time line of all samples")
      .def("getTimeStepSize", (&TChemDriver::getTimeStepSize), py::return_value_policy::take_ownership,
           "Retrieve time step sizes of all samples")
      /// view utils
      .def("createAllViews", (&TChemDriver::createAllViews), "Allocate all necessary workspace for this driver")
      .def("freeAllViews", (&TChemDriver::freeAllViews), "Free all necessary workspace for this driver")
      .def("showViewStatus", (&TChemDriver::showViewStatus), "Print member variable view status");

  m.attr("__version__") = "development";
}

#endif
