/* =====================================================================================
TChem version 2.1.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

This file is part of TChem. TChem is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */


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
  void createKineticModel(const std::string& chem_file, const std::string& therm_file) {
    _obj->createKineticModel(chem_file, therm_file);
  }
  void setNumberOfSamples(const int n_sample) {
    _obj->setNumberOfSamples(n_sample);
  }
  int getNumberOfSamples() const {
    return _obj->getNumberOfSamples();
  }
  int getLengthOfStateVector() const {
    return _obj->getLengthOfStateVector();
  }
  int getNumerOfSpecies() const {
    return _obj->getNumerOfSpecies();
  }
  int getNumerOfReactions() const {
    return _obj->getNumerOfReactions();
  }
  void createStateVector() {
    _obj->createStateVector();
  }
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
      .def("createKineticModel",
	   py_overload_cast<const std::string&,const std::string&>()(&TChemDriver::createKineticModel),
	   py::arg("chemkin_input"), py::arg("thermo_data"),
	   "Create a kinetic model from CHEMKIN input files")
      .def("setNumberOfSamples",
	   (&TChemDriver::setNumberOfSamples), py::arg("number_of_samples"),
	   "Set the number of samples; this is used for Kokkos view allocation")
      .def("getNumberOfSamples",
	   (&TChemDriver::getNumberOfSamples),
	   "Get the number of samples which is currently used in the driver")
      .def("getLengthOfStateVector",
	   (&TChemDriver::getLengthOfStateVector),
	   "Get the size of state vector i.e., rho, P, T, Y_{0-Nspec-1}")
      .def("getNumerOfSpecies",
	   (&TChemDriver::getNumerOfSpecies),
	   "Get the number of species registered in the kinetic model")
      .def("getNumerOfReactions",
	   (&TChemDriver::getNumerOfReactions),
	   "Get the number of reactions registered in the kinetic model")
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
	   (&TChemDriver::computeNetProductionRatePerMass),
	   "Compute net production rate")
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
