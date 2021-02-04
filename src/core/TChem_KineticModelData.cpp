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
#include "TChem_KineticModelData.hpp"
#include "TC_kmodint.hpp"

namespace TChem {

KineticModelData::KineticModelData(const std::string& mechfile,
                                   const std::string& thermofile)
{
  char* mechfile_name_ptr = const_cast<char*>(mechfile.c_str());
  char* thermofile_name_ptr = const_cast<char*>(thermofile.c_str());

  // Given input files, it generates kmod.list file
  TC_kmodint_(mechfile_name_ptr, thermofile_name_ptr);

  // Using the kmod.list, it populates data arrays
  initChem();
}

KineticModelData::KineticModelData(const std::string& mechfile,
                                   const std::string& thermofile,
                                   const std::string& mechSurffile,
                                   const std::string& thermoSurffile)
{
  char* mechfile_name_ptr = const_cast<char*>(mechfile.c_str());
  char* thermofile_name_ptr = const_cast<char*>(thermofile.c_str());
  char* mechSurffile_name_ptr = const_cast<char*>(mechSurffile.c_str());
  char* thermSurfofile_name_ptr = const_cast<char*>(thermoSurffile.c_str());

  // Given input files, it generates kmod.list file
  TC_kmodint_(mechfile_name_ptr, thermofile_name_ptr);

  // Using the kmod.list, it populates data arrays
  initChem();

  auto eNamesHost = eNames_.view_host();
  auto eMassHost = eMass_.view_host();
  auto sNamesHost = sNames_.view_host();
  auto sMassHost = sMass_.view_host();
  auto elemCountHost = elemCount_.view_host();
  auto sPhaseHost = sPhase_.view_host();
  auto sChargeHost = sCharge_.view_host();
  auto TloHost = Tlo_.view_host();
  auto TmiHost = Tmi_.view_host();
  auto ThiHost = Thi_.view_host();
  auto cppolHost = cppol_.view_host();

  element* listelem;
  // listelem = (element  *) malloc(nElem_ * sizeof(listelem[0])) ;
  std::vector<char> element_names(nElem_ * sizeof(listelem[0]));
  listelem = (element*)(&element_names[0u]);

  for (int i = 0; i < nElem_; i++) {
    listelem[i].mass = eMassHost(i);
    // memcpy( listelem[i].name, &eNamesHost(i,0)+i*LENGTHOFELEMNAME,
    // LENGTHOFELEMNAME * sizeof(char) );
    strncat(listelem[i].name, &eNamesHost(i, 0), LENGTHOFELEMNAME);
  }

  /* Assemble species container */
  speciesGas* listspec;
  // listspec = (speciesGas  *) malloc(nSpec_ * sizeof(listspec[0])) ;

  std::vector<char> species_names(nSpec_ * sizeof(listspec[0]));
  listspec = (speciesGas*)(&species_names[0u]);

  for (int i = 0; i < nSpec_; i++) {
    listspec[i].mass = sMassHost(i);
    strncat(listspec[i].name, &sNamesHost(i, 0), LENGTHOFSPECNAME);
    // memcpy(listspec[i].name, &sNamesHost(i,0)+i*LENGTHOFSPECNAME,
    // LENGTHOFSPECNAME * sizeof(char) );
  }

  /* Element content */
  // ElemCounts
  for (int i = 0; i < nSpec_; i++) {
    listspec[i].elemcontent = (int*)malloc(nElem_ * sizeof(int));
    for (int j = 0; j < nElem_; j++)
      listspec[i].elemcontent[j] = elemCountHost(i, j);
  }

  for (int i = 0; i < nSpec_; i++) {
    listspec[i].phase = sPhaseHost(i);
    listspec[i].charge = sChargeHost(i);
    listspec[i].nasapoltemp[0] = TloHost(i);
    listspec[i].nasapoltemp[1] = TmiHost(i);
    listspec[i].nasapoltemp[2] = ThiHost(i);
  }

  for (int i = 0; i < nSpec_; i++) {
    const auto ptr = &cppolHost(i, 0, 0);
    for (int j = 0; j < 14; j++)
      listspec[i].nasapolcoefs[j] = ptr[j];
  }

  infoGasChem modelGas;
  modelGas.Nelem = nElem_;
  modelGas.Nspec = nSpec_;
  modelGas.infoE = listelem;
  modelGas.infoS = listspec;

  // printf("Elements present in gas phase\n");
  // for ( int i=0 ; i<nElem_ ; i++ )
  //   printf( "%-3d\t%-4s\t%f10.7\n",i+1,&eNamesHost(i,0),eMassHost(i)) ;
  //
  // printf("Species present in gas phase\n");
  // for (int i=0; i<nSpec_; i++)
  //   printf( "%-3d\t%-4s\t%f10.7\n",i+1,&sNamesHost(i,0),sMassHost(i)) ;

  // //  /* call the interpreter */
  TC_kmodint_surface_(
    mechSurffile_name_ptr, thermSurfofile_name_ptr, &modelGas);

  initChemSurf();
}

void
KineticModelData::allocateViews(FILE* errfile)
{
  std::string errmsg;

  /* Elements' name and weights */
  eNames_ = string_type_1d_dual_view<LENGTHOFELEMNAME + 1>(
    do_not_init_tag("KMD::eNames"), nElem_);
  eMass_ = real_type_1d_dual_view(do_not_init_tag("KMD::eMass"), nElem_);

  /* Species' name and weights */
  sNames_ = string_type_1d_dual_view<LENGTHOFSPECNAME + 1>(
    do_not_init_tag("KMD::sNames"), nSpec_);
  sMass_ = real_type_1d_dual_view(do_not_init_tag("KMD::sMass"), nSpec_);

  /* Species' elemental composition */
  elemCount_ = ordinal_type_2d_dual_view(
    do_not_init_tag("KMD::elemCount"), nSpec_, nElem_);

  /* Species charges, no of tempfits, phase */
  sCharge_ = ordinal_type_1d_dual_view(do_not_init_tag("KMD::sCharge"), nSpec_);
  sTfit_ = ordinal_type_1d_dual_view(do_not_init_tag("KMD::sTfit"), nSpec_);
  sPhase_ = ordinal_type_1d_dual_view(do_not_init_tag("KMD::sPhase"), nSpec_);

  /* Range of temperatures for thermo fits */
  Tlo_ = real_type_1d_dual_view(do_not_init_tag("KMD::Tlo"), nSpec_);
  Tmi_ = real_type_1d_dual_view(do_not_init_tag("KMD::Tmi"), nSpec_);
  Thi_ = real_type_1d_dual_view(do_not_init_tag("KMD::Thi"), nSpec_);

  /* Polynomial coeffs for thermo fits */
  cppol_ = real_type_3d_dual_view(
    do_not_init_tag("KMD::cppol"), nSpec_, nNASAinter_, (nCpCoef_ + 2));

  if (nNASA9coef_ > 0) {
    spec9t_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::spec9t"), nNASA9coef_);
    spec9nrng_ = ordinal_type_1d_dual_view(do_not_init_tag("KMD::spec9tnrng"),
                                           nNASA9coef_);
    spec9trng_ = real_type_3d_dual_view(
      do_not_init_tag("KMD::spec9trng"), nNASA9coef_, NTH9RNGMAX, 2);
    spec9coefs_ = real_type_3d_dual_view(
      do_not_init_tag("KMD::spec9coefs"), nNASA9coef_, NTH9RNGMAX, 9);
  }

  /* Ionic species */
  if (nIonEspec_ > 0) {
    sNion_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::sNion"), nIonEspec_);
  }

  /* reaction info */
  if (nReac_ > 0) {
    isRev_ = ordinal_type_1d_dual_view(do_not_init_tag("KMD::isRev"), nReac_);
    reacNrp_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacNrp"), nReac_);
    reacNreac_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacNrp"), nReac_);
    reacNprod_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacNrp"), nReac_);
    reacNuki_ = ordinal_type_2d_dual_view(
      do_not_init_tag("KMD::reacNrp"), nReac_, maxSpecInReac_);
    reacSidx_ = ordinal_type_2d_dual_view(
      do_not_init_tag("KMD::reacNrp"), nReac_, maxSpecInReac_);
    reacScoef_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacScoef"), nReac_);
    reacArhenFor_ =
      real_type_2d_dual_view(do_not_init_tag("KMD::reacArhenFor"), nReac_, 3);
    isDup_ = ordinal_type_1d_dual_view(do_not_init_tag("KMD::isDup"), nReac_);
  }

  /* Reactions with reversible Arrhenius parameters given */
  if (nRevReac_ > 0) {
    reacRev_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacRev"), nRevReac_);
    reacArhenRev_ = real_type_2d_dual_view(
      do_not_init_tag("KMD::reacArhenRev"), nRevReac_, 3);
  }

  /* Pressure-dependent reactions */
  if (nFallReac_ > 0) {
    reacPfal_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacPfal"), nFallReac_);
    reacPtype_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacPtype"), nFallReac_);
    reacPlohi_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacPlohi"), nFallReac_);
    reacPspec_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacPspec"), nFallReac_);
    reacPpar_ = real_type_2d_dual_view(
      do_not_init_tag("KMD::reacPpar"), nFallReac_, nFallPar_);
  }

  /* Third-body reactions */
  if (nThbReac_ > 0) {
    reacTbdy_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacTbdy"), nThbReac_);
    reacTbno_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacTbno"), nThbReac_);
    reac_to_Tbdy_index_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::reac_to_Tbdy_index"), nReac_);
    specTbdIdx_ = ordinal_type_2d_dual_view(
      do_not_init_tag("KMD::specTbdIdx"), nThbReac_, maxTbInReac_);
    specTbdEff_ = real_type_2d_dual_view(
      do_not_init_tag("KMD::specTbdEff"), nThbReac_, maxTbInReac_);
  }

  /* Reactions with real stoichiometric coefficients */
  if (nRealNuReac_ > 0) {
    reacRnu_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacRnu"), nRealNuReac_);
    reacRealNuki_ = real_type_2d_dual_view(
      do_not_init_tag("KMD::reacRealNuki"), nRealNuReac_, maxSpecInReac_);
  }

  /* Arbitrary reaction orders */
  if (nOrdReac_ > 0) {
    reacAOrd_ =
      ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacAOrd"), nOrdReac_);
    specAOidx_ = ordinal_type_2d_dual_view(
      do_not_init_tag("KMD::specAOidx"), nOrdReac_, maxOrdPar_);
    specAOval_ = real_type_2d_dual_view(
      do_not_init_tag("KMD::specAOval"), nOrdReac_, maxOrdPar_);
  }

  ///
  /// KJ: original header does not include the following member values nor views
  ///

  // /* Landau-Teller reactions */
  // if ( nLtReac_ > 0 )
  //   errmsg += "kmod.list : Error: Landau-Teller reactions are not supported
  //   yet !!!\n";

  // /* reverse Landau-Teller reactions */
  // if ( nRltReac_ > 0 )
  //   errmsg += "kmod.list : Error: Landau-Teller reactions are not supported
  //   yet !!!";

  // /* radiation wavelength */
  // if ( nHvReac_ > 0 ) {
  //   reacHvIdx_ = ordinal_type_1d_dual_view(do_not_init_tag("KMD::reacHvIdx"),
  //   nHvReac_); reacHvPar_ =
  //   real_type_1d_dual_view(do_not_init_tag("KMD::reacHvPar"), nHvReac_);
  // }

  // /* JAN fits */
  // if ( nJanReac_ > 0 )
  //   errmsg += "kmod.list : Error: JAN fits are not supported yet !!!\n";

  // /* FIT1 fits */
  // if ( nFit1Reac_ > 0 )
  //   errmsg += "kmod.list : Error: FIT1 fits are not supported yet !!!\n";

  // /* Excitation reactions */
  // if ( nExciReac_ > 0 )
  //   errmsg += "kmod.list : Error: Excitation reactions are not supported yet
  //   !!!\n";

  // /* Plasma Momentum transfer collision */
  // if ( nMomeReac_ > 0 )
  //   errmsg += "kmod.list : Error: Plasma Momentum transfer collision is not
  //   supported yet !!!\n";

  // /* Ion Momentum transfer collision */
  // if ( nXsmiReac_ > 0 )
  //   errmsg += "kmod.list : Error: Ion Momentum transfer collision is not
  //   supported yet !!!\n";

  // /* Species temperature dependency */
  // if ( nTdepReac_ > 0 )
  //   errmsg += "kmod.list : Error: Custom species temperature dependency is
  //   not supported yet !!!\n";

  /* Reactions with PLOG formulation */
  if (nPlogReac_ > 0) {
    reacPlogIdx_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::reacPlogIdx"), nPlogReac_);
    reacPlogPno_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::reacPlogPno"), nPlogReac_ + 1);
    // KJ: reacPlogPars should be allocated  after reacPlogPno_ is initialized
    // reacPlogPars_ =
    // real_type_2d_dual_view(do_not_init_tag("KMD::reacPlogPars"),
    // reacPlogPno_(nPlogReac_), 4);
  }

  ///
  /// KJ: The following members do not seem to follow the same naming
  /// conventions. Please check.
  ///
  /* some arrays */
  /* sum(nu) for each reaction */
  sigNu_ = ordinal_type_1d_dual_view(do_not_init_tag("KMD::sigNu"), nReac_);
  NuIJ_ =
    ordinal_type_2d_dual_view(do_not_init_tag("KMD::NuIJ"), nReac_, nSpec_);
  kc_coeff_ = real_type_1d_dual_view(do_not_init_tag("KMD::kc_coeff"), nReac_);

  if (nRealNuReac_ > 0) {
    sigRealNu_ =
      real_type_1d_dual_view(do_not_init_tag("KMD::sigRealNu"), nRealNuReac_);
    RealNuIJ_ = real_type_2d_dual_view(
      do_not_init_tag("KMD::RealNuIJ"), nRealNuReac_, nSpec_);
  }

  if (errmsg.length() > 0) {
    fprintf(errfile, "Error: %s\n", errmsg.c_str());
    std::runtime_error("Error: TChem::KineticModelData \n" + errmsg);
  }
}

void
KineticModelData::allocateViewsSurf(FILE* errfile)
{
  std::string errmsg;
  /* Species' name and weights */
  TCsurf_sNames_ = string_type_1d_dual_view<LENGTHOFSPECNAME + 1>(
    do_not_init_tag("KMD::TCsurf_sNames_"), TCsurf_Nspec_);
  TCsurf_sMass_ = real_type_1d_dual_view(do_not_init_tag("KMD::TCsurf_sMass_"),
                                         TCsurf_Nspec_);

  /* Species' elemental composition */
  TCsurf_elemcount_ = ordinal_type_2d_dual_view(
    do_not_init_tag("KMD::elemCount"), TCsurf_Nspec_, nElem_);

  /* Species charges, no of tempfits, phase */
  TCsurf_sCharge_ = ordinal_type_1d_dual_view(
    do_not_init_tag("KMD::TCsurf_sCharge_"), TCsurf_Nspec_);
  TCsurf_sTfit_ = ordinal_type_1d_dual_view(
    do_not_init_tag("KMD::TCsurf_sTfit_"), TCsurf_Nspec_);
  TCsurf_sPhase_ = ordinal_type_1d_dual_view(
    do_not_init_tag("KMD::TCsurf_sPhase_"), TCsurf_Nspec_);

  /* Range of temperatures for thermo fits */
  TCsurf_Tlo_ =
    real_type_1d_dual_view(do_not_init_tag("KMD::TCsurf_Tlo_"), TCsurf_Nspec_);
  TCsurf_Tmi_ =
    real_type_1d_dual_view(do_not_init_tag("KMD::TCsurf_Tmi_"), TCsurf_Nspec_);
  TCsurf_Thi_ =
    real_type_1d_dual_view(do_not_init_tag("KMD::TCsurf_Thi_"), TCsurf_Nspec_);

  /* Polynomial coeffs for thermo fits */
  TCsurf_cppol_ = real_type_3d_dual_view(do_not_init_tag("KMD::TCsurf_cppol_"),
                                         TCsurf_Nspec_,
                                         TCsurf_nNASAinter_,
                                         (TCsurf_nCpCoef_ + 2));

  /* reaction info */
  if (TCsurf_Nreac_ > 0) {
    TCsurf_isRev_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::TCsurf_isRev_"), TCsurf_Nreac_);
    TCsurf_reacNrp_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::TCsurf_reacNrp_"), TCsurf_Nreac_);
    TCsurf_reacNreac_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::TCsurf_reacNreac_"), TCsurf_Nreac_);
    TCsurf_reacNprod_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::TCsurf_reacNprod_"), TCsurf_Nreac_);
    TCsurf_reacScoef_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::TCsurf_reacScoef_"), TCsurf_Nreac_);
    TCsurf_reacNuki_ =
      ordinal_type_2d_dual_view(do_not_init_tag("KMD::TCsurf_reacNuki_"),
                                TCsurf_Nreac_,
                                TCsurf_maxSpecInReac_);
    TCsurf_reacSidx_ =
      ordinal_type_2d_dual_view(do_not_init_tag("KMD::TCsurf_reacSidx_"),
                                TCsurf_Nreac_,
                                TCsurf_maxSpecInReac_);
    TCsurf_reacSsrf_ =
      ordinal_type_2d_dual_view(do_not_init_tag("KMD::TCsurf_reacSsrf_"),
                                TCsurf_Nreac_,
                                TCsurf_maxSpecInReac_);
    TCsurf_reacArhenFor_ = real_type_2d_dual_view(
      do_not_init_tag("KMD::TCsurf_reacArhenFor_"), TCsurf_Nreac_, 3);
    TCsurf_isDup_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::TCsurf_isDup_"), TCsurf_Nreac_);
    TCsurf_isStick_ = ordinal_type_1d_dual_view(
      do_not_init_tag("KMD::TCsurf_isStick_"), TCsurf_Nreac_);
    // stoichiometric matrix only gas species
    vski_ = ordinal_type_2d_dual_view(
      do_not_init_tag("KMD::stoichiometric_matrix_gas"), nSpec_, TCsurf_Nreac_);
    vsurfki_ = ordinal_type_2d_dual_view(
      do_not_init_tag("KMD::stoichiometric_matrix_surf"),
      TCsurf_Nspec_,
      TCsurf_Nreac_);
  }
}

void
KineticModelData::syncToDevice()
{
  /* Elements' name and weights */
  eNames_.sync_device();
  eMass_.sync_device();

  /* Species' name and weights */
  sNames_.sync_device();
  sMass_.sync_device();

  /* Species' elemental composition */
  elemCount_.sync_device();

  /* Species charges, no of tempfits, phase */
  sCharge_.sync_device();
  sTfit_.sync_device();
  sPhase_.sync_device();

  /* Range of temperatures for thermo fits */
  Tlo_.sync_device();
  Tmi_.sync_device();
  Thi_.sync_device();

  /* Polynomial coeffs for thermo fits */
  cppol_.sync_device();

  spec9t_.sync_device();
  spec9nrng_.sync_device();
  spec9trng_.sync_device();
  spec9coefs_.sync_device();

  /* Ionic species */
  sNion_.sync_device();

  /* reaction info */
  isRev_.sync_device();
  reacNrp_.sync_device();
  reacNreac_.sync_device();
  reacNprod_.sync_device();
  reacNuki_.sync_device();
  reacSidx_.sync_device();
  reacScoef_.sync_device();
  reacArhenFor_.sync_device();
  isDup_.sync_device();

  /* Reactions with reversible Arrhenius parameters given */
  reacRev_.sync_device();
  reacArhenRev_.sync_device();

  /* Pressure-dependent reactions */
  reacPfal_.sync_device();
  reacPtype_.sync_device();
  reacPlohi_.sync_device();
  reacPspec_.sync_device();
  reacPpar_.sync_device();

  /* Third-body reactions */
  reacTbdy_.sync_device();
  reacTbno_.sync_device();
  reac_to_Tbdy_index_.sync_device();
  specTbdIdx_.sync_device();
  specTbdEff_.sync_device();

  /* Reactions with real stoichiometric coefficients */
  reacRnu_.sync_device();
  reacRealNuki_.sync_device();

  /* Arbitrary reaction orders */
  reacAOrd_.sync_device();
  specAOidx_.sync_device();
  specAOval_.sync_device();

  ///
  /// KJ: the following does not support
  ///

  /* Landau-Teller reactions */
  /* reverse Landau-Teller reactions */
  /* radiation wavelength */
  /* JAN fits */
  /* FIT1 fits */
  /* Excitation reactions */
  /* Plasma Momentum transfer collision */
  /* Ion Momentum transfer collision */
  /* Species temperature dependency */

  /* Reactions with PLOG formulation */
  reacPlogIdx_.sync_device();
  reacPlogPno_.sync_device();
  reacPlogPars_.sync_device();

  /* sum(nu) for each reaction */
  sigNu_.sync_device();
  NuIJ_.sync_device();

  sigRealNu_.sync_device();
  RealNuIJ_.sync_device();
  kc_coeff_.sync_device();
}

void
KineticModelData::syncSurfToDevice()
{

  TCsurf_isStick_.sync_device();
  TCsurf_isDup_.sync_device();
  TCsurf_reacArhenFor_.sync_device();
  TCsurf_reacNuki_.sync_device();
  TCsurf_reacSidx_.sync_device();
  TCsurf_reacSsrf_.sync_device();
  TCsurf_reacNreac_.sync_device();
  TCsurf_reacNprod_.sync_device();
  TCsurf_reacScoef_.sync_device();
  TCsurf_reacNrp_.sync_device();
  TCsurf_isRev_.sync_device();
  TCsurf_cppol_.sync_device();
  TCsurf_Tlo_.sync_device();
  TCsurf_Tmi_.sync_device();
  TCsurf_Thi_.sync_device();
  TCsurf_sCharge_.sync_device();
  TCsurf_sTfit_.sync_device();
  TCsurf_sPhase_.sync_device();
  TCsurf_elemcount_.sync_device();
  TCsurf_sMass_.sync_device();
  TCsurf_sNames_.sync_device();

  vski_.sync_device();
  vsurfki_.sync_device();
}

int
KineticModelData::initChem()
{
#define DASHLINE(file)                                                         \
  fprintf(file,                                                                \
          "------------------------------------------------------------"       \
          "-------------\n")
  double reacbalance;
  char charvar4[4];

  FILE *chemfile, *echofile, *errfile;

  /* zero-out variables */
  isInit_ = 0;
  Runiv_ = 0.0;
  Rcal_ = 0.0;
  Rcgs_ = 0.0;

  /* integers */
  maxSpecInReac_ = maxTbInReac_ = nNASAinter_ = nCpCoef_ = nArhPar_ = nLtPar_ =
    0;
  nFallPar_ = nJanPar_ = maxOrdPar_ = nFit1Par_ = 0;
  nElem_ = nSpec_ = nReac_ = nRevReac_ = nFallReac_ = nPlogReac_ = nThbReac_ =
    0;
  nRealNuReac_ = nOrdReac_ = 0;
  electrIndx_ = nIonEspec_ = nNASA9coef_ = 0;

  int nLtReac_ = 0, nRltReac_ = 0, nHvReac_ = 0, nIonSpec_ = 0, nJanReac_ = 0,
      nFit1Reac_ = 0;
  int nExciReac_ = 0, nMomeReac_ = 0, nXsmiReac_ = 0, nTdepReac_ = 0;

  /* work variables */
  isInit_ = 0;

  /* Retrieve things from kmod.list */
  chemfile = fopen("kmod.list", "r");
  echofile = fopen("kmod.echo", "w");
  errfile = fopen("kmod.err", "w");

  fscanf(chemfile, "%s", charvar4);

  if (strcmp(charvar4, "ERRO") == 0) {
    printf("kmod.list : Error when interpreting kinetic model  !!!");
    exit(1);
  }
  #ifdef TCHEM_ENABLE_VERBOSE
    printf("kmod.list : %s\n", charvar4);
  #endif

  fscanf(chemfile, "%d", &maxSpecInReac_);
  fscanf(chemfile, "%d", &maxTbInReac_);
  fscanf(chemfile, "%d", &nNASAinter_);
  fscanf(chemfile, "%d", &nCpCoef_);
  fscanf(chemfile, "%d", &nArhPar_);
  fscanf(chemfile, "%d", &nLtPar_);
  fscanf(chemfile, "%d", &nFallPar_);
  fscanf(chemfile, "%d", &nJanPar_);
  fscanf(chemfile, "%d", &maxOrdPar_);
  fscanf(chemfile, "%d", &nFit1Par_);

  fscanf(chemfile, "%d", &nElem_);
  fscanf(chemfile, "%d", &nSpec_);
  fscanf(chemfile, "%d", &nReac_);
  fscanf(chemfile, "%d", &nRevReac_);
  fscanf(chemfile, "%d", &nFallReac_);
  fscanf(chemfile, "%d", &nPlogReac_);
  fscanf(chemfile, "%d", &nThbReac_);
  fscanf(chemfile, "%d", &nLtReac_);
  fscanf(chemfile, "%d", &nRltReac_);
  fscanf(chemfile, "%d", &nHvReac_);
  fscanf(chemfile, "%d", &nIonSpec_);
  fscanf(chemfile, "%d", &nJanReac_);
  fscanf(chemfile, "%d", &nFit1Reac_);
  fscanf(chemfile, "%d", &nExciReac_);
  fscanf(chemfile, "%d", &nMomeReac_);
  fscanf(chemfile, "%d", &nXsmiReac_);
  fscanf(chemfile, "%d", &nTdepReac_);
  fscanf(chemfile, "%d", &nRealNuReac_);
  fscanf(chemfile, "%d", &nOrdReac_);
  fscanf(chemfile, "%d", &electrIndx_);
  fscanf(chemfile, "%d", &nIonEspec_);
  fscanf(chemfile, "%d", &nNASA9coef_);

  fprintf(
    echofile,
    "kmod.list : Max # of species in a reaction                    : %d\n",
    maxSpecInReac_);
  fprintf(
    echofile,
    "kmod.list : Max # of third-body efficiencies in a reaction    : %d\n",
    maxTbInReac_);
  fprintf(
    echofile,
    "kmod.list : # of temperature regions for thermo fits          : %d\n",
    nNASAinter_);
  fprintf(
    echofile,
    "kmod.list : # of polynomial coefficients for thermo fits      : %d\n",
    nCpCoef_);
  fprintf(
    echofile,
    "kmod.list : # of species with 9 coefficients thermo props     : %d\n",
    nNASA9coef_);
  fprintf(
    echofile,
    "kmod.list : # of Arrhenius parameters                         : %d\n",
    nArhPar_);
  fprintf(
    echofile,
    "kmod.list : # of parameters for Landau-Teller reactions       : %d\n",
    nLtPar_);
  fprintf(
    echofile,
    "kmod.list : # of parameters for pressure-dependent reactions  : %d\n",
    nFallPar_);
  fprintf(
    echofile,
    "kmod.list : # of parameters for Jannev-Langer fits (JAN)      : %d\n",
    nJanPar_);
  fprintf(
    echofile,
    "kmod.list : # of parameters for arbitrary order reactions     : %d\n",
    maxOrdPar_);
  fprintf(
    echofile,
    "kmod.list : # of parameters for FIT1 fits                     : %d\n",
    nFit1Par_);
  fprintf(
    echofile,
    "kmod.list : # of elements                                     : %d\n",
    nElem_);
  fprintf(
    echofile,
    "kmod.list : # of species                                      : %d\n",
    nSpec_);
  fprintf(
    echofile,
    "kmod.list : # of reactions                                    : %d\n",
    nReac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with REV given                     : %d\n",
    nRevReac_);
  fprintf(
    echofile,
    "kmod.list : # of pressure-dependent reactions                 : %d\n",
    nFallReac_);
  fprintf(
    echofile,
    "kmod.list : # of PLOG reactions                               : %d\n",
    nPlogReac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions using third-body efficiencies      : %d\n",
    nThbReac_);
  fprintf(
    echofile,
    "kmod.list : # of Landau-Teller reactions                      : %d\n",
    nLtReac_);
  fprintf(
    echofile,
    "kmod.list : # of Landau-Teller reactions with RLT given       : %d\n",
    nRltReac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with HV                            : %d\n",
    nHvReac_);
  fprintf(
    echofile,
    "kmod.list : # of ion species                                  : %d\n",
    nIonSpec_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with JAN fits                      : %d\n",
    nJanReac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with FIT1 fits                     : %d\n",
    nFit1Reac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with EXCI                          : %d\n",
    nExciReac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with MOME                          : %d\n",
    nMomeReac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with XSMI                          : %d\n",
    nXsmiReac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with TDEP                          : %d\n",
    nTdepReac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with non-int stoichiometric coeffs : %d\n",
    nRealNuReac_);
  fprintf(
    echofile,
    "kmod.list : # of reactions with arbitrary order               : %d\n",
    nOrdReac_);
  fprintf(
    echofile,
    "kmod.list : Index of the electron species                     : %d\n",
    electrIndx_);
  fprintf(
    echofile,
    "kmod.list : # of ion species excluding the electron species   : %d\n",
    nIonEspec_);
  DASHLINE(echofile);
  fflush(echofile);

  fscanf(chemfile, "%lf", &reacbalance);
  fprintf(
    echofile,
    "kmod.list : Tolerance for reaction balance                    : %e\n",
    reacbalance);
  DASHLINE(echofile);
  fflush(echofile);

  ///
  /// Allocate views
  ///
  allocateViews(errfile);

  /// Host views to read data from a file
  auto eNamesHost = eNames_.view_host();
  auto eMassHost = eMass_.view_host();
  auto sNamesHost = sNames_.view_host();
  auto sMassHost = sMass_.view_host();
  auto elemCountHost = elemCount_.view_host();
  auto sChargeHost = sCharge_.view_host();
  auto sTfitHost = sTfit_.view_host();
  auto sPhaseHost = sPhase_.view_host();
  auto TloHost = Tlo_.view_host();
  auto TmiHost = Tmi_.view_host();
  auto ThiHost = Thi_.view_host();
  auto cppolHost = cppol_.view_host();
  auto spec9tHost = spec9t_.view_host();
  auto spec9nrngHost = spec9nrng_.view_host();
  auto spec9trngHost = spec9trng_.view_host();
  auto spec9coefsHost = spec9coefs_.view_host();
  auto sNionHost = sNion_.view_host();

  auto isRevHost = isRev_.view_host();
  auto reacNrpHost = reacNrp_.view_host();
  auto reacNreacHost = reacNreac_.view_host();
  auto reacNprodHost = reacNprod_.view_host();
  auto reacNukiHost = reacNuki_.view_host();
  auto reacSidxHost = reacSidx_.view_host();
  auto reacScoefHost = reacScoef_.view_host();
  auto reacArhenForHost = reacArhenFor_.view_host();
  auto isDupHost = isDup_.view_host();
  auto reacRevHost = reacRev_.view_host();
  auto reacArhenRevHost = reacArhenRev_.view_host();

  auto reacPfalHost = reacPfal_.view_host();
  auto reacPtypeHost = reacPtype_.view_host();
  auto reacPlohiHost = reacPlohi_.view_host();
  auto reacPspecHost = reacPspec_.view_host();
  auto reacPparHost = reacPpar_.view_host();

  auto reacTbdyHost = reacTbdy_.view_host();
  auto reacTbnoHost = reacTbno_.view_host();
  auto reac_to_Tbdy_indexHost = reac_to_Tbdy_index_.view_host();
  auto specTbdIdxHost = specTbdIdx_.view_host();
  auto specTbdEffHost = specTbdEff_.view_host();

  auto reacRnuHost = reacRnu_.view_host();
  auto reacRealNukiHost = reacRealNuki_.view_host();

  auto reacAOrdHost = reacAOrd_.view_host();
  auto specAOidxHost = specAOidx_.view_host();
  auto specAOvalHost = specAOval_.view_host();

  auto reacPlogIdxHost = reacPlogIdx_.view_host();
  auto reacPlogPnoHost = reacPlogPno_.view_host();

  auto sigNuHost = sigNu_.view_host();

  auto sigRealNuHost = sigRealNu_.view_host();
  auto NuIJHost = NuIJ_.view_host();
  auto RealNuIJHost = RealNuIJ_.view_host();
  auto kc_coeffHost = kc_coeff_.view_host();

  {
    // Elements
    for (int i = 0; i < nElem_; i++) {
      char elemNm[LENGTHOFELEMNAME] = {};
      fscanf(chemfile, "%s", elemNm);
      strncat(&eNamesHost(i, 0), elemNm, LENGTHOFELEMNAME);
    }

    for (int i = 0; i < nElem_; i++)
      fscanf(chemfile, "%lf", &(eMassHost(i)));

    fprintf(echofile, "No. \t Element \t Mass\n");
    for (int i = 0; i < nElem_; i++)
      fprintf(echofile,
              "%-3d\t%-4s\t%f10.7\n",
              i + 1,
              &eNamesHost(i, 0),
              eMassHost(i));
    DASHLINE(echofile);
    fflush(echofile);

    // Species
    for (int i = 0; i < nSpec_; i++) {
      char specNm[LENGTHOFSPECNAME];
      fscanf(chemfile, "%s", specNm);
      strncat(&sNamesHost(i, 0), specNm, LENGTHOFSPECNAME);
    }

    for (int i = 0; i < nSpec_; i++)
      fscanf(chemfile, "%lf", &(sMassHost(i)));

    fprintf(echofile, "No. \t Species \t Mass\n");
    for (int i = 0; i < nSpec_; i++)
      fprintf(echofile,
              "%-3d\t%-32s\t%12.7f\n",
              i + 1,
              &sNamesHost(i, 0),
              sMassHost(i));
    DASHLINE(echofile);
    fflush(echofile);

    // ElemCounts
    for (int i = 0; i < nSpec_; i++)
      for (int j = 0; j < nElem_; j++)
        fscanf(chemfile, "%d", &elemCountHost(i, j));

    fprintf(echofile, "Elemental composition of species\n");
    fprintf(echofile, "No. \t Species \t\t Element\n\t\t\t\t");
    for (int i = 0; i < nElem_; i++)
      fprintf(echofile, "%s\t", &eNamesHost(i, 0));
    fprintf(echofile, "\n");

    for (int i = 0; i < nSpec_; i++) {
      fprintf(echofile, "%-3d\t%-32s", i + 1, &sNamesHost(i, 0));
      for (int j = 0; j < nElem_; j++)
        fprintf(echofile, "%-3d\t", elemCountHost(i, j));
      fprintf(echofile, "\n");
    }

    /* Species charges, no of tempfits, phase */
    for (int i = 0; i < nSpec_; i++)
      fscanf(chemfile, "%d", &(sChargeHost(i)));
    for (int i = 0; i < nSpec_; i++)
      fscanf(chemfile, "%d", &(sTfitHost(i)));
    for (int i = 0; i < nSpec_; i++)
      fscanf(chemfile, "%d", &(sPhaseHost(i)));

    /* Range of temperatures for thermo fits */
    TthrmMin_ = 1.e-5;
    TthrmMax_ = 1.e+5;
    for (int i = 0; i < nSpec_; i++) {
      fscanf(chemfile, "%lf", &(TloHost(i)));
      fscanf(chemfile, "%lf", &(TmiHost(i)));
      fscanf(chemfile, "%lf", &(ThiHost(i)));
      TthrmMin_ = std::min(TthrmMin_, TloHost(i));
      TthrmMax_ = std::max(TthrmMax_, ThiHost(i));
    }

    fprintf(echofile, "Range of temperature for thermodynamic fits\n");
    fprintf(echofile, "No. \t Species \t\t Tlow \tTmid \tThigh\n");
    for (int i = 0; i < nSpec_; i++) {
      fprintf(echofile,
              "%-3d\t%-32s %12.4f\t%12.4f\t%12.4f\n",
              i + 1,
              &sNamesHost(i, 0),
              TloHost(i),
              TmiHost(i),
              ThiHost(i));
    }
    DASHLINE(echofile);

    /* Polynomial coeffs for thermo fits */
    for (int i = 0; i < nSpec_; i++)
      for (int j = 0; j < nNASAinter_; j++)
        for (int k = 0; k < nCpCoef_ + 2; k++)
          fscanf(chemfile, "%lf", &cppolHost(i, j, k));

    fprintf(echofile, "List of coefficients for thermodynamic fits\n");
    for (int i = 0; i < nSpec_; i++) {
      const auto ptr = &cppolHost(i, 0, 0);
      fprintf(echofile,
              "%-4d %-32s\n %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e\n",
              i + 1,
              &sNamesHost(i, 0),
              ptr[0],
              ptr[1],
              ptr[2],
              ptr[3],
              ptr[4],
              ptr[5],
              ptr[6]);
      fprintf(echofile,
              " %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e\n",
              ptr[7],
              ptr[8],
              ptr[9],
              ptr[10],
              ptr[11],
              ptr[12],
              ptr[13]);
    }
    DASHLINE(echofile);

    /* Polynomial coeffs for 9-term thermo fits */
    if (nNASA9coef_ > 0) {
      int nNASA9coef_again;
      fscanf(chemfile, "%d", &nNASA9coef_again);
      assert(nNASA9coef_again == nNASA9coef_);

      for (int i = 0; i < nNASA9coef_; i++) {
        fscanf(chemfile, "%d", &spec9tHost(i));
        spec9tHost(i) -= 1;
        fscanf(chemfile, "%d", &spec9nrngHost(i));

        for (int j = 0; j < spec9nrngHost(i); j++) {
          fscanf(chemfile, "%lf", &spec9trngHost(i, j, 0));
          fscanf(chemfile, "%lf", &spec9trngHost(i, j, 1));
          for (int k = 0; k < 9; k++)
            fscanf(chemfile, "%lf", &spec9coefsHost(i, j, k));
        } /* done loop over temperature ranges */
      }   /* done loop over species */
    }     /* done if for species with 9-coefficients thermo props */

    /* Ionic species */
    if (nIonEspec_ > 0) {
      int nIonEspec_again;
      fscanf(chemfile, "%d", &nIonEspec_again);
      assert(nIonEspec_again == nIonEspec_);

      for (int i = 0; i < nIonEspec_; i++) {
        fscanf(chemfile, "%d", &(sNionHost(i)));
        sNionHost(i) -= 1;
      }
    }

    /* reaction info */
    if (nReac_ > 0) {

      /* no of reactants and products */
      for (int i = 0; i < nReac_; i++) {
        int itemp;
        fscanf(chemfile, "%d", &itemp);
        isRevHost(i) = (itemp >= 0);
        reacNrpHost(i) = abs(itemp);
      }

      /* no of reactants only */
      for (int i = 0; i < nReac_; i++)
        fscanf(chemfile, "%d", &(reacNreacHost(i)));

      /* no of products */
      for (int i = 0; i < nReac_; i++)
        reacNprodHost(i) = reacNrpHost(i) - reacNreacHost(i);

      /* Stoichiometric coefficients */
      for (int i = 0; i < nReac_; i++) {

        int nusumk = 0;
        /* by default reaction has integer stoichiometric coefficients */
        reacScoefHost(i) = -1;

        nusumk = 0;
        for (int j = 0; j < maxSpecInReac_; j++) {
          fscanf(chemfile, "%d", &(reacNukiHost(i, j)));
          fscanf(chemfile, "%d", &(reacSidxHost(i, j)));
          reacSidxHost(i, j) -= 1;
          nusumk += reacNukiHost(i, j);
          /// KJ: this is not necessary
          /// reacNukiDblHost(i,j) = reacNukiHost(i,j);
        }
        int itemp;
        fscanf(chemfile, "%d", &itemp);
        assert(itemp == nusumk);
      }

      /* Arrhenius parameters */
      for (int i = 0; i < nReac_; i++)
        for (int j = 0; j < 3; ++j)
          fscanf(chemfile, "%lf", &(reacArhenForHost(i, j)));

      for (int i = 0; i < nReac_; i++) {
        fscanf(chemfile, "%d", &(isDupHost(i)));
      }

      fprintf(echofile, "Reaction data : species and Arrhenius pars\n");
      for (int i = 0; i < nReac_; i++) {
        fprintf(echofile,
                "%-5d\t%1d\t%2d\t%2d | ",
                i + 1,
                isRevHost(i),
                reacNreacHost(i),
                reacNprodHost(i));
        for (int j = 0; j < reacNreacHost(i); j++)
          fprintf(echofile,
                  "%d*%s | ",
                  reacNukiHost(i, j),
                  &sNamesHost(reacSidxHost(i, j), 0));

        /// KJ why do we do this way ?
        const int joff = maxSpecInReac_ / 2;
        for (int j = 0; j < reacNprodHost(i); j++)
          fprintf(echofile,
                  "%d*%s | ",
                  reacNukiHost(i, j + joff),
                  &sNamesHost(reacSidxHost(i, j + joff), 0));

        fprintf(echofile,
                "%16.8e\t%16.8e\t%16.8e",
                reacArhenForHost(i, 0),
                reacArhenForHost(i, 1),
                reacArhenForHost(i, 2));
        if (isDupHost(i) == 1)
          fprintf(echofile, "  DUPLICATE\n");
        else
          fprintf(echofile, "\n");
      }
      DASHLINE(echofile);
    }
    if (verboseEnabled)
      printf("KineticModelData::initChem() : Done reading reaction data\n");

    /* Reactions with reversible Arrhenius parameters given */
    if (nRevReac_ > 0) {
      /* No. of such reactions */
      int nRevReac_again;
      fscanf(chemfile, "%d", &nRevReac_again);
      assert(nRevReac_again == nRevReac_);

      /* Their indices */
      for (int i = 0; i < nRevReac_; i++) {
        fscanf(chemfile, "%d", &(reacRevHost(i)));
        reacRevHost(i) -= 1;
      }

      /* reverse Arrhenius parameters */
      for (int i = 0; i < nRevReac_; i++)
        for (int j = 0; j < 3; j++)
          fscanf(chemfile, "%lf", &(reacArhenRevHost(i, j)));

      fprintf(echofile,
              "Reaction data : reverse Arrhenius pars for %d reactions :\n",
              nRevReac_);
      for (int i = 0; i < nRevReac_; i++) {
        fprintf(echofile,
                "%-4d\t%16.8e\t%16.8e\t%16.8e\n",
                reacRevHost(i) + 1,
                reacArhenRevHost(i, 0),
                reacArhenRevHost(i, 1),
                reacArhenRevHost(i, 2));
      }
      DASHLINE(echofile);

    } /* Done if nRevReac_ > 0 */
    if (verboseEnabled)
      printf(
        "KineticModelData::initChem() : Done reading reverse reaction data\n");

    /* Pressure-dependent reactions */
    if (nFallReac_ > 0) {
      int nFallReac_again, nFallPar_again;
      fscanf(chemfile, "%d", &nFallReac_again);
      fscanf(chemfile, "%d", &nFallPar_again);
      assert(nFallReac_again == nFallReac_);
      assert(nFallPar_again == nFallPar_);

      for (int i = 0; i < nFallReac_; i++) {
        fscanf(chemfile, "%d", &(reacPfalHost(i)));
        reacPfalHost(i) -= 1;
        fscanf(chemfile, "%d", &(reacPtypeHost(i)));
        fscanf(chemfile, "%d", &(reacPlohiHost(i)));
        fscanf(chemfile, "%d", &(reacPspecHost(i)));
        reacPspecHost(i) -= 1;
      }

      for (int i = 0; i < nFallReac_; i++)
        for (int j = 0; j < nFallPar_; j++)
          fscanf(chemfile, "%lf", &(reacPparHost(i, j)));

      fprintf(echofile,
              "Reaction data : Pressure dependencies for %d reactions :\n",
              nFallReac_);
      for (int i = 0; i < nFallReac_; i++) {
        fprintf(echofile, "%-4d\t", reacPfalHost(i) + 1);

        if (reacPtypeHost(i) == 1)
          fprintf(echofile, "Lind \t");
        if (reacPtypeHost(i) == 2)
          fprintf(echofile, "SRI  \t");
        if (reacPtypeHost(i) == 3)
          fprintf(echofile, "Troe3\t");
        if (reacPtypeHost(i) == 4)
          fprintf(echofile, "Troe4\t");
        if (reacPtypeHost(i) == 6)
          fprintf(echofile, "Cheb \t");

        if (reacPlohiHost(i) == 0)
          fprintf(echofile, "Low  \t");
        if (reacPlohiHost(i) == 1)
          fprintf(echofile, "High \t");

        if (reacPspecHost(i) < 0)
          fprintf(echofile, "Mixture");
        if (reacPspecHost(i) >= 0)
          fprintf(echofile, "%s\n", &sNamesHost(reacPspecHost(i), 0));
      }
      DASHLINE(echofile);

    } /* Done if fall-off reactions */
    if (verboseEnabled)
      printf("KineticModelData::initChem() : Done reading pressure-dependent "
             "reaction data\n");

    /* Third-body reactions */
    if (nThbReac_ > 0) {
      int nThbReac_again, maxTbInReac_again;
      fscanf(chemfile, "%d", &nThbReac_again);
      fscanf(chemfile, "%d", &maxTbInReac_again);
      assert(nThbReac_again == nThbReac_);
      assert(maxTbInReac_again == maxTbInReac_);

      for (int i = 0; i < nThbReac_; i++) {
        fscanf(chemfile, "%d", &(reacTbdyHost(i)));
        reacTbdyHost(i) -= 1;
        fscanf(chemfile, "%d", &(reacTbnoHost(i)));
      }

      int itbdy = 0;
      Kokkos::deep_copy(reac_to_Tbdy_indexHost, -1);
      for (int ireac = 0; ireac < nReac_; ireac++)
        if (itbdy < nThbReac_)
          if (ireac == reacTbdyHost(itbdy))
            reac_to_Tbdy_indexHost(ireac) = itbdy++;

      for (int i = 0; i < nThbReac_; i++)
        for (int j = 0; j < maxTbInReac_; j++) {
          int itemp;
          fscanf(chemfile, "%d", &itemp);
          specTbdIdxHost(i, j) = itemp - 1;
        }
      for (int i = 0; i < nThbReac_; i++)
        for (int j = 0; j < maxTbInReac_; j++)
          fscanf(chemfile, "%lf", &(specTbdEffHost(i, j)));

      fprintf(echofile, "Reaction data : Third body efficiencies :\n");
      for (int i = 0; i < nThbReac_; i++) {
        fprintf(echofile, "%-4d\t", reacTbdyHost(i) + 1);
        for (int j = 0; j < reacTbnoHost(i); j++)
          fprintf(echofile,
                  "%s->%5.2f, ",
                  &sNamesHost(specTbdIdxHost(i, j), 0),
                  specTbdEffHost(i, j));
        fprintf(echofile, "\n");
      }
      DASHLINE(echofile);
    }
    if (verboseEnabled)
      printf("KineticModelData::initChem() : Done reading third-body data\n");

    /* Reactions with real stoichiometric coefficients */
    if (nRealNuReac_ > 0) {
      int nRealNuReac_again;
      fscanf(chemfile, "%d", &nRealNuReac_again);
      assert(nRealNuReac_again == nRealNuReac_);

      for (int i = 0; i < nRealNuReac_; i++) {
        fscanf(chemfile, "%d", &(reacRnuHost(i)));
        reacRnuHost(i) -= 1;
        reacScoefHost(reacRnuHost(i)) = i;
      }

      /* Real stoichiometric coefficients */
      for (int i = 0; i < nRealNuReac_; i++) {
        double dtemp;
        double nusumk = 0.0;
        for (int j = 0; j < maxSpecInReac_; j++) {
          fscanf(chemfile, "%lf", &(reacRealNukiHost(i, j)));
          nusumk += reacRealNukiHost(i, j);
        }
        fscanf(chemfile, "%lf", &dtemp);
        assert(fabs(dtemp - nusumk) < reacbalance);
      }
    }

    /* Arbitrary reaction orders */
    if (nOrdReac_ > 0) {
      int nOrdReac_again, maxOrdPar_again;
      fscanf(chemfile, "%d", &nOrdReac_again);
      fscanf(chemfile, "%d", &maxOrdPar_again);
      assert(nOrdReac_again == nOrdReac_);
      assert(maxOrdPar_again == maxOrdPar_);

      for (int i = 0; i < nOrdReac_; i++) {
        fscanf(chemfile, "%d", &(reacAOrdHost(i)));
        reacAOrdHost(i) -= 1;
      }

      for (int i = 0; i < nOrdReac_; i++)
        for (int j = 0; j < maxOrdPar_; j++)
          fscanf(chemfile, "%d", &(specAOidxHost(i, j)));
      for (int i = 0; i < nOrdReac_; i++)
        for (int j = 0; j < maxOrdPar_; j++)
          fscanf(chemfile, "%lf", &(specAOvalHost(i, j)));

      if (verboseEnabled) {
        printf("Reading arbitrary orders: %d, %d \n", nOrdReac_, maxOrdPar_);
        for (int i = 0; i < nOrdReac_; i++) {
          for (int j = 0; j < maxOrdPar_; j++)
            printf("%d, ", specAOidxHost(i, j));
          printf("\n");
        }
        for (int i = 0; i < nOrdReac_; i++) {
          for (int j = 0; j < maxOrdPar_; j++)
            printf("%e, ", specAOvalHost(i, j));
          printf("\n");
        }
      }
    }

    /* Landau-Teller reactions */
    /* reverse Landau-Teller reactions */

    /* radiation wavelength */
    // if ( nHvReac_ > 0 ) {
    //   for ( int i=0 ; i < nHvReac_ ; i++ ) {
    //     fscanf( chemfile,"%d", &(reacHvIdxHost(i))) ; reacHvIdxHost(i) -= 1 ;
    //   }
    //   for ( int i=0 ; i < nHvReac_ ; i++ )
    //     fscanf( chemfile,"%lf", &(reacHvParHost(i))) ;
    // }

    /* JAN fits */
    /* FIT1 fits */
    /* Excitation reactions */
    /* Plasma Momentum transfer collision */
    /* Ion Momentum transfer collision */
    /* Species temperature dependency */

    /* Reactions with PLOG formulation */
    if (nPlogReac_ > 0) {
      int nPlogReac_again;
      fscanf(chemfile, "%d", &nPlogReac_again);
      assert(nPlogReac_again == nPlogReac_);

      /* Their indices and no of PLOG intervals */
      reacPlogPnoHost(0) = 0;
      for (int i = 0; i < nPlogReac_; i++) {
        fscanf(chemfile, "%d", &(reacPlogIdxHost(i)));
        reacPlogIdxHost(i) -= 1;
        fscanf(chemfile, "%d", &(reacPlogPnoHost(i + 1)));
        reacPlogPnoHost(i + 1) += reacPlogPnoHost(i);
      }

      /* Plog parameters */
      reacPlogPars_ = real_type_2d_dual_view(
        do_not_init_tag("KMD::reacPlogPars"), reacPlogPnoHost(nPlogReac_), 4);
      auto reacPlogParsHost = reacPlogPars_.view_host();
      for (int i = 0; i < reacPlogPnoHost(nPlogReac_); i++) {
        fscanf(chemfile, "%lf", &(reacPlogParsHost(i, 0)));
        reacPlogParsHost(i, 0) = log(reacPlogParsHost(i, 0));
        fscanf(chemfile, "%lf", &(reacPlogParsHost(i, 1)));
//        reacPlogParsHost(i, 1) = log(reacPlogParsHost(i, 1));
        fscanf(chemfile, "%lf", &(reacPlogParsHost(i, 2)));
        fscanf(chemfile, "%lf", &(reacPlogParsHost(i, 3)));
      }

    } /* Done if nPlogReac > 0 */

    fclose(chemfile);
    fclose(errfile);
    fclose(echofile);

    /* universal gas constant */
    Runiv_ = RUNIV * 1.0e3; // j/kmol/K
    Rcal_ = Runiv_ / (CALJO * 1.0e3);
    Rcgs_ = Runiv_ * 1.e4;

    /* populate some arrays */

    /* sum(nu) for each reaction */
    for (int i = 0; i < nReac_; i++) {
      /* reactants */
      for (int j = 0; j < reacNreacHost(i); j++)
        sigNuHost(i) += reacNukiHost(i, j);
      /* products */
      const int joff = maxSpecInReac_ / 2;
      for (int j = 0; j < reacNprodHost(i); j++)
        sigNuHost[i] += reacNukiHost(i, j + joff);
    }
    if (nRealNuReac_ > 0) {
      for (int ir = 0; ir < nRealNuReac_; ir++) {
        int irR = reacRnuHost(ir);
        /* reactants */
        for (int j = 0; j < reacNreacHost(irR); j++)
          sigRealNuHost(ir) += reacRealNukiHost(ir, j);
        /* products */
        const int joff = reacNreacHost(irR);
        for (int j = 0; j < reacNprodHost(irR); j++)
          sigRealNuHost(ir) += reacRealNukiHost(ir, j + joff);
      }
    }

    /* NuIJ=NuII''-NuIJ' */
    for (int j = 0; j < nReac_; j++) {
      /* reactants */
      for (int i = 0; i < reacNreacHost(j); i++) {

        int kspec = reacSidxHost(j, i);

        NuIJHost(j, kspec) += reacNukiHost(j, i);
      }
      /* products */
      const int ioff = maxSpecInReac_ / 2;
      for (int i = 0; i < reacNprodHost(j); i++) {
        int kspec = reacSidxHost(j, i + ioff);
        NuIJHost(j, kspec) += reacNukiHost(j, i + ioff);
      }
    }

    if (nRealNuReac_ > 0) {
      for (int j = 0; j < nRealNuReac_; j++) {
        /* reactants */
        int jR = reacRnuHost(j);
        for (int i = 0; i < reacNreacHost(jR); i++) {
          int kspec = reacSidxHost(jR, i);
          RealNuIJHost(j, kspec) += reacRealNukiHost(j, i);
        }
        /* products */
        // indxR = j *TC_maxSpecInReac_+TC_maxSpecInReac_/2 ;  // Bugfix Cosmin
        // 2016/12/28
        const int ioff_j = reacNreacHost(jR);
        const int ioff_jR = maxSpecInReac_ / 2;
        for (int i = 0; i < reacNprodHost(jR); i++) {
          int kspec = reacSidxHost(jR, i + ioff_jR);
          RealNuIJHost(j, kspec) += reacRealNukiHost(j, i + ioff_j);
        }
      }
    }

    /* Store coefficients for kc */
    for (int i = 0; i < nReac_; i++) {
      int ir = reacScoefHost(i);
      kc_coeffHost(i) =
        std::pow(ATMPA * real_type(10) / Rcgs_,
                 ir == -1 ? real_type(sigNuHost(i)) : sigRealNuHost(ir));
    }

    /* done */
    isInit_ = 1;

    /* Reactions with real stoichiometric coefficients */
    if (nRealNuReac_ > 0) {

      for (int i = 0; i < nRealNuReac_; i++) {
        fprintf(echofile, "%d\n", reacRnuHost(i));
      }

      /* Real stoichiometric coefficients */
      for (int i = 0; i < nRealNuReac_; i++) {
        for (int j = 0; j < maxSpecInReac_; j++) {
          fprintf(echofile, "%f ", reacRealNukiHost(i, j));
        }
        fprintf(echofile, "\n");
      }
    }

    // count elements that are present in gas species
    NumberofElementsGas_ = 0;
    for (int i = 0; i < elemCountHost.extent(1); i++) {   // loop over elements
      for (int j = 0; j < elemCountHost.extent(0); j++) { // loop over species
        if (elemCountHost(j, i) != 0) {
          NumberofElementsGas_++;
          break;
        }
      }
    }

    /// Raise modify flags for all modified dual views
    eNames_.modify_host();
    eMass_.modify_host();
    sNames_.modify_host();
    sMass_.modify_host();
    elemCount_.modify_host();
    sCharge_.modify_host();
    sTfit_.modify_host();
    sPhase_.modify_host();
    Tlo_.modify_host();
    Tmi_.modify_host();
    Thi_.modify_host();
    cppol_.modify_host();
    spec9t_.modify_host();
    spec9nrng_.modify_host();
    spec9trng_.modify_host();
    spec9coefs_.modify_host();
    isRev_.modify_host();
    reacNrp_.modify_host();
    reacNreac_.modify_host();
    reacNprod_.modify_host();
    reacNuki_.modify_host();
    reacSidx_.modify_host();
    reacScoef_.modify_host();
    reacArhenFor_.modify_host();
    isDup_.modify_host();
    reacRev_.modify_host();
    reacArhenRev_.modify_host();

    sNion_.modify_host();

    reacPfal_.modify_host();
    reacPtype_.modify_host();
    reacPlohi_.modify_host();
    reacPspec_.modify_host();
    reacPpar_.modify_host();

    reacTbdy_.modify_host();
    reacTbno_.modify_host();
    reac_to_Tbdy_index_.modify_host();
    specTbdIdx_.modify_host();
    specTbdEff_.modify_host();

    reacRnu_.modify_host();
    reacRealNuki_.modify_host();

    reacAOrd_.modify_host();
    specAOidx_.modify_host();
    specAOval_.modify_host();

    reacPlogIdx_.modify_host();
    reacPlogPno_.modify_host();

    sigNu_.modify_host();

    sigRealNu_.modify_host();
    NuIJ_.modify_host();
    RealNuIJ_.modify_host();
    kc_coeff_.modify_host();

    /// Sync to device
    syncToDevice();

    return (0);
  }
}

int
KineticModelData::initChemSurf()
{

  char charvar4[4];
  double reacbalance;

  TCsurf_maxSpecInReac_ = 0;
  TCsurf_nNASAinter_ = 0;
  TCsurf_nCpCoef_ = 0;
  TCsurf_nArhPar_ = 0;
  TCsurf_Nspec_ = 0;
  TCsurf_Nreac_ = 0;

  FILE *chemfile, *echofile, *errfile;
  /* Retrieve things from kmod.list */
  chemfile = fopen("kmodSurf.list", "r");
  echofile = fopen("kmodSurf.echo", "w");
  errfile = fopen("kmodSurf.err", "w");

  fscanf(chemfile, "%s", charvar4);
  /* printf("%s\n",charvar4) ; */
  if (strcmp(charvar4, "ERRO") == 0) {
    printf("kmodSurf.list : Error when interpreting kinetic model  !!!");
    exit(1);
  }

  printf("kmod.list : %s\n", charvar4);

  fscanf(chemfile, "%d", &TCsurf_maxSpecInReac_);
  fscanf(chemfile, "%d", &TCsurf_nNASAinter_);
  fscanf(chemfile, "%d", &TCsurf_nCpCoef_);
  fscanf(chemfile, "%d", &TCsurf_nArhPar_);
  fscanf(chemfile, "%d", &TCsurf_Nspec_);
  fscanf(chemfile, "%d", &TCsurf_Nreac_);

  fprintf(
    echofile,
    "kmodSurf.list : Max # of species in a reaction                    : %d\n",
    TCsurf_maxSpecInReac_);
  fprintf(
    echofile,
    "kmodSurf.list : # of temperature regions for thermo fits          : %d\n",
    TCsurf_nNASAinter_);
  fprintf(
    echofile,
    "kmodSurf.list : # of polynomial coefficients for thermo fits      : %d\n",
    TCsurf_nCpCoef_);
  fprintf(
    echofile,
    "kmodSurf.list : # of Arrhenius parameters                         : %d\n",
    TCsurf_nArhPar_);
  fprintf(
    echofile,
    "kmodSurf.list : # of species                                      : %d\n",
    TCsurf_Nspec_);
  fprintf(
    echofile,
    "kmodSurf.list : # of reactions                                    : %d\n",
    TCsurf_Nreac_);
  fprintf(echofile,
          "----------------------------------------------------------"
          "---------------\n");
  fflush(echofile);

  fscanf(chemfile, "%lf", &reacbalance);
  fprintf(
    echofile,
    "kmodSurf.list : Tolerance for reaction balance                    : %e\n",
    reacbalance);
  fprintf(echofile,
          "----------------------------------------------------------"
          "---------------\n");
  fflush(echofile);

  fscanf(chemfile, "%lf", &TCsurf_siteden_);
  fprintf(
    echofile,
    "kmodSurf.list : Site density                                      : %e\n",
    TCsurf_siteden_);
  fprintf(echofile,
          "----------------------------------------------------------"
          "---------------\n");
  fflush(echofile);

  ///
  /// Allocate views
  ///
  allocateViewsSurf(errfile);

  /// Host views to read data from a file
  auto TCsurf_sNamesHost = TCsurf_sNames_.view_host();
  auto TCsurf_sMassHost = TCsurf_sMass_.view_host();
  auto TCsurf_elemcountHost = TCsurf_elemcount_.view_host();
  auto TCsurf_sChargeHost = TCsurf_sCharge_.view_host();
  auto TCsurf_sTfitHost = TCsurf_sTfit_.view_host();
  auto TCsurf_sPhaseHost = TCsurf_sPhase_.view_host();

  auto TCsurf_TloHost = TCsurf_Tlo_.view_host();
  auto TCsurf_TmiHost = TCsurf_Tmi_.view_host();
  auto TCsurf_ThiHost = TCsurf_Thi_.view_host();
  auto TCsurf_cppolHost = TCsurf_cppol_.view_host();

  auto TCsurf_isRevHost = TCsurf_isRev_.view_host();
  auto TCsurf_reacNrpHost = TCsurf_reacNrp_.view_host();

  auto TCsurf_reacNreacHost = TCsurf_reacNreac_.view_host();
  auto TCsurf_reacNprodHost = TCsurf_reacNprod_.view_host();

  auto TCsurf_reacScoefHost = TCsurf_reacScoef_.view_host();
  auto TCsurf_reacNukiHost = TCsurf_reacNuki_.view_host();
  auto TCsurf_reacSidxHost = TCsurf_reacSidx_.view_host();
  auto TCsurf_reacSsrfHost = TCsurf_reacSsrf_.view_host();

  auto TCsurf_reacArhenForHost = TCsurf_reacArhenFor_.view_host();
  // auto TCsurf_reacArhenRevHost = TCsurf_reacArhenRev_.view_host();

  auto TCsurf_isDupHost = TCsurf_isDup_.view_host();
  auto TCsurf_isStickHost = TCsurf_isStick_.view_host();
  // auto .view_host();

  // from gas reader
  auto eNamesHost = eNames_.view_host();
  auto sNamesHost = sNames_.view_host();

  /* Species' name and weights */

  /* stoichiometric matrix gas species*/
  auto vskiHost = vski_.view_host();
  auto vsurfkiHost = vsurfki_.view_host();

  // Species
  for (int i = 0; i < TCsurf_Nspec_; i++) {
    char specNm[LENGTHOFSPECNAME];
    fscanf(chemfile, "%s", specNm);
    strncat(&TCsurf_sNamesHost(i, 0), specNm, LENGTHOFSPECNAME);
  }

  for (int i = 0; i < TCsurf_Nspec_; i++)
    fscanf(chemfile, "%lf", &(TCsurf_sMassHost(i)));

  fprintf(echofile, "No. \t Species \t Mass\n");
  for (int i = 0; i < TCsurf_Nspec_; i++)
    fprintf(echofile,
            "%-3d\t%-32s\t%12.7f\n",
            i + 1,
            &TCsurf_sNamesHost(i, 0),
            TCsurf_sMassHost(i));
  DASHLINE(echofile);
  fflush(echofile);

  /* Species' elemental composition */

  for (int i = 0; i < TCsurf_Nspec_; i++)
    for (int j = 0; j < nElem_; j++)
      fscanf(chemfile, "%d", &TCsurf_elemcountHost(i, j));

  fprintf(echofile, "Elemental composition of species\n");
  fprintf(echofile, "No. \t Species \t\t\t\t Element\n\t\t\t\t\t");

  for (int i = 0; i < nElem_; i++)
    fprintf(echofile, "%s\t", &eNamesHost(i, 0));
  fprintf(echofile, "\n");

  for (int i = 0; i < TCsurf_Nspec_; i++) {
    fprintf(echofile, "%-3d\t%-32s", i + 1, &TCsurf_sNamesHost(i, 0));
    for (int j = 0; j < nElem_; j++)
      fprintf(echofile, "%-3d\t", TCsurf_elemcountHost(i, j));
    fprintf(echofile, "\n");
  }

  /* Species charges, no of tempfits, phase */
  for (int i = 0; i < TCsurf_Nspec_; i++)
    fscanf(chemfile, "%d", &(TCsurf_sChargeHost(i)));
  for (int i = 0; i < TCsurf_Nspec_; i++)
    fscanf(chemfile, "%d", &(TCsurf_sTfitHost(i)));
  for (int i = 0; i < TCsurf_Nspec_; i++)
    fscanf(chemfile, "%d", &(TCsurf_sPhaseHost(i)));

  /* Range of temperatures for thermo fits */
  TCsurf_TthrmMin_ = 1.e-5;
  TCsurf_TthrmMax_ = 1.e+5;
  for (int i = 0; i < TCsurf_Nspec_; i++) {
    fscanf(chemfile, "%lf", &(TCsurf_TloHost(i)));
    fscanf(chemfile, "%lf", &(TCsurf_TmiHost(i)));
    fscanf(chemfile, "%lf", &(TCsurf_ThiHost(i)));
    TCsurf_TthrmMin_ = std::min(TCsurf_TthrmMin_, TCsurf_TloHost(i));
    TCsurf_TthrmMax_ = std::max(TCsurf_TthrmMax_, TCsurf_ThiHost(i));
  }

  fprintf(echofile, "Range of temperature for thermodynamic fits\n");
  fprintf(echofile, "No. \t Species \t\t Tlow \tTmid \tThigh\n");
  for (int i = 0; i < TCsurf_Nspec_; i++) {
    fprintf(echofile,
            "%-3d\t%-32s %12.4f\t%12.4f\t%12.4f\n",
            i + 1,
            &TCsurf_sNamesHost(i, 0),
            TCsurf_TloHost(i),
            TCsurf_TmiHost(i),
            TCsurf_ThiHost(i));
  }
  DASHLINE(echofile);

  /* Polynomial coeffs for thermo fits */
  for (int i = 0; i < TCsurf_Nspec_; i++)
    for (int j = 0; j < TCsurf_nNASAinter_; j++)
      for (int k = 0; k < TCsurf_nCpCoef_ + 2; k++)
        fscanf(chemfile, "%lf", &TCsurf_cppolHost(i, j, k));

  fprintf(echofile, "List of coefficients for thermodynamic fits\n");

  for (int i = 0; i < TCsurf_Nspec_; i++) {
    const auto ptr = &TCsurf_cppolHost(i, 0, 0);
    fprintf(echofile,
            "%-4d %-32s\n %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e\n",
            i + 1,
            &TCsurf_sNamesHost(i, 0),
            ptr[0],
            ptr[1],
            ptr[2],
            ptr[3],
            ptr[4],
            ptr[5],
            ptr[6]);
    fprintf(echofile,
            " %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e\n",
            ptr[7],
            ptr[8],
            ptr[9],
            ptr[10],
            ptr[11],
            ptr[12],
            ptr[13]);
  }
  DASHLINE(echofile);

  /* reaction info */
  if (TCsurf_Nreac_ > 0) {

    /* no of reactants and products */
    for (int i = 0; i < TCsurf_Nreac_; i++) {
      int itemp;
      fscanf(chemfile, "%d", &itemp);
      if (itemp < 0)
        TCsurf_isRevHost(i) = 0;
      else
        TCsurf_isRevHost(i) = 1;
      // TCsurf_isRevHost(i) = (itemp>=0);
      TCsurf_reacNrpHost(i) = abs(itemp);
    }

    /* no of reactants only */
    for (int i = 0; i < TCsurf_Nreac_; i++)
      fscanf(chemfile, "%d", &(TCsurf_reacNreacHost(i)));

    /* no of products */
    for (int i = 0; i < TCsurf_Nreac_; i++)
      TCsurf_reacNprodHost(i) = TCsurf_reacNrpHost(i) - TCsurf_reacNreacHost(i);

    // /* Stoichiometric coefficients */
    for (int i = 0; i < TCsurf_Nreac_; i++) {

      int nusumk = 0;
      /* by default reaction has integer stoichiometric coefficients */
      TCsurf_reacScoefHost(i) = -1;

      nusumk = 0;
      for (int j = 0; j < TCsurf_maxSpecInReac_; j++) {
        fscanf(chemfile, "%d", &(TCsurf_reacNukiHost(i, j)));
        fscanf(chemfile, "%d", &(TCsurf_reacSsrfHost(i, j)));
        fscanf(chemfile, "%d", &(TCsurf_reacSidxHost(i, j)));

        TCsurf_reacSidxHost(i, j) -= 1;
        nusumk += TCsurf_reacNukiHost(i, j);
      }
      int itemp;
      fscanf(chemfile, "%d", &itemp);
      assert(itemp == nusumk);
    }

    /* Arrhenius parameters */
    for (int i = 0; i < TCsurf_Nreac_; i++)
      for (int j = 0; j < 3; ++j)
        fscanf(chemfile, "%lf", &(TCsurf_reacArhenForHost(i, j)));

    for (int i = 0; i < TCsurf_Nreac_; i++) {
      fscanf(chemfile, "%d", &(TCsurf_isDupHost(i)));
    }

    for (int i = 0; i < TCsurf_Nreac_; i++) {
      fscanf(chemfile, "%d", &(TCsurf_isStickHost(i)));
    }

    fprintf(echofile, "Reaction data : species and Arrhenius pars\n");
    for (int i = 0; i < TCsurf_Nreac_; i++) {
      fprintf(echofile,
              "%-5d\t%1d\t%2d\t%2d | ",
              i + 1,
              TCsurf_isRevHost(i),
              TCsurf_reacNreacHost(i),
              TCsurf_reacNprodHost(i));
      for (int j = 0; j < TCsurf_reacNreacHost(i); j++) {
        if (TCsurf_reacSsrfHost(i, j) == 1) {
          fprintf(echofile,
                  "%d*%s | ",
                  TCsurf_reacNukiHost(i, j),
                  &TCsurf_sNamesHost(TCsurf_reacSidxHost(i, j), 0));
        } else {
          fprintf(echofile,
                  "%d*%s | ",
                  TCsurf_reacNukiHost(i, j),
                  &sNamesHost(TCsurf_reacSidxHost(i, j), 0));
        }
      }

      const int joff = TCsurf_maxSpecInReac_ / 2;

      for (int j = 0; j < TCsurf_reacNprodHost(i); j++)
        if (TCsurf_reacSsrfHost(i, j + joff) == 1) {
          fprintf(echofile,
                  "%d*%s | ",
                  TCsurf_reacNukiHost(i, j + joff),
                  &TCsurf_sNamesHost(TCsurf_reacSidxHost(i, j + joff), 0));
        } else {
          fprintf(echofile,
                  "%d*%s | ",
                  TCsurf_reacNukiHost(i, j + joff),
                  &sNamesHost(TCsurf_reacSidxHost(i, j + joff), 0));
        }

      fprintf(echofile,
              "%16.8e\t%16.8e\t%16.8e",
              TCsurf_reacArhenForHost(i, 0),
              TCsurf_reacArhenForHost(i, 1),
              TCsurf_reacArhenForHost(i, 2));

      if (TCsurf_isDupHost[i] == 1)
        fprintf(echofile, "  DUPLICATE");
      if (TCsurf_isStickHost[i] == 1)
        fprintf(echofile, "  STICK");
      fprintf(echofile, "\n");
    }
    DASHLINE(echofile);
    if (verboseEnabled)
      printf("KineticModelData::initChem() : Done reading reaction data\n");

    fclose(chemfile);
    fclose(errfile);
    fclose(echofile);

    // vskiHost stoichiometric matrix gas species in surface reaction mechanism
    // vsurfkiHost  stoichiometric matrix surface species in surface reaction
    // mechanism
    for (ordinal_type i = 0; i < TCsurf_Nreac_; i++) {
      // reactans
      for (ordinal_type j = 0; j < TCsurf_reacNreacHost(i); ++j) {
        const ordinal_type kspec = TCsurf_reacSidxHost(i, j);
        if (TCsurf_reacSsrfHost(i, j) == 0) { // only gas species
          vskiHost(kspec, i) = TCsurf_reacNukiHost(i, j);
        } else if (TCsurf_reacSsrfHost(i, j) == 1) { // only surface species
          vsurfkiHost(kspec, i) = TCsurf_reacNukiHost(i, j);
        }
      }

      const ordinal_type joff = TCsurf_maxSpecInReac_ / 2;
      for (ordinal_type j = 0; j < TCsurf_reacNprodHost(i); ++j) {
        const ordinal_type kspec = TCsurf_reacSidxHost(i, j + joff);
        if (TCsurf_reacSsrfHost(i, j + joff) == 0) { // only gas species
          vskiHost(kspec, i) = TCsurf_reacNukiHost(i, j + joff);
        } else if (TCsurf_reacSsrfHost(i, j + joff) ==
                   1) { // only surface species
          vsurfkiHost(kspec, i) = TCsurf_reacNukiHost(i, j + joff);
        }
      }
    }
  }

  /// Raise modify flags for all modified dual views
  TCsurf_isStick_.modify_host();
  TCsurf_isDup_.modify_host();
  TCsurf_reacArhenFor_.modify_host();
  TCsurf_reacNuki_.modify_host();
  TCsurf_reacSidx_.modify_host();
  TCsurf_reacSsrf_.modify_host();
  TCsurf_reacNreac_.modify_host();
  TCsurf_reacNprod_.modify_host();
  TCsurf_reacScoef_.modify_host();
  TCsurf_reacNrp_.modify_host();
  TCsurf_isRev_.modify_host();
  TCsurf_cppol_.modify_host();
  TCsurf_Tlo_.modify_host();
  TCsurf_Tmi_.modify_host();
  TCsurf_Thi_.modify_host();
  TCsurf_sCharge_.modify_host();
  TCsurf_sTfit_.modify_host();
  TCsurf_sPhase_.modify_host();
  TCsurf_elemcount_.modify_host();
  TCsurf_sMass_.modify_host();
  TCsurf_sNames_.modify_host();
  vski_.modify_host();
  vsurfki_.modify_host();

  syncSurfToDevice();

  return (0);
}

void
KineticModelData::modifyArrheniusForwardParametersBegin() {
  /// back up data (soft copy and reference counting keep the memory)
  auto reacArhenFor = reacArhenFor_.view_host();

  /// create a new allocation and copy the content of the previous arhenius parameters
  reacArhenFor_ = real_type_2d_dual_view
    (do_not_init_tag("KMD::reacArhenFor"), reacArhenFor.extent(0), reacArhenFor.extent(1));
  Kokkos::deep_copy(reacArhenFor_.view_host(), reacArhenFor);

  /// raise a flag modify host so that we can transfer data to device
  reacArhenFor_.modify_host();
}

real_type
KineticModelData::getArrheniusForwardParameter(const int i, const int j) {
  TCHEM_CHECK_ERROR(i >= reacArhenFor_.view_host().extent(0),
		    "Error: indeix i is greater than the view extent(0)");
  TCHEM_CHECK_ERROR(j >= reacArhenFor_.view_host().extent(1),
		    "Error: indeix j is greater than the view extent(1)");

  /// modify arhenius parameter
  return reacArhenFor_.view_host()(i,j);
}

void
KineticModelData::modifyArrheniusForwardParameter(const int i, const int j, const real_type value) {
  TCHEM_CHECK_ERROR(i >= reacArhenFor_.view_host().extent(0),
		    "Error: indeix i is greater than the view extent(0)");
  TCHEM_CHECK_ERROR(j >= reacArhenFor_.view_host().extent(1),
		    "Error: indeix j is greater than the view extent(1)");

  /// modify arhenius parameter
  reacArhenFor_.view_host()(i,j) = value;
}

void
KineticModelData::modifyArrheniusForwardParametersEnd() {
  /// device view is now synced with the host view
  reacArhenFor_.sync_device();
}

} // namespace TChem
