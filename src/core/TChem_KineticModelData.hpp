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
#ifndef __TCHEM_KINETIC_MODELMDATA_HPP__
#define __TCHEM_KINETIC_MODELMDATA_HPP__

#include "TC_kmodint_surface.hpp"
#include "TChem_Util.hpp"

#if defined(TCHEM_ENABLE_TPL_YAML_CPP)
#include "yaml-cpp/yaml.h"
#endif

namespace TChem {


template<typename DeviceType>
struct KineticModelNCAR_ConstData
{
public:

  using device_type = DeviceType;
  using exec_space_type = typename device_type::execution_space;

  /// non const
  using real_type_0d_dual_view = Tines::value_type_0d_dual_view<real_type,device_type>;
  using real_type_1d_dual_view = Tines::value_type_1d_dual_view<real_type,device_type>;
  using real_type_2d_dual_view = Tines::value_type_2d_dual_view<real_type,device_type>;
  using real_type_3d_dual_view = Tines::value_type_3d_dual_view<real_type,device_type>;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
  using real_type_3d_view_type = Tines::value_type_3d_view<real_type,device_type>;

  using ordinal_type_1d_dual_view = Tines::value_type_1d_dual_view<ordinal_type,device_type>;
  using ordinal_type_2d_dual_view = Tines::value_type_2d_dual_view<ordinal_type,device_type>;

  using ordinal_type_1d_view = Tines::value_type_1d_view<ordinal_type,device_type>;
  using ordinal_type_2d_view = Tines::value_type_2d_view<ordinal_type,device_type>;

  using string_type_1d_dual_view_type = Tines::value_type_1d_dual_view<char [LENGTHOFSPECNAME + 1],device_type>;
  using string_type_1d_view_type = Tines::value_type_1d_view<char [LENGTHOFSPECNAME + 1],device_type>;

  using arrhenius_reaction_type_1d_view_type =  Tines::value_type_1d_view<ArrheniusReactionType,device_type>;

  //// const views
  using kmcd_ordinal_type_1d_view = ConstUnmanaged<ordinal_type_1d_view>;
  using kmcd_ordinal_type_2d_view = ConstUnmanaged<ordinal_type_2d_view>;

  using kmcd_real_type_1d_view = ConstUnmanaged<real_type_1d_view_type>;
  using kmcd_real_type_2d_view = ConstUnmanaged<real_type_2d_view_type>;
  using kmcd_real_type_3d_view = ConstUnmanaged<real_type_3d_view_type>;

  using kmcd_string_type_1d_view =ConstUnmanaged<string_type_1d_view_type>;

  using kmcd_arrhenius_reaction_type_1d_view_1d_view = ConstUnmanaged<arrhenius_reaction_type_1d_view_type>;

  kmcd_string_type_1d_view speciesNames;
  kmcd_ordinal_type_1d_view reacNreac;
  kmcd_ordinal_type_1d_view reacNprod;
  kmcd_real_type_2d_view reacArhenFor;
  kmcd_real_type_2d_view reacNuki;
  kmcd_ordinal_type_2d_view reacSidx;
  kmcd_arrhenius_reaction_type_1d_view_1d_view ArrheniusCoef;
  ordinal_type nSpec;
  ordinal_type nReac;
  kmcd_ordinal_type_1d_view reacPfal;
  kmcd_real_type_2d_view reacPpar;
  ordinal_type nConstSpec;


};

template<typename DeviceType>
struct KineticModelConstData
{
public:

  using device_type = DeviceType;
  using exec_space_type = typename device_type::execution_space;

  /// non const
  using real_type_0d_dual_view = Tines::value_type_0d_dual_view<real_type,device_type>;
  using real_type_1d_dual_view = Tines::value_type_1d_dual_view<real_type,device_type>;
  using real_type_2d_dual_view = Tines::value_type_2d_dual_view<real_type,device_type>;
  using real_type_3d_dual_view = Tines::value_type_3d_dual_view<real_type,device_type>;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
  using real_type_3d_view_type = Tines::value_type_3d_view<real_type,device_type>;

  using ordinal_type_1d_dual_view = Tines::value_type_1d_dual_view<ordinal_type,device_type>;
  using ordinal_type_2d_dual_view = Tines::value_type_2d_dual_view<ordinal_type,device_type>;

  using ordinal_type_1d_view = Tines::value_type_1d_view<ordinal_type,device_type>;
  using ordinal_type_2d_view = Tines::value_type_2d_view<ordinal_type,device_type>;

  using string_type_1d_dual_view_type = Tines::value_type_1d_dual_view<char [LENGTHOFSPECNAME + 1],device_type>;
  using string_type_1d_view_type = Tines::value_type_1d_view<char [LENGTHOFSPECNAME + 1],device_type>;

  //// const views
  using kmcd_ordinal_type_1d_view = ConstUnmanaged<ordinal_type_1d_view>;
  using kmcd_ordinal_type_2d_view = ConstUnmanaged<ordinal_type_2d_view>;

  using kmcd_real_type_1d_view = ConstUnmanaged<real_type_1d_view_type>;
  using kmcd_real_type_2d_view = ConstUnmanaged<real_type_2d_view_type>;
  using kmcd_real_type_3d_view = ConstUnmanaged<real_type_3d_view_type>;

  using kmcd_string_type_1d_view =ConstUnmanaged<string_type_1d_view_type>;

  real_type rho;
  kmcd_string_type_1d_view speciesNames;

  // ordinal_type nNASAinter;
  // ordinal_type nCpCoef;
  // ordinal_type nArhPar;
  // ordinal_type nLtPar;
  // ordinal_type nJanPar;
  // ordinal_type nFit1Par;
  // ordinal_type nIonSpec;
  // ordinal_type electrIndx;
  // ordinal_type nIonEspec;
  ordinal_type nElem; // this include all elements from chem.inp
  ordinal_type
    NumberofElementsGas; // this only include elements present in gas species
  ordinal_type nSpec;
  ordinal_type nReac;
  // ordinal_type nNASA9coef;
  ordinal_type nThbReac;
  // ordinal_type maxTbInReac;
  ordinal_type nFallReac;
  ordinal_type nFallPar;
  // ordinal_type maxSpecInReac;
  ordinal_type maxOrdPar;
  ordinal_type nRealNuReac;
  ordinal_type nRevReac;
  ordinal_type nOrdReac;
  ordinal_type nPlogReac;

  ordinal_type jacDim;
  bool enableRealReacScoef;

  real_type Runiv;
  real_type Rcal;
  real_type Rcgs;
  real_type TthrmMin;
  real_type TthrmMax;

  // kmcd_string_type_1d_view<LENGTHOFSPECNAME+1> sNames ;

  // kmcd_ordinal_type_1d_view spec9t;
  // kmcd_ordinal_type_1d_view spec9nrng;
  // kmcd_ordinal_type_1d_view sigNu;
  kmcd_ordinal_type_1d_view reacTbdy;
  kmcd_ordinal_type_1d_view reacTbno;
  // kmcd_ordinal_type_1d_view reac_to_Tbdy_index;
  kmcd_ordinal_type_1d_view reacPfal;
  kmcd_ordinal_type_1d_view reacPtype;
  kmcd_ordinal_type_1d_view reacPlohi;
  kmcd_ordinal_type_1d_view reacPspec;
  kmcd_ordinal_type_1d_view isRev;
  // kmcd_ordinal_type_1d_view isDup;
  kmcd_ordinal_type_1d_view reacAOrd;
  // kmcd_ordinal_type_1d_view reacNrp;
  kmcd_ordinal_type_1d_view reacNreac;
  kmcd_ordinal_type_1d_view reacNprod;
  kmcd_ordinal_type_1d_view reacScoef;
  kmcd_ordinal_type_1d_view reacRnu;
  kmcd_ordinal_type_1d_view reacRev;
  kmcd_ordinal_type_1d_view reacHvIdx;
  kmcd_ordinal_type_1d_view reacPlogIdx;
  kmcd_ordinal_type_1d_view reacPlogPno;
  // kmcd_ordinal_type_1d_view sNion;
  // kmcd_ordinal_type_1d_view sCharge;
  // kmcd_ordinal_type_1d_view sTfit;
  // kmcd_ordinal_type_1d_view sPhase;

  kmcd_real_type_2d_view NuIJ;
  kmcd_ordinal_type_2d_view specTbdIdx;
  kmcd_real_type_2d_view reacNuki;
  kmcd_ordinal_type_2d_view reacSidx;
  kmcd_ordinal_type_2d_view specAOidx;
  // kmcd_ordinal_type_2d_view elemCount;

  kmcd_real_type_1d_view sMass;
  // kmcd_real_type_1d_view Tlo;
  kmcd_real_type_1d_view Tmi;
  // kmcd_real_type_1d_view Thi;
  // kmcd_real_type_1d_view sigRealNu;
  // kmcd_real_type_1d_view reacHvPar;
  kmcd_real_type_1d_view kc_coeff;
  // kmcd_real_type_1d_view eMass;

  kmcd_real_type_2d_view RealNuIJ;
  kmcd_real_type_2d_view reacArhenFor;
  kmcd_real_type_2d_view reacArhenRev;
  kmcd_real_type_2d_view specTbdEff;
  kmcd_real_type_2d_view reacPpar;
  kmcd_real_type_2d_view reacRealNuki;
  kmcd_real_type_2d_view specAOval;
  kmcd_real_type_2d_view reacPlogPars;

  kmcd_real_type_3d_view cppol;
  kmcd_real_type_2d_view stoiCoefMatrix;

  // kmcd_real_type_3d_view spec9trng;
  // kmcd_real_type_3d_view spec9coefs;
};

template<typename DeviceType>
struct KineticSurfModelConstData
{
public:

  using device_type = DeviceType;
  using exec_space_type = typename device_type::execution_space;

  /// non const
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
  using real_type_3d_view_type = Tines::value_type_3d_view<real_type,device_type>;

  using ordinal_type_1d_view = Tines::value_type_1d_view<ordinal_type,device_type>;
  using ordinal_type_2d_view = Tines::value_type_2d_view<ordinal_type,device_type>;

  using string_type_1d_view_type = Tines::value_type_1d_view<char [LENGTHOFSPECNAME + 1],device_type>;

  using kmcd_coverage_modification_type = Tines::value_type_1d_view<CoverageModification, device_type>;

  //// const views
  using kmcd_ordinal_type_1d_view = ConstUnmanaged<ordinal_type_1d_view>;
  using kmcd_ordinal_type_2d_view = ConstUnmanaged<ordinal_type_2d_view>;

  using kmcd_real_type_1d_view = ConstUnmanaged<real_type_1d_view_type>;
  using kmcd_real_type_2d_view = ConstUnmanaged<real_type_2d_view_type>;
  using kmcd_real_type_3d_view = ConstUnmanaged<real_type_3d_view_type>;

  using kmcd_string_type_1d_view = ConstUnmanaged<string_type_1d_view_type>;

  using kmcd_coverage_modification_type_1d_view = ConstUnmanaged<kmcd_coverage_modification_type>;


  kmcd_coverage_modification_type_1d_view coverageFactor;

  kmcd_string_type_1d_view speciesNames;

  ordinal_type nSpec;
  ordinal_type nReac;
  real_type sitedensity;

  kmcd_real_type_3d_view cppol;
  real_type Runiv;
  // real_type Rcal;
  // real_type Rcgs;
  real_type TthrmMin;
  real_type TthrmMax;
  kmcd_real_type_1d_view Tmi;
  kmcd_real_type_2d_view reacArhenFor;
  kmcd_ordinal_type_1d_view isStick;
  // kmcd_real_type_2d_view reacArhenRev;

  kmcd_ordinal_type_1d_view isRev;

  ordinal_type maxSpecInReac;

  kmcd_ordinal_type_1d_view reacNreac;
  kmcd_ordinal_type_1d_view reacNprod;
  kmcd_ordinal_type_2d_view reacSsrf;
  kmcd_ordinal_type_2d_view reacNuki;
  kmcd_ordinal_type_2d_view reacSidx;

  kmcd_ordinal_type_2d_view vki;
  kmcd_ordinal_type_2d_view vsurfki;

  real_type TChem_reltol;
  real_type TChem_abstol;
};

struct KineticModelData
{
public:
  /** \var real_type_1d_dual_view TC_sMass_
      \brief array of species molar masses */
  real_type_1d_dual_view sMass_;

  /* temperature limits for NASA polynomials and their coefficients */
  real_type_1d_dual_view Tlo_, Tmi_, Thi_;
  real_type_3d_dual_view cppol_;
  real_type TthrmMin_, TthrmMax_;

  /* universal gas constant */
  real_type Runiv_, Rcal_, Rcgs_;

  /* temperature limits for NASA polynomials and their coefficients */
  ordinal_type nNASA9coef_;
  ordinal_type_1d_dual_view spec9t_, spec9nrng_;
  real_type_3d_dual_view spec9trng_, spec9coefs_;

  real_type_1d_dual_view sigNu_;
  real_type_2d_dual_view NuIJ_;
  real_type_1d_dual_view sigRealNu_;
  real_type_2d_dual_view RealNuIJ_;

  /* Arrhenius parameters, forward and reverse */
  real_type_2d_dual_view reacArhenFor_, reacArhenRev_;

  /* Third-body data */
  ordinal_type nThbReac_, maxTbInReac_;
  ordinal_type_1d_dual_view reacTbdy_, reacTbno_;
  ordinal_type_2d_dual_view specTbdIdx_;
  real_type_2d_dual_view specTbdEff_;

  ordinal_type_1d_dual_view
    reac_to_Tbdy_index_; // Mapping of reaction index to for 3rd_body

  /* Pressure dependent reactions */
  ordinal_type nFallReac_, nFallPar_;
  ordinal_type_1d_dual_view reacPfal_, reacPtype_, reacPlohi_, reacPspec_;
  real_type_2d_dual_view reacPpar_;

  /** \var ordinal_type maxSpecInReac_
   *  \ingroup maxpar
   *  \brief Maximum number of species in a reaction */
  ordinal_type maxSpecInReac_;
  /** \var ordinal_type maxOrdPar_
   *  \ingroup maxpar
   *  \brief # of parameters for arbitrary order reactions */
  ordinal_type maxOrdPar_;

  ordinal_type_1d_dual_view isRev_;

  /* is reaction a duplicate ? */
  ordinal_type_1d_dual_view isDup_;

  /* no. of reac+prod, no. of reac only, no. of prod only, stoichiom.coef.
   * indicator */
  ordinal_type_1d_dual_view reacNrp_, reacNreac_, reacNprod_, reacScoef_;

  /* stoichiometric coeffs and reactants and product indices */
  ordinal_type_2d_dual_view reacSidx_;
  real_type_2d_dual_view reacNuki_;

  real_type_2d_dual_view stoiCoefMatrix_;

  /* real stoichiometric coeffs */
  ordinal_type nRealNuReac_;
  ordinal_type_1d_dual_view reacRnu_;
  real_type_2d_dual_view reacRealNuki_;

  /* list of reactions with given reverse Arrhenius parameters */
  ordinal_type nRevReac_;
  ordinal_type_1d_dual_view reacRev_;

  /* Reactions with arbitrary orders  */
  ordinal_type nOrdReac_;
  ordinal_type_1d_dual_view reacAOrd_;
  ordinal_type_2d_dual_view specAOidx_;
  real_type_2d_dual_view specAOval_;

  /* Radiation wavelength data */
  ordinal_type_1d_dual_view reacHvIdx_;
  real_type_1d_dual_view reacHvPar_;

  /* PLOG reactions */
  ordinal_type nPlogReac_;
  ordinal_type_1d_dual_view reacPlogIdx_, reacPlogPno_;
  real_type_2d_dual_view reacPlogPars_;

  /* Equilibrium constants */
  real_type_1d_dual_view kc_coeff_;

  /** \var ordinal_type isInit_
   *  \ingroup maxpar
   \brief class is initiazed or not */
  ordinal_type isInit_;

  /** \var static ordinal_type TC_nNASAinter_
   *  \ingroup maxpar
   \brief # of temperature regions for thermo fits */
  ordinal_type nNASAinter_;

  /** \var static ordinal_type TC_nCpCoef_
   *  \ingroup maxpar
   \brief # of polynomial coefficients for thermo fits */
  ordinal_type nCpCoef_;
  /** \var static ordinal_type TC_nArhPar_
   *  \ingroup maxpar
   \brief # of Arrhenius parameters */
  ordinal_type nArhPar_;
  /** \var static ordinal_type TC_nLtPar_
   *  \ingroup maxpar
   \brief # of parameters for Landau-Teller reactions */
  ordinal_type nLtPar_;
  /** \var static ordinal_type TC_nJanPar_
   *  \ingroup maxpar
   \brief # of parameters for Jannev-Langer fits (JAN) */
  ordinal_type nJanPar_;
  /** \var static ordinal_type TC_nFit1Par_
   *  \ingroup maxpar
   \brief # of parameters for FIT1 fits */
  ordinal_type nFit1Par_;

  /** \var static ordinal_type TC_nIonSpec_
   *  \ingroup nospec
   \brief # of ion species */
  ordinal_type nIonSpec_;
  /** \var static ordinal_type TC_electrIndx_
   *  \ingroup nospec
   \brief Index of the electron species */
  ordinal_type electrIndx_;
  /** \var static ordinal_type TC_nIonEspec_
   *  \ingroup nospec
   \brief # of ion species excluding the electron species */
  ordinal_type nIonEspec_;
  /** \var static ordinal_type TC_nNASA9coef_
   *  \ingroup nospec
   \brief # of species with 9-term NASA polynomial fits */

  /* ---------------------------------------------------
     elements & species -> names, masses, and element count
     ---------------------------------------------------- */
  ordinal_type nElem_, nSpec_;
  ordinal_type NumberofElementsGas_;
  /** \var StringVector TC_sNames_
      \brief array of species names */
  string_type_1d_dual_view<LENGTHOFSPECNAME + 1> sNames_;
  /** \var StringVector TC_eNames_
      \brief species names, name of species i stored at LENGTHOFELEMNAME*i */
  string_type_1d_dual_view<LENGTHOFELEMNAME + 1> eNames_;
  /** \var real_type_1d_dual_view TC_eMass_
      \brief array of element molar masses */
  real_type_1d_dual_view eMass_;
  /** \var static ordinal_type TC_elemcount_
      \brief no. of atoms of element j in each species i at (i*TC_nElem_+j)*/
  ordinal_type_2d_dual_view elemCount_; // (nSpec_, nElem_)

  /* list of non-electron ion species */
  ordinal_type_1d_dual_view sNion_;

  /* Reaction data */
  /* is reaction reversible ? */
  ordinal_type nReac_;

  ordinal_type_1d_dual_view sCharge_, sTfit_, sPhase_;

  void allocateViews(FILE* errfile);
  void syncToDevice();

  /* surface chemistry */
  ordinal_type TCsurf_maxSpecInReac_, TCsurf_nNASAinter_, TCsurf_nCpCoef_,
    TCsurf_nArhPar_, TCsurf_Nspec_, TCsurf_Nreac_, TCsurf_NcoverageFactors;

  string_type_1d_dual_view<LENGTHOFSPECNAME + 1> TCsurf_sNames_;

  real_type_1d_dual_view TCsurf_sMass_;
  ordinal_type_2d_dual_view TCsurf_elemcount_;
  ordinal_type_1d_dual_view TCsurf_sCharge_, TCsurf_sTfit_, TCsurf_sPhase_;

  /* temperature limits for NASA polynomials and their coefficients */
  real_type_1d_dual_view TCsurf_Tlo_, TCsurf_Tmi_, TCsurf_Thi_;

  real_type TCsurf_TthrmMin_, TCsurf_TthrmMax_, TCsurf_siteden_;

  real_type_3d_dual_view TCsurf_cppol_;
  ordinal_type_1d_dual_view TCsurf_reacNrp_, TCsurf_isRev_;
  ordinal_type_1d_dual_view TCsurf_reacNreac_, TCsurf_reacNprod_,
    TCsurf_reacScoef_;
  /* stoichiometric coeffs and reactants and product indices */
  ordinal_type_2d_dual_view TCsurf_reacNuki_, TCsurf_reacSidx_,
    TCsurf_reacSsrf_;

  /* Arrhenius parameters, forward and reverse */
  real_type_2d_dual_view TCsurf_reacArhenFor_;
  /* is reaction a duplicate ? */
  ordinal_type_1d_dual_view TCsurf_isDup_, TCsurf_isStick_;

  ordinal_type_2d_dual_view vski_;
  ordinal_type_2d_dual_view vsurfki_;

  coverage_modification_type_1d_dual_view coverageFactor_;

  // aerosol chemistry
  arrhenius_reaction_type_1d_dual_view ArrheniusCoef_;
  ordinal_type  nConstSpec_;



  void syncSurfToDevice();
  void allocateViewsSurf(FILE* errfile);

public:
  KineticModelData(const std::string& mechfile, const bool& hasSurface=false);


  KineticModelData(const std::string& mechfile, const std::string& thermofile);

  KineticModelData(const std::string& mechfile,
                   const std::string& thermofile,
                   const std::string& mechSurffile,
                   const std::string& thermoSurffile);

  /// constructor and destructor
  KineticModelData() = default;
  KineticModelData(const KineticModelData& b) = default;
  ~KineticModelData() = default;

  ordinal_type initChem();
  ordinal_type initChemSurf();

#if defined(TCHEM_ENABLE_TPL_YAML_CPP)
  ordinal_type initChemYaml(YAML::Node& doc, const int& gasPhaseIndex);
  ordinal_type initChemSurfYaml(YAML::Node& doc, const int& surfacePhaseIndex);
  ordinal_type initChemNCAR(YAML::Node& doc);
#endif
/// this modify begin allocate a view to replace the existing reacArhenFor_
void modifyArrheniusForwardParametersBegin();

/// change the values as much as a user want within the begin, end clauses
real_type getArrheniusForwardParameter(const int i, const int j);
void modifyArrheniusForwardParameter(const int i, const int j, const real_type value);

/// this sync the host values to device ; now it is ready to construct a const object
void modifyArrheniusForwardParametersEnd();

void modifyArrheniusForwardSurfaceParametersBegin();

real_type getArrheniusForwardSurfaceParameter(const int i, const int j);
void modifyArrheniusForwardSurfaceParameter(const int i, const int j, const real_type value);

void modifyArrheniusForwardSurfaceParametersEnd();

// to modify plog reaction's paramters
void modifyArrheniusForwardParametersPLOG_Begin();
real_type getArrheniusForwardParameterPLOG(const int i, const int j);
void modifyArrheniusForwardParameterPLOG(const int i, const int j, const real_type value);
void modifyArrheniusForwardParametersPLOG_End();



  /// copy only things needed; we need to review what is actually needed for
  /// computations
  // gas combustion
  template<typename SpT>
  KineticModelConstData<SpT> createConstData()
  {
    KineticModelConstData<SpT> data;
    // using SpT = typename DeviceType::execution_space;

    /// given from non-const kinetic model data
    data.rho =
      real_type(-1); /// there is no minus density if this is minus, rhoset = 0
    data.speciesNames = sNames_.template view<SpT>();
    // data.nNASAinter = nNASAinter_;
    // data.nCpCoef = nCpCoef_;
    // data.nArhPar = nArhPar_;
    // data.nLtPar = nLtPar_;
    // data.nJanPar = nJanPar_;
    // data.nFit1Par = nFit1Par_;
    // data.nIonSpec = nIonSpec_;
    // data.electrIndx = electrIndx_;
    // data.nIonEspec = nIonEspec_;
    data.nElem = nElem_; // includes all elements from chem.inp
    data.NumberofElementsGas =
      NumberofElementsGas_; // only includes elments present in gas phase
    data.nSpec = nSpec_;
    data.nReac = nReac_;
    // data.nNASA9coef = nNASA9coef_;
    data.nThbReac = nThbReac_;
    // data.maxTbInReac = maxTbInReac_;
    data.nFallReac = nFallReac_;
    data.nFallPar = nFallPar_;
    // data.maxSpecInReac = maxSpecInReac_;
    data.maxOrdPar = maxOrdPar_;
    data.nRealNuReac = nRealNuReac_;
    data.nRevReac = nRevReac_;
    data.nOrdReac = nOrdReac_;
    data.nPlogReac = nPlogReac_;

    data.jacDim = nSpec_ + 3; /// rho, temperature and pressure

    data.enableRealReacScoef = true;
    {
      const auto tmp = reacScoef_.template view<host_exec_space>();
      for (ordinal_type i = 0; i < nReac_; ++i) {
        const bool flag = (tmp(i) != -1);
        data.enableRealReacScoef &= flag;
      }
    }

    data.Runiv = Runiv_;
    data.Rcal = Rcal_;
    data.Rcgs = Rcgs_;
    data.TthrmMin = TthrmMin_;
    data.TthrmMax = TthrmMax_;

    // data.spec9t = spec9t_.template view<SpT>();
    // data.spec9nrng = spec9nrng_.template view<SpT>();
    // data.sigNu = sigNu_.template view<SpT>();
    data.reacTbdy = reacTbdy_.template view<SpT>();
    data.reacTbno = reacTbno_.template view<SpT>();
    // data.reac_to_Tbdy_index = reac_to_Tbdy_index_.template view<SpT>();
    data.reacPfal = reacPfal_.template view<SpT>();
    data.reacPtype = reacPtype_.template view<SpT>();
    data.reacPlohi = reacPlohi_.template view<SpT>();
    data.reacPspec = reacPspec_.template view<SpT>();
    data.isRev = isRev_.template view<SpT>();
    // data.isDup = isDup_.template view<SpT>();
    data.reacAOrd = reacAOrd_.template view<SpT>();
    // data.reacNrp = reacNrp_.template view<SpT>();
    data.reacNreac = reacNreac_.template view<SpT>();
    data.reacNprod = reacNprod_.template view<SpT>();
    data.reacScoef = reacScoef_.template view<SpT>();
    data.reacRnu = reacRnu_.template view<SpT>();
    data.reacRev = reacRev_.template view<SpT>();
    data.reacHvIdx = reacHvIdx_.template view<SpT>();
    data.reacPlogIdx = reacPlogIdx_.template view<SpT>();
    data.reacPlogPno = reacPlogPno_.template view<SpT>();
    // data.sNion = sNion_.template view<SpT>();
    // data.sCharge = sCharge_.template view<SpT>();
    // data.sTfit = sTfit_.template view<SpT>();
    // data.sPhase = sPhase_.template view<SpT>();

    data.NuIJ = NuIJ_.template view<SpT>();
    data.specTbdIdx = specTbdIdx_.template view<SpT>();
    data.reacNuki = reacNuki_.template view<SpT>();
    data.reacSidx = reacSidx_.template view<SpT>();
    data.specAOidx = specAOidx_.template view<SpT>();
    // data.elemCount = elemCount_.template view<SpT>(); // (nSpec_, nElem_)

    data.sMass = sMass_.template view<SpT>();
    // data.Tlo = Tlo_.template view<SpT>();
    data.Tmi = Tmi_.template view<SpT>();
    // data.Thi = Thi_.template view<SpT>();
    // data.sigRealNu = sigRealNu_.template view<SpT>();
    // data.reacHvPar = reacHvPar_.template view<SpT>();
    data.kc_coeff = kc_coeff_.template view<SpT>();
    // data.eMass = eMass_.template view<SpT>();

    data.RealNuIJ = RealNuIJ_.template view<SpT>();
    data.reacArhenFor = reacArhenFor_.template view<SpT>();
    data.reacArhenRev = reacArhenRev_.template view<SpT>();
    data.specTbdEff = specTbdEff_.template view<SpT>();
    data.reacPpar = reacPpar_.template view<SpT>();
    data.reacRealNuki = reacRealNuki_.template view<SpT>();
    data.specAOval = specAOval_.template view<SpT>();
    data.reacPlogPars = reacPlogPars_.template view<SpT>();

    data.cppol = cppol_.template view<SpT>();
    // data.spec9trng = spec9trng_.template view<SpT>();
    // data.spec9coefs = spec9coefs_.template view<SpT>();
    // data.sNames = sNames_.template view<SpT>();
    data.stoiCoefMatrix = stoiCoefMatrix_.template view<SpT>();

    return data;
  }
   // surface combustion
  template<typename SpT>
  KineticSurfModelConstData<SpT> createConstSurfData()
  {
    KineticSurfModelConstData<SpT> data;

    data.speciesNames = TCsurf_sNames_.template view<SpT>();
    data.nSpec = TCsurf_Nspec_; //
    data.nReac = TCsurf_Nreac_;
    data.sitedensity = TCsurf_siteden_;

    data.cppol = TCsurf_cppol_.template view<SpT>();

    data.Runiv = Runiv_;
    // data.Rcal = Rcal_;
    // data.Rcgs = Rcgs_;
    data.TthrmMin = TCsurf_TthrmMin_;
    data.TthrmMax = TCsurf_TthrmMax_;
    data.Tmi = TCsurf_Tmi_.template view<SpT>();
    data.reacArhenFor = TCsurf_reacArhenFor_.template view<SpT>();
    data.isStick = TCsurf_isStick_.template view<SpT>();
    data.isRev = TCsurf_isRev_.template view<SpT>();

    data.maxSpecInReac = TCsurf_maxSpecInReac_;

    data.reacNreac =
      TCsurf_reacNreac_.template view<SpT>(); //  /* no of reactants only */
    data.reacNprod =
      TCsurf_reacNprod_.template view<SpT>(); // no of products per reaction
    data.reacSsrf =
      TCsurf_reacSsrf_.template view<SpT>(); // gas species: 0 surface specie:1
    data.reacNuki =
      TCsurf_reacNuki_.template view<SpT>(); // Stoichiometric coefficients
    data.reacSidx =
      TCsurf_reacSidx_
        .template view<SpT>(); // specie index in gas list or surface list

    /* determine machine precision parameters (for numerical Jac) */
    const real_type two(2);
    const real_type eps = ats<real_type>::epsilon();
    data.TChem_reltol = sqrt(two * eps); // 1e-6;//
    data.TChem_abstol = data.TChem_reltol;

    data.vki = vski_.template view<SpT>();
    data.vsurfki = vsurfki_.template view<SpT>();

    data.coverageFactor = coverageFactor_.template view<SpT>();

    return data;
  }
  // aerosol chemistry
  template<typename SpT>
  KineticModelNCAR_ConstData<SpT> createNCAR_ConstData()
  {
    KineticModelNCAR_ConstData<SpT> data;
    // forward arrhenius coefficients
    data.speciesNames = sNames_.template view<SpT>();
    data.reacNuki = reacNuki_.template view<SpT>();
    data.reacSidx = reacSidx_.template view<SpT>();
    data.reacArhenFor = reacArhenFor_.template view<SpT>();
    data.reacNreac = reacNreac_.template view<SpT>();
    data.reacNprod = reacNprod_.template view<SpT>();
    data.nSpec = nSpec_;
    data.nReac = nReac_;
    // troe type
    data.reacPpar = reacPpar_.template view<SpT>();
    data.reacPfal = reacPfal_.template view<SpT>();
    // arrhenius pressure parameters
    data.ArrheniusCoef = ArrheniusCoef_.template view<SpT>();
    // species that are assumed constant like tracers
    data.nConstSpec = nConstSpec_;

    return data;
  }

};

/// make an array of kinetic mode to vary kinetic models
static inline
Kokkos::View<KineticModelData*,Kokkos::HostSpace>
cloneKineticModelData(const KineticModelData reference, const int nModels) {
  Kokkos::View<KineticModelData*,Kokkos::HostSpace> r_val("KMD::cloned models", nModels);
  Kokkos::deep_copy(r_val, reference);
  return r_val;
}


template<typename SpT>
static inline
Kokkos::View<KineticModelConstData<SpT>*,SpT>
createKineticModelConstData(Kokkos::View<KineticModelData*,Kokkos::HostSpace> kmds) {
  Kokkos::View<KineticModelConstData<SpT>*,SpT> r_val("KMCD::const data objects", kmds.extent(0));
  auto r_val_host = Kokkos::create_mirror_view(r_val);
  Kokkos::parallel_for
    (Kokkos::RangePolicy<host_exec_space>(0, kmds.extent(0)),
     KOKKOS_LAMBDA(const int i) {
      r_val_host(i) = kmds(i).createConstData<SpT>();
    });
  Kokkos::deep_copy(r_val, r_val_host);
  return r_val;
}

template<typename SpT>
static inline
Kokkos::View<KineticSurfModelConstData<SpT>*,SpT>
createKineticSurfModelConstData(Kokkos::View<KineticModelData*,Kokkos::HostSpace> kmds) {
  Kokkos::View<KineticSurfModelConstData<SpT>*,SpT> r_val("KMCD::const data objects", kmds.extent(0));
  auto r_val_host = Kokkos::create_mirror_view(r_val);
  Kokkos::parallel_for
    (Kokkos::RangePolicy<host_exec_space>(0, kmds.extent(0)),
     KOKKOS_LAMBDA(const int i) {
      r_val_host(i) = kmds(i).createConstSurfData<SpT>();
    });
  Kokkos::deep_copy(r_val, r_val_host);
  return r_val;
}

static inline
void
KineticModelsModifyWithArrheniusForwardParameters(Kokkos::View<KineticModelData*,Kokkos::HostSpace> kmds,
  const std::string& filename, const ordinal_type& nBatch )
{

  std::vector<real_type> valuesFiles;
  {
    std::ifstream file(filename);

    if (file.is_open()) {
      printf("readSample: Reading reaction modification parameters from %s\n", filename.c_str());
      real_type value;
      while (file >> value) {
        valuesFiles.push_back(value);
      }

    } else{
      printf("readSample : Could not open %s -> Abort !\n", filename.c_str());
      exit(1);
    }
    file.close();

  }
  const ordinal_type NTotalItems = valuesFiles.size();
  const ordinal_type NreacWModification = NTotalItems/(3*nBatch + 1);

  ordinal_type_1d_view_host reacIndx("reacIndx", NreacWModification);

  for (ordinal_type ireac = 0; ireac < NreacWModification; ireac++) {
    reacIndx(ireac) = valuesFiles[ireac];
  }

  auto& kmd_at_p = kmds(0);
  ordinal_type_1d_view_host iplogs("iplogs", kmd_at_p.nReac_);

  /// compute iterators
  ordinal_type iplog(0);
  auto reacPlogIdxHost = kmd_at_p.reacPlogIdx_.view_host();
  auto reacPlogPnoHost = kmd_at_p.reacPlogPno_.view_host();

  for (ordinal_type i = 0; i < kmd_at_p.nReac_; ++i) {
    iplogs(i) = iplog;
    iplog += (iplog < kmd_at_p.nPlogReac_) && (i == reacPlogIdxHost(iplog));
  }


  for (int p=0;p<nBatch;++p) {
    /// use reference so that we modify an object in the view
    auto& kmd_at_p = kmds(p);
    kmd_at_p.modifyArrheniusForwardParametersBegin();
    kmd_at_p.modifyArrheniusForwardParametersPLOG_Begin();

    /// this should be some range value that a user want to modify
    {
      for (ordinal_type ireac = 0; ireac < NreacWModification; ireac++) {
        const ordinal_type reac_index = reacIndx[ireac];

        for (ordinal_type i = 0; i < 3; i++) {
          const real_type reference_value = kmd_at_p.getArrheniusForwardParameter(reac_index, i);
          /// do actual modification of the value and set it
          /// for this test, we just set the same value again
          const real_type value = reference_value * valuesFiles[NreacWModification*(p*3+i+1) + ireac];
          kmd_at_p.modifyArrheniusForwardParameter(reacIndx[ireac], i, value);

        }

        //check if reaction is Plog type
        const ordinal_type iplog = iplogs(reac_index);
        const bool plogtest =
          (iplog < kmd_at_p.nPlogReac_) && (reac_index == reacPlogIdxHost(iplog));

        if (plogtest) {
          //loop over plog intervals
          for (ordinal_type j = reacPlogPnoHost(iplog); j < reacPlogPnoHost(iplog + 1);
               ++j) {
              //loop over parameters A, B, E/R
              for (ordinal_type i = 0; i < 3; i++) {
                 const real_type reference_value = kmd_at_p.getArrheniusForwardParameterPLOG(j,i+1); // A
                 const real_type value = reference_value* valuesFiles[NreacWModification*(p*3+i+1) + ireac]; // use same perturbation for all plog intervals
                 kmd_at_p.modifyArrheniusForwardParameterPLOG(j, i+1, value);
               } // end parameter's loop
           }// end plog-intervale's loop
        }//end plog

      } // reactions loop


    }

    kmd_at_p.modifyArrheniusForwardParametersEnd();
    kmd_at_p.modifyArrheniusForwardParametersPLOG_End();


  }

}

#if defined(TCHEM_ENABLE_TPL_YAML_CPP)
static inline
void
KineticModelsModifyWithArrheniusForwardParameters(Kokkos::View<KineticModelData*,Kokkos::HostSpace> kmds,
  const YAML::Node& parameters, const ordinal_type& nBatch )
{


  const auto reacIndx = parameters["reaction_index"];
  const ordinal_type NreacWModification = reacIndx.size();

  for (int p=0;p<nBatch;++p) {
    /// use reference so that we modify an object in the view
    auto& kmd_at_p = kmds(p);
    kmd_at_p.modifyArrheniusForwardParametersBegin();

    auto pre_exponential_factor =
    parameters["modifier_pre_exponential_No_"+std::to_string(p)];
    auto temperature_coefficient_factor =
    parameters["modifier_temperature_coefficient_No_"+std::to_string(p)];
    auto activation_energy_factor =
    parameters["modifier_activation_energy_No_"+std::to_string(p)];

    /// this should be some range value that a user want to modify
    for (ordinal_type ireac = 0; ireac < NreacWModification; ireac++) {

      const ordinal_type reacIndx_ireac = reacIndx[ireac].as<ordinal_type>();
      const real_type pre_exponential = kmd_at_p.getArrheniusForwardParameter(reacIndx_ireac, 0);
      const real_type temperature_coefficient = kmd_at_p.getArrheniusForwardParameter(reacIndx_ireac, 1);
      const real_type activation_energy = kmd_at_p.getArrheniusForwardParameter(reacIndx_ireac, 2);

      kmd_at_p.modifyArrheniusForwardParameter(reacIndx_ireac, 0,
        pre_exponential_factor[ireac].as<real_type>() * pre_exponential);
      kmd_at_p.modifyArrheniusForwardParameter(reacIndx_ireac, 1,
        temperature_coefficient_factor[ireac].as<real_type>() * temperature_coefficient);
      kmd_at_p.modifyArrheniusForwardParameter(reacIndx_ireac, 2,
        activation_energy_factor[ireac].as<real_type>() * activation_energy);

      }
    kmd_at_p.modifyArrheniusForwardParametersEnd();


  }

}



static inline
void
KineticModelsModifyWithArrheniusForwardSurfaceParameters(Kokkos::View<KineticModelData*,Kokkos::HostSpace> kmds,
  const YAML::Node& parameters, const ordinal_type& nBatch )
{

  const auto reacIndx = parameters["reaction_index"];
  const ordinal_type NreacWModification = reacIndx.size();



  for (int p=0;p<nBatch;++p) {
    /// use reference so that we modify an object in the view
    auto& kmd_at_p = kmds(p);
    kmd_at_p.modifyArrheniusForwardSurfaceParametersBegin();

    auto pre_exponential_factor =
    parameters["modifier_pre_exponential_No_"+std::to_string(p)];
    auto temperature_coefficient_factor =
    parameters["modifier_temperature_coefficient_No_"+std::to_string(p)];
    auto activation_energy_factor =
    parameters["modifier_activation_energy_No_"+std::to_string(p)];

    /// this should be some range value that a user want to modify
    for (ordinal_type ireac = 0; ireac < NreacWModification; ireac++) {
      const ordinal_type reacIndx_ireac = reacIndx[ireac].as<ordinal_type>();
      const real_type pre_exponential = kmd_at_p.getArrheniusForwardSurfaceParameter(reacIndx_ireac, 0);
      const real_type temperature_coefficient = kmd_at_p.getArrheniusForwardSurfaceParameter(reacIndx_ireac, 1);
      const real_type activation_energy = kmd_at_p.getArrheniusForwardSurfaceParameter(reacIndx_ireac, 2);

      kmd_at_p.modifyArrheniusForwardSurfaceParameter(reacIndx_ireac, 0,
        pre_exponential_factor[ireac].as<real_type>() * pre_exponential);
      kmd_at_p.modifyArrheniusForwardSurfaceParameter(reacIndx_ireac, 1,
        temperature_coefficient_factor[ireac].as<real_type>() * temperature_coefficient);
      kmd_at_p.modifyArrheniusForwardSurfaceParameter(reacIndx_ireac, 2,
        activation_energy_factor[ireac].as<real_type>() * activation_energy);

    }
    kmd_at_p.modifyArrheniusForwardSurfaceParametersEnd();


  }

}

#endif


} // namespace TChem
#endif
