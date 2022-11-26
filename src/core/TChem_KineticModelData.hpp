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
#include <iostream>
#endif

namespace TChem {
  struct KineticModelData {
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
    ordinal_type nReac_{0};

    ordinal_type_1d_dual_view sCharge_, sTfit_, sPhase_;
    // parameters such ranges and reaction index
    chebyshev_reaction_type_1d_dual_view ChebyshevCoef_;
    // data or coefficients
    real_type_2d_dual_view Chebyshev_data_;
    ordinal_type Chebyshev_max_nrows_{0};

    ordinal_type_1d_dual_view reactionType_;

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

    bool motz_wise_;  

    /* Arrhenius parameters, forward and reverse */
    real_type_2d_dual_view TCsurf_reacArhenFor_;
    /* is reaction a duplicate ? */
    ordinal_type_1d_dual_view TCsurf_isDup_, TCsurf_isStick_;

    ordinal_type_2d_dual_view vski_;
    ordinal_type_2d_dual_view vsurfki_;

    coverage_modification_type_1d_dual_view coverageFactor_;

    ordinal_type_1d_dual_view reaction_No_arbitrary_order_, reacNreac_arbitrary_order_;
    ordinal_type_2d_dual_view reacSidx_arbitrary_order_, reacSsrf_arbitrary_order_;
    real_type_2d_dual_view reacNuki_arbitrary_order_;

    // aerosol chemistry
    arrhenius_reaction_type_1d_dual_view ArrheniusCoef_;
    cmaq_h2o2_type_1d_dual_view CMAQ_H2O2Coef_;
    emission_source_type_1d_dual_view EmissionCoef_;
    ordinal_type  nConstSpec_;
    real_type CONV_PPM_;
    ordinal_type_1d_dual_view convExponent_;

    void syncSurfToDevice();
    void allocateViewsSurf(FILE* errfile);

  public:
      /**
       * constructor that allows use of a yaml file and custom echo and error streams
       * @param mechfile
       * @param echofile
       * @param errfile
       * @param surfechofile
       * @param hasSurface
       */
      KineticModelData(const std::string& mechfile, std::ostream& echofile, std::ostream& errfile, const bool& hasSurface=false);

      /**
       * constructor that allows use of a yaml file with default file echo and error streams
       * @param mechfile
       * @param hasSurface
       */
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
    void initYamlFile(const std::string &mechfile, const bool &hasSurface, std::ostream& echofile, std::ostream& errfile, std::ostream& surfechofile);
    ordinal_type initChemYaml(YAML::Node& doc, const int& gasPhaseIndex, std::ostream& echofile, std::ostream& errfile);
    ordinal_type initChemSurfYaml(YAML::Node& doc, const int& surfacePhaseIndex, std::ostream& echofile);
    ordinal_type initChemNCAR(YAML::Node& doc, std::ostream& echofile);
#endif

    /// create multiple models sharing the data in this model
    kmd_type_1d_view_host clone(const int n_models);

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
  };

  void
  modifyArrheniusForwardParameter(const kmd_type_1d_view_host kmds,
				  const ordinal_type mode,
				  const ordinal_type_1d_view_host & reac_indices,
				  /// (imodel, reac index, value_type)
				  /// imodel - range kmds
				  /// reac index - range reac_indicies
				  /// value_type -
				  ///     0: pre exp , 1: activation energy, 2: temperature coeff
				  const real_type_3d_view_host & factors);

  void
  modifyArrheniusForwardParameter(const kmd_type_1d_view_host kmds,
				  const ordinal_type mode,
				  const std::string& filename);

#if defined(TCHEM_ENABLE_TPL_YAML_CPP)
  void
  modifyArrheniusForwardParameter(const kmd_type_1d_view_host kmds,
				  const ordinal_type mode,
				  const YAML::Node& parameters);
#endif

} // namespace TChem
#endif


#include "TChem_KineticModelGasConstData.hpp"
#include "TChem_KineticModelNCAR_ConstData.hpp"
#include "TChem_KineticModelSurfaceConstData.hpp"
