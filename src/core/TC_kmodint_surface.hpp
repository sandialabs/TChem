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
#ifndef TCkmodintSurfHSeen
#define TCkmodintSurfHSeen

/// This header should never be exposed to users
#include "TChem_Util.hpp"
using namespace TChem;

#include "TC_kmodint.hpp"
// #include "TC_data.h"

/**
 * \brief Species data
 */
typedef struct
{
  char name[LENGTHOFSPECNAME];
  double mass;
  int charge;
  int phase;
  int* elemcontent;
  /* NASA polynomials */
  double nasapoltemp[3];
  double nasapolcoefs[14];
} speciesGas;

/**
 * \brief Species data
 */
typedef struct
{
  int Nelem;
  int Nspec;
  element* infoE;
  speciesGas* infoS;
} infoGasChem;

/**
 * \brief Species data
 */
typedef struct
{

  char name[LENGTHOFSPECNAME];
  int hasmass;
  double mass;
  int charge;
  int phase;
  int elemcontent[200];
  /* NASA polynomials */
  int hasthermo;
  double nasapoltemp[3];
  double nasapolcoefs[14];
} speciesSurf;

/**
 * \brief Surface reaction data
 */
typedef struct
{
  int isdup; /* Duplicate reactions */
  int isstick;
  int isrev;  /* Is reversible */
  int isbal;  /* Is balanced */
  int iscomp; /* Is complete */
  int inreac; /* Number of reactants */
  int inprod; /* Number of products */

  int isrevset; /*  REV  parameters set */
  int ishvset;  /*  HV   wavelength set */

  int spec[2 * NSPECREACMAX]; /* List of species */
  int surf[2 * NSPECREACMAX]; /* List of species flag (surface or not) */
  int nuki[2 * NSPECREACMAX]; /* Stoichiometric coefficients (integer) */

  double arhenfor[3]; /* Arrhenius parameters (forward) */
  double arhenrev[3]; /* Arrhenius parameters (reverse) */

  double hvpar; /* Radiation wavelength */

  char aunits[4]; /* pre-exponential factor units */
  char eunits[4]; /* activation energy units */

} reactionSurf;

int
TC_kmodint_surface_(char* mechfile, char* thermofile, infoGasChem* kmGas);

int
TCKMI_getsite(char* linein,
              char* singleword,
              int Nelem,
              speciesSurf** listspecaddr,
              int* Nspec,
              int* Nspecmax,
              int* iread,
              int* ierror);

void
TC_KMI_resetspecsurfdata(speciesSurf* currentspec, int Nelem);

int
TCKMI_setspecsurfmass(element* listelem,
                      int* Nelem,
                      speciesSurf* listspec,
                      int* Nspec,
                      int* ierror);

int
TC_kmodint_surface_(char* mechfile,
                    int* lmech,
                    char* thermofile,
                    int* lthrm,
                    infoGasChem* kmGas);

void
TCKMI_sitedens(char* linein, char* singleword, double* sden);

int
TCKMI_getthermosurf(char* linein,
                    char* singleword,
                    FILE* mechin,
                    FILE* thermoin,
                    element* listelem,
                    int* Nelem,
                    speciesSurf* listspec,
                    int* Nspec,
                    double* Tglobal,
                    int* ithermo,
                    int* iread,
                    int* ierror);

void
TCKMI_resetreacdatasurf(reactionSurf* currentreac, char* aunits, char* eunits);

int
TCKMI_getreactionssurf(char* linein,
                       char* singleword,
                       speciesGas* listspec,
                       int* Nspec,
                       speciesSurf* listspecSurf,
                       int* NspecSurf,
                       reactionSurf* listreac,
                       int* NreacSurf,
                       char* aunits,
                       char* eunits,
                       int* ierror);

int
TCKMI_getreacauxlsurf(char* linein,
                      char* singleword,
                      speciesSurf* listspecSurf,
                      int* NspecSurf,
                      reactionSurf* listreacSurf,
                      int* NreacSurf,
                      int* ierror);

int
TCKMI_getreaclinesurf(char* linein,
                      char* singleword,
                      speciesGas* listspec,
                      int* Nspec,
                      speciesSurf* listspecSurf,
                      int* NspecSurf,
                      reactionSurf* listreacSurf,
                      int* NreacSurf,
                      int* ierror);

void
TCKMI_checkspecinlistsurf(char* specname,
                          speciesSurf* listspecSurf,
                          int* NspecSurf,
                          int* ipos);
void
TCKMI_checkspecinlistgas(char* specname,
                         speciesGas* listspec,
                         int* Nspec,
                         int* ipos);

int
TCKMI_outformsurf(element* listelem,
                  int* Nelem,
                  speciesGas* listspec,
                  int* Nspec,
                  speciesSurf* listspecSurf,
                  int* NspecSurf,
                  reactionSurf* listreacSurf,
                  int* NreacSurf,
                  char* aunits,
                  char* eunits,
                  FILE* fileascii);

//
int
saveSurfRectionEquations(speciesGas* listspec,
                     speciesSurf* listspecSurf,
                     reactionSurf* listreacSurf,
                     int* NreacSurf,
                     FILE* fileascii);

int
TCKMI_outunformsurf(int* Nelem,
                    double siteden,
                    speciesSurf* listspecSurf,
                    int* NspecSurf,
                    reactionSurf* listreacSurf,
                    int* NreacSurf,
                    char* aunits,
                    char* eunits,
                    FILE* filelist,
                    int* ierror);

int
TCKMI_rescalereacsurf(reactionSurf* listreac, int* Nreac);

#endif
