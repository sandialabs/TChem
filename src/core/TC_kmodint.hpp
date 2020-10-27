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
#ifndef TCkmodintHSeen
#define TCkmodintHSeen

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// This header should never be exposed to users
#include "TChem_Util.hpp"
using namespace TChem;

int
TC_kmodint_(char* mechfile, char* thermofile);

/* Length of filenames */
#define lenfile 100

/* Lengths of character strings */
#define lenstr01 2000
#define lenstr02 200
#define lenstr03 20
#define lenread 512

/* Reallocation factors for element, species, and reaction lists */
#define nelemalloc 1
#define nspecalloc 1
#define nreacalloc 5

/* -------------------------------------------------------------------------
                           Structure definitions
   ------------------------------------------------------------------------- */

/**
 * \brief Entry in the table of periodic elements
 */
typedef struct
{
  char name[LENGTHOFELEMNAME];
  double mass;
} elemtable;

/**
 * \brief Element data
 */
typedef struct
{
  char name[LENGTHOFELEMNAME];
  int hasmass;
  double mass;
} element;

/**
 * \brief Species data
 */
typedef struct
{
  char name[LENGTHOFSPECNAME];
  int hasthermo;
  int hasmass;
  double mass;
  int charge;
  int phase;
  int numofelem;
  int elemindx[NUMBEROFELEMINSPEC];
  int elemcontent[NUMBEROFELEMINSPEC];
  double nasapoltemp[3];
  double nasapolcoefs[14];
  /* new NASA polynomials */
  int Nth9rng;
  double Nth9Temp[2 * NTH9RNGMAX];
  double Nth9coefs[9 * NTH9RNGMAX];
} species;

/**
 * \brief Reaction data
 */
typedef struct
{
  int isdup;    /* Duplicate reactions */
  int isreal;   /* Real stoichiometric coefficients */
  int isrev;    /* Is reversible */
  int isfall;   /* Is pressure dependent */
  int specfall; /* Type of concetration used for pressure dependency */
  int isthrdb;  /* Uses third body */
  int nthrdb;   /* Number of third body coefficients */
  int iswl;     /* Is photon activation */
  int isbal;    /* Is balanced */
  int iscomp;   /* Is complete */
  int inreac;   /* Number of reactants */
  int inprod;   /* Number of products */
  int ismome;   /* Is a momentum-transfer collision */
  int isxsmi;   /* Is a ion momentum-transfer collision cross-section */
  int isford;   /* Arbitrary reaction order - forward coeffs */
  int isrord;   /* Arbitrary reaction order - reverse coeffs */

  int islowset;  /*  LOW  parameters set */
  int ishighset; /*  HIGH parameters set */
  int istroeset; /*  TROE parameters set */
  int isplogset; /*  PLOG parameters set */
  int issriset;  /*  SRI  parameters set */
  int isrevset;  /*  REV  parameters set */
  int isltset;   /*  LT   parameters set */
  int isrltset;  /*  RLT  parameters set */
  int ishvset;   /*  HV   wavelength set */
  int istdepset; /*  TDEP species    set */
  int isexciset; /*  EXCI parameter  set */
  int isjanset;  /*  JAN  parameters set */
  int isfit1set; /*  FIT1 parameters set */

  int spec[2 * NSPECREACMAX];     /* List of species */
  int nuki[2 * NSPECREACMAX];     /* Stoichiometric coefficients (integer) */
  double rnuki[2 * NSPECREACMAX]; /* Stoichiometric coefficients (real) */

  double arhenfor[3]; /* Arrhenius parameters (forward) */
  double arhenrev[3]; /* Arrhenius parameters (reverse) */

  int ithrdb[NTHRDBMAX];    /* Indices of third-body species */
  double rthrdb[NTHRDBMAX]; /* Enhanced efficiencies of third-body species */

  double plog[4 * NPLOGMAX]; /* PLOG parameters */

  double fallpar[8]; /* Parameters for pressure-dependent reactions */
  double ltpar[2];   /* Landau-Teller parameters */
  double rltpar[2];  /* Reverse Landau-Teller parameters */

  double hvpar; /* Radiation wavelength */

  char aunits[4]; /* pre-exponential factor units */
  char eunits[4]; /* activation energy units */

  int tdeppar; /* Species # for temperature dependence */

  double excipar; /* EXCI (energy loss) parameter */

  double optfit[9]; /* Optional rate-fit parameters */

  /* Arbitrary reaction order */
  int arbspec[4 * NSPECREACMAX]; /* List of species for arbitrary reac order */
  double arbnuki[4 * NSPECREACMAX]; /* and list of coefficients */

} reaction;

/* -------------------------------------------------------------------------
                           Elements' functions
   ------------------------------------------------------------------------- */

void
TCKMI_setperiodictable(elemtable* periodictable, int* Natoms, int iflag);

void
TCKMI_checkeleminlist(char* elemname, element* listelem, int* Nelem, int* ipos);

int
TCKMI_getelements(char* linein,
                  char* singleword,
                  element** listelemaddr,
                  int* Nelem,
                  int* Nelemmax,
                  int* iread,
                  int* ierror);

void
TCKMI_resetelemdata(element* currentelem);

int
TCKMI_setelementmass(element* listelem,
                     int* Nelem,
                     elemtable* periodictable,
                     int* Natoms,
                     int* ierror);

/* -------------------------------------------------------------------------
                           Species' functions
   ------------------------------------------------------------------------- */

int
TCKMI_getspecies(char* linein,
                 char* singleword,
                 species** listspecaddr,
                 int* Nspec,
                 int* Nspecmax,
                 int* iread,
                 int* ierror);

void
TCKMI_resetspecdata(species* currentspec);

int
TCKMI_setspecmass(element* listelem,
                  int* Nelem,
                  species* listspec,
                  int* Nspec,
                  int* ierror);

/* -------------------------------------------------------------------------
                           Thermo functions
   ------------------------------------------------------------------------- */
int
TCKMI_checkthermo(species* listspec, int* Nspec);

void
TCKMI_checkspecinlist(char* specname, species* listspec, int* Nspec, int* ipos);

int
TCKMI_getthermo(char* linein,
                char* singleword,
                FILE* mechin,
                FILE* thermoin,
                element* listelem,
                int* Nelem,
                species* listspec,
                int* Nspec,
                double* Tglobal,
                int* ithermo,
                int* iread,
                int* ierror);

/* -------------------------------------------------------------------------
                           Reactions' functions
   ------------------------------------------------------------------------- */
void
TCKMI_resetreacdata(reaction* currentreac, char* aunits, char* eunits);

void
TCKMI_checkunits(char* linein, char* singleword, char* aunits, char* eunits);

int
TCKMI_getreacline(char* linein,
                  char* singleword,
                  species* listspec,
                  int* Nspec,
                  reaction* listreac,
                  int* Nreac,
                  int* ierror);

int
TCKMI_getreacauxl(char* linein,
                  char* singleword,
                  species* listspec,
                  int* Nspec,
                  reaction* listreac,
                  int* Nreac,
                  int* ierror);

int
TCKMI_getreactions(char* linein,
                   char* singleword,
                   species* listspec,
                   int* Nspec,
                   reaction* listreac,
                   int* Nreac,
                   char* aunits,
                   char* eunits,
                   int* ierror);

int
TCKMI_rescalereac(reaction* listreac, int* Nreac);

int
TCKMI_verifyreac(element* listelem,
                 int* Nelem,
                 species* listspec,
                 int* Nspec,
                 reaction* listreac,
                 int* Nreac,
                 int* ierror);

int
TCKMI_kmodsum(element* listelem,
              int* Nelem,
              species* listspec,
              int* Nspec,
              reaction* listreac,
              int* Nreac,
              int* nIonEspec,
              int* electrIndx,
              int* nIonSpec,
              int* maxSpecInReac,
              int* maxTbInReac,
              int* maxOrdPar,
              int* nFallPar,
              int* maxTpRange,
              int* nLtReac,
              int* nRltReac,
              int* nFallReac,
              int* nPlogReac,
              int* nThbReac,
              int* nRevReac,
              int* nHvReac,
              int* nTdepReac,
              int* nJanReac,
              int* nFit1Reac,
              int* nExciReac,
              int* nMomeReac,
              int* nXsmiReac,
              int* nRealNuReac,
              int* nOrdReac,
              int* nNASAinter,
              int* nCpCoef,
              int* nNASAfit,
              int* nArhPar,
              int* nLtPar,
              int* nJanPar,
              int* nFit1Par,
              int* nNASA9coef);
/* -------------------------------------------------------------------------
                           I/O functions
   ------------------------------------------------------------------------- */
int
TCKMI_outform(element* listelem,
              int* Nelem,
              species* listspec,
              int* Nspec,
              reaction* listreac,
              int* Nreac,
              char* aunits,
              char* eunits,
              FILE* fileascii);
int
TCKMI_outunform(element* listelem,
                int* Nelem,
                species* listspec,
                int* Nspec,
                reaction* listreac,
                int* Nreac,
                char* aunits,
                char* eunits,
                FILE* filelist,
                int* ierror);
int
TCKMI_outmath(element* listelem,
              int* Nelem,
              species* listspec,
              int* Nspec,
              reaction* listreac,
              int* Nreac,
              char* aunits,
              char* eunits);

int
saveRectionEquations(species* listspec,
                     reaction* listreac,
                     int* Nreac,
                     FILE* filereacEqn);
/* -------------------------------------------------------------------------
                           Error functions
   ------------------------------------------------------------------------- */
void
TCKMI_errormsg(int ierror);

/* -------------------------------------------------------------------------
                           Character string functions
   ------------------------------------------------------------------------- */
int
TCKMI_elimleads(char* linein);
int
TCKMI_elimends(char* linein);
int
TCKMI_elimspaces(char* linein);
int
TCKMI_elimcomm(char* linein);
int
TCKMI_tab2space(char* linein);
int
TCKMI_extractWordLeft(char* linein, char* oneword);
int
TCKMI_extractWordRight(char* linein, char* oneword);
int
TCKMI_extractWordLeftauxline(char* linein,
                             char* oneword,
                             char* twoword,
                             int* inum,
                             int* ierror);
int
TCKMI_extractWordLeftNoslash(char* linein, char* oneword);

int
TCKMI_extractdouble(char* wordval, double* dvalues, int* inum, int* ierror);

void
TCKMI_wordtoupper(char* linein, char* oneword, int Npos);

void
TCKMI_cleancharstring(char* linein, int* len1);

int
TCKMI_charfixespc(char* singleword, int* len1);

int
TCKMI_checkstrnum(char* singleword, int* len1, int* ierror);

int
TCKMI_findnonnum(char* specname, int* ipos);

#endif
