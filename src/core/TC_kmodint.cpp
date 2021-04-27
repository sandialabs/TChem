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
/*! \file TC_kmodint.c

    \brief Collection of functions used to parse kinetic models from files

    \details

       TC_kmodint - utility used to parse kinetic models
       ----------------------------------------------

       Usage: TC_kmodint_(char *mechfile,int *lmech,char *thermofile,int *lthrm)
       <ul>
       <li> mechfile: file containing the kinetic model
       <li> lmech: length of the character string above (introduced to
                    enable passing of character strings from Fortran to C)
       <li> thermofile: file containing thermodynamic properties (NASA
   polynomials) <li> lmech: length of the character string thermofile
   (introduced to enable passing of character strings from Fortran to C)
       </ul>
       Output:
       <ul>
       <li>kmod.out - ascii file containing kinetic model info formatted
                      for visual inspection
       <li>kmod.list - ascii file containing unformatted data for tchem
       </ul>

       <b> Brief description of kinetic model input format  </b>
       ( For a detailed description of the kinetic model format and keywords
   see: Robert J. Kee, Fran M. Rupley, Ellen Meeks, and James A. Miller
       "CHEMKIN-III:A Fortran Chemical Kinetics Package for the Analysis of
   Gas-phase Chemical and Pllasma Kinetics", Sandia Report, SAND96-8216, (1996)
   )

       \b Elements :
                 - Number of elements is "unlimited"
                 - Elements not present in file periodictable.dat must
                     be followed by their atomic weight, e.g.
                     N+ /14.0010/
                 - Element names are one or two characters long
                 - Any duplicate listing of an element is ignored

       \b Species :
                 - Number of species is "unlimited"
                 - Species need to be formed only of elements declared in the
                     list of elements
                 - Species names are "LENGTHOFSPECNAME" characters long
                 - Any duplicate listing of an species is ignored
                 - A species can contain at most "NUMBEROFELEMINSPEC"
                     distinct elements

       <b> Thermodynamic data </b>:
                 - Data can be provided in the kinetic model file and/or
                     the thermodynamic file;
                 - Currently, only NASA polynomials are accepted; two (2)
                     temperature intervals
                 - Data needs to be provided for all species

       \b Reactions :
                 - Number of reactions is "unlimited"
                 - Pre-exponential factor units MOLES or MOLECULES;
                     default is MOLES
                 - Activation energy units: CAL/MOLE, KCAL/MOLE,
                     JOULES/MOLE, KJOULES/MOLE, KELVINS, eVOLTS ;
                     default is CAL/MOLE. Units are converted to KELVINS
                     if necessary; conversion factors are based on
                     NIST data as of July 2007
                 - maximum number of reactants or products is "NSPECREACMAX"
                 - reactants and products are separated by "<=>" or "="
                     (reversible reactions) or "=>" (irreversible reactions)
                 - species are separated by "+"
                 - three Arrhenius parameters should be given for each reaction
                     in the order : pre-exponential factor, temperature
   exponent, activation energy
                 - reaction lines that are too long can be split on
                     several lines using the character "&" at the end of each
                     line

       <b> Auxiliary reaction info </b> :
                 - Auxiliary data needs to be provided immediately
                     following the reaction to which it corresponds to
                 - Any keywords except DUPLICATE,MOME, and XSMI need
                     to be followed by numerical values enclosed
                     between "/"
                 - Duplicate reactions \n
                       \b DUPLICATE
                 - third-body efficiencies for reactions containing
                     "+M" (not "(+M)" ) as a reactant and/or product:\n
                        speciesname /value/
                     the maximum number of third-body efficiencies is
                     given by "NTHRDBMAX"
                 - pressure-dependent reaction are signaled by the
                     inclusion of "(+M)" as a reactant and/or product
                     or by the inclusion of a particular species, e.g.
                     "(+H2)" as a reactant and/or product. Some of
                     the following parameters are required to describe
                     the pressure dependency:\n
                       \b LOW  /value1 value2 value3/\n
                       \b HIGH /value1 value2 value3/\n
                       \b TROE /value1 value2 value3 value4/ (if value4 is
   ommited then the corresponding term is ommited in the corresponding Troe
   formulation) \n \b SRI  /value1 value2 value3 value4 value5/ (if value4 and
   value5 are ommited then value4=1.0, value5=0.0)
                 - Landau-Teller reactions\n
                       \b LT /value1 value2/ for the forward rate \n
                       \b RLT /value1 value2/ for the reverse rate. If REV is
                           given then RLT is mandatory; if not then RLT is
                           optional
                 - Additional rate fit expressions: \n
                       \b JAN /value1 value2 ... value9/\n
                       \b FIT1 /value1 value2 value3 value4/
                 - Radiation wavelength for reactions containing HV as
                     a reactant and/or product \n
                       \b HV /value1/
                 - Reaction rate dependence on a particular species
                     temperature\n
                       \b TDEP /speciename/
                            <ul>
                             <li> Energy loss parameter
                            </ul>
                       \b EXCI /value1/
                 - Plasma (Ion)  momentum-transfer collision frequency\n
                       \b MOME (\b XSMI)
                 - Reverse reaction Arrhenius parameters\n
                       \b REV /value1 value2 value3/
                 - Change reaction order parameters\n
                       \b FORD /specname value1/ (for forward rate)\n
                       \b RORD /specname value1/ (for reverse rate)
                 - Reaction units for reactions with units different
                     than most of the other reactions\n
                       \b UNITS /unit1 unit2/ \n(the number of keywords between
                       "//" can be one if only one set of units is changed
                       or two if both pre-exponential factor and
                       activation energy are to be modified)

*/

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "TC_kmodint.hpp"

/**
 * \def MAX
 * Maximum of two expressions
 */
#define MAX(A, B) (((A) > (B)) ? (A) : (B))
/**
 * \def MIN
 * Minimum of two expressions
 */
#define MIN(A, B) (((A) < (B)) ? (A) : (B))

int
TCKMI_getthermo9(char* singleword,
                 FILE* thermoin,
                 element* listelem,
                 int* Nelem,
                 species* listspec,
                 int* Nspec,
                 int* ithermo,
                 int* iread,
                 int* ierror);

#define VERBOSE

//#include "TC_getthc9.c"

int TC_7TCoefs = 1, TC_9TCoefs = 0;
char* TCKMI_thf9 = NULL;

void
TC_set7term_()
{
  TC_7TCoefs = 1;
  return;
}
void
TC_unset7term_()
{
  TC_7TCoefs = 0;
  return;
}

void
TC_set9term_()
{
  TC_9TCoefs = 1;
  return;
}
void
TC_unset9term_()
{
  TC_9TCoefs = 0;
  return;
}

void
TC_setthf9_(char* thf9)
{
  TCKMI_thf9 = thf9;
  return;
}
void
TC_unsetthf9_()
{
  TCKMI_thf9 = NULL;
  return;
}

/* ---------------------------Main function----------------------------- */
/**
 * \ingroup init
 * \brief Kinetic model interpretor.
 */
int
TC_kmodint_(char* mechfile, char* thermofile)
{
  /**
   * \param mechfile : name of file containing kinetic model in chemkin format
   * \param lmech    : length of mechfile character string
   * \param thermofile : name of file containing coefficients for NASA
   * polynomials \param lthrm    : length of thermofile character string
   */
  /*
       _                           _  _         _
      | | __ _ __ ___    ___    __| |(_) _ __  | |_
      | |/ /| '_ ` _ \  / _ \  / _` || || '_ \ | __|
      |   < | | | | | || (_) || (_| || || | | || |_
      |_|\_\|_| |_| |_| \___/  \__,_||_||_| |_| \__|

  */

  /* Counters */
  int i, icline;

  int Natoms = 0;
  elemtable* periodictable = 0;

  /* ----------Number of elements, species, reactions----------------- */
  int Nelem, Nelemmax;
  int Nspec, Nspecmax;
  int Nreac, Nreacmax;

  /* -----------Element, species, and reaction data structures-------- */
  element* listelem;
  species* listspec;
  reaction* listreac;

  /* --------Global temperature range (thermo data)----- */
  double Tglobal[3];

  /* ---Pre-exonential factor and activation energy units--- */
  char aunits[lenstr03], eunits[lenstr03];

  /* File I/O */
  FILE *mechin, *thermoin, *thermoin9, *filelist, *fileascii, *filereacEqn;
  char listfile[lenfile], asciifile[lenfile], reactfile[lenfile];

  /* Character strings */
  char linein[lenstr01], linein2[lenstr01], singleword[lenstr01], kwd[lenstr03];

  /* Integer flags */
  int ierror, iread, ithermo, iremove;

#ifdef VERBOSE
  printf("\n");
  printf("       _                           _  _         _      \n");
  printf("      | | __ _ __ ___    ___    __| |(_) _ __  | |_    \n");
  printf("      | |/ /| '_ ` _ \\  / _ \\  / _` || || '_ \\ | __|   \n");
  printf("      |   < | | | | | || (_) || (_| || || | | || |_    \n");
  printf("      |_|\\_\\|_| |_| |_| \\___/  \\__,_||_||_| |_| \\__|   \n");
  printf("                                                       \n\n");
#endif

#ifdef VERBOSE
  printf("Reading kinetic model from : %s\n", mechfile);
  printf("      and thermo data from : %s\n", thermofile);
#endif

  /*--------------------Set periodic table--------------------------- */
  TCKMI_setperiodictable(periodictable, &Natoms, 1);
  periodictable = (elemtable*)malloc(Natoms * sizeof(elemtable));
  TCKMI_setperiodictable(periodictable, &Natoms, 2);

  /* ----------Number of elements, species, reactions----------------- */
  Nelem = 0;
  Nelemmax = nelemalloc;
  Nspec = 0;
  Nspecmax = nspecalloc;
  Nreac = 0;
  Nreacmax = nreacalloc;

  /* -----------Element, species, and reaction data structures-------- */
  listelem = (element*)malloc(Nelemmax * sizeof(listelem[0]));
  listspec = (species*)malloc(Nspecmax * sizeof(listspec[0]));
  listreac = (reaction*)malloc(Nreacmax * sizeof(listreac[0]));

  /* --------Global temperature range (thermo data)----- */
  for (i = 0; i < 3; i++)
    Tglobal[i] = -100.0;

  /* Integer flags */
  ierror = 0;
  iread = 0;
  ithermo = 0;
  iremove = 0;

  /* --------------------Input/Output file---------------------------- */
  strcpy(listfile, "kmod.list");       /* Unformatted ASCII output file */
  strcpy(asciifile, "kmod.out");       /* Formatted   ASCII output file */
  strcpy(reactfile, "kmod.reactions"); /* Formatted   ASCII output file */

  mechin = NULL;
  thermoin = NULL;
  filelist = NULL;
  fileascii = NULL;
  filereacEqn = NULL;
  mechin = fopen(mechfile, "r");
  thermoin = fopen(thermofile, "r");
  filelist = fopen(listfile, "w");
  fileascii = fopen(asciifile, "w");
  filereacEqn = fopen(reactfile, "w");
  if (!mechin) {
    printf("TC_kmodin() : Could not open %s -> Abort !\n", mechfile);
    fflush(stdout);
    exit(1);
  }
  if (!thermoin) {
    printf("TC_kmodin() : Could not open %s -> Abort !\n", thermofile);
    fflush(stdout);
    exit(1);
  }
  if (!filelist) {
    printf("TC_kmodin() : Could not open %s -> Abort !\n", listfile);
    fflush(stdout);
    exit(1);
  }
  if (!fileascii) {
    printf("TC_kmodin() : Could not open %s -> Abort !\n", asciifile);
    fflush(stdout);
    exit(1);
  }

  if (!filereacEqn) {
    printf("TC_kmodin() : Could not open %s -> Abort !\n", reactfile);
    fflush(stdout);
    exit(1);
  }

  if ((TC_9TCoefs == 1) && (TCKMI_thf9 != NULL)) {
    thermoin9 = fopen(TCKMI_thf9, "r");
    if (!thermoin) {
      printf("TC_kmodin() : Could not open %s -> Abort !\n", TCKMI_thf9);
      fflush(stdout);
      exit(1);
    }
  }
  memset(kwd, 0, lenstr03);

  icline = 0;
  ierror = 0;

  /* Start reading the file */
  while (feof(mechin) == 0) {

    int len1, len2;

    icline++;
    fgets(linein, lenread, mechin);
    if (feof(mechin) > 0)
      break;

      //#define DEBUGMSG
#ifdef DEBUGMSG
    printf(
      "Line #%d, String: |%s|, Length: %d \n", icline, linein, strlen(linein));
#endif
    /* replace tab characters and eliminate comments and leading spaces */
    TCKMI_cleancharstring(linein, &len1);
#ifdef DEBUGMSG
    printf(
      "Line #%d, String: |%s|, Length: %d \n", icline, linein, strlen(linein));
#endif
    //#undef DEBUGMSG

    /* Convert first four characters to upper case */
    TCKMI_wordtoupper(linein, kwd, MIN(4, len1));

    /* Set approriate flags */
    iremove = 0;
    if (strncmp(kwd, "ELEM", 4) == 0) {
      /* Found elements keyword, set remove flag */
      iread = 1;
      iremove = 1;
    }

    else if (strncmp(kwd, "SPEC", 4) == 0) {
      /* Found species keyword, set remove flag */
      iread = 2;
      iremove = 1;
    }

    else if (strncmp(kwd, "THER", 4) == 0) {
      /* Found thermo keyword, set remove flag */
      iread = 3;
      iremove = 1;
      /* Update atomic weights for all elements */
      TCKMI_setelementmass(listelem, &Nelem, periodictable, &Natoms, &ierror);
    }

    else if (strncmp(kwd, "REAC", 4) == 0) {
      /* Found reactions keyword */
      iread = 4;
      iremove = 1;
      /* Update atomic weights for all elements */
      TCKMI_setelementmass(listelem, &Nelem, periodictable, &Natoms, &ierror);
      /* Get thermodynamic species for all elements */
      if (TC_7TCoefs == 1)
        TCKMI_getthermo(linein,
                        singleword,
                        mechin,
                        thermoin,
                        listelem,
                        &Nelem,
                        listspec,
                        &Nspec,
                        Tglobal,
                        &ithermo,
                        &iread,
                        &ierror);
      if (TC_9TCoefs == 1)
        TCKMI_getthermo9(singleword,
                         thermoin9,
                         listelem,
                         &Nelem,
                         listspec,
                         &Nspec,
                         &ithermo,
                         &iread,
                         &ierror);
    }

    else if (strncmp(kwd, "END", 3) == 0) {
      if (iread == 4)
        break;
      iread = 0;
    }

    else if ((iread == 0) && (strlen(linein) > 0)) {
      TCKMI_elimleads(linein);
      if (strlen(linein) > 0)
        ierror = 9999;
    }

    if (ierror > 0) {
      TCKMI_errormsg(ierror);
      return (ierror);
    }

    /* Remove leftmost word if needed */
    if (iremove == 1)
      TCKMI_extractWordLeft(linein, singleword);

    if ((iremove == 1) && (iread == 4))
      TCKMI_checkunits(linein, singleword, aunits, eunits);

    /* Route things as approriate */
    if (iread == 1) {
      /* In the element mode */
      len1 = strlen(linein);
      while (len1 > 0) {
        TCKMI_getelements(
          linein, singleword, &listelem, &Nelem, &Nelemmax, &iread, &ierror);
        if (ierror > 0) {
          TCKMI_errormsg(ierror);
          return (ierror);
        }
        len1 = strlen(linein);
      }
    }

    else if (iread == 2) {
      /* In the species mode */
      len1 = strlen(linein);
      while (len1 > 0) {
        TCKMI_getspecies(
          linein, singleword, &listspec, &Nspec, &Nspecmax, &iread, &ierror);
        if (ierror > 0) {
          TCKMI_errormsg(ierror);
          return (ierror);
        }
        len1 = strlen(linein);
      }
    }

    else if (iread == 3)
      /* In the thermo mode */
      TCKMI_getthermo(linein,
                      singleword,
                      mechin,
                      thermoin,
                      listelem,
                      &Nelem,
                      listspec,
                      &Nspec,
                      Tglobal,
                      &ithermo,
                      &iread,
                      &ierror);

    else if (iread == 4) {
      /* In the reaction mode, see if the line is continued */
      len1 = strlen(linein);
      while ((len1 > 0) && (strncmp(&linein[len1 - 1], "&", 1) == 0)) {
        memset(&linein[len1 - 1], 0, 1);
        fgets(linein2, lenread, mechin);
        if (feof(mechin) > 0) {
          ierror = 800;
          break;
        }
        TCKMI_cleancharstring(linein2, &len2);
        if (len1 + len2 > lenstr01) {
          ierror = 810;
          break;
        }
        strcat(linein, linein2);
        memset(linein2, 0, len1);
        len1 = strlen(linein);
      }

      /* loop until there is nothing to interpret on the current line */
      while (len1 > 0) {
        //#define DEBUGMSG
#ifdef DEBUGMSG
        printf("1-Line #%d, String: |%s|, Length: %d \n",
               icline,
               linein,
               strlen(linein));
#endif
        TCKMI_getreactions(linein,
                           singleword,
                           listspec,
                           &Nspec,
                           listreac,
                           &Nreac,
                           aunits,
                           eunits,
                           &ierror);
#ifdef DEBUGMSG
        printf("2-Line #%d, String: |%s|, Length: %d \n",
               icline,
               linein,
               strlen(linein));
#endif
        //#undef DEBUGMSG

        if (ierror > 0)
          break;
        len1 = strlen(linein);
      } /* done while loop for length of linein */

      /* Re-allocate reaction list if needed */
      if (Nreac > Nreacmax - nreacalloc) {
        Nreacmax += nreacalloc;
        listreac = (reaction*)realloc(listreac, Nreacmax * sizeof(listreac[0]));
      }

    } /* end if iread = 4 */

    if (ierror > 0) {
      TCKMI_errormsg(ierror);
      return (ierror);
      ///break; /// statement unreachable
    }

    /* Reset kwd */
    len1 = strlen(kwd);
    memset(kwd, 0, len1);

  } /* Done reading the file */

  if (Nreac == 0) {
    /* Input file has no reaction, make sure that all elements and
       species have everything defined */
    TCKMI_setelementmass(listelem, &Nelem, periodictable, &Natoms, &ierror);
    iread = 4;
    if (TC_7TCoefs == 1)
      TCKMI_getthermo(linein,
                      singleword,
                      mechin,
                      thermoin,
                      listelem,
                      &Nelem,
                      listspec,
                      &Nspec,
                      Tglobal,
                      &ithermo,
                      &iread,
                      &ierror);
    if (TC_9TCoefs == 1)
      TCKMI_getthermo9(singleword,
                       thermoin9,
                       listelem,
                       &Nelem,
                       listspec,
                       &Nspec,
                       &ithermo,
                       &iread,
                       &ierror);
    if (ierror > 0) {
      TCKMI_errormsg(ierror);
      return (ierror);
    }
  }

  /* Verify completness and correctness for each reaction */
  TCKMI_verifyreac(
    listelem, &Nelem, listspec, &Nspec, listreac, &Nreac, &ierror);
  if (ierror > 0) {
    TCKMI_errormsg(ierror);
    return (ierror);
  }

  /* Output to ascii file (formatted) */
  TCKMI_outform(listelem,
                &Nelem,
                listspec,
                &Nspec,
                listreac,
                &Nreac,
                aunits,
                eunits,
                fileascii);

  //
  TCKMI_saveRectionEquations(listspec, listreac, &Nreac, filereacEqn);

  /* Rescale pre-exponential factors and activation energies (if needed) */
  TCKMI_rescalereac(listreac, &Nreac);

  /* Output to unformatted ascii file */
  if (ierror == 0)
    TCKMI_outunform(listelem,
                    &Nelem,
                    listspec,
                    &Nspec,
                    listreac,
                    &Nreac,
                    aunits,
                    eunits,
                    filelist,
                    &ierror);
  if (ierror > 0) {
    TCKMI_errormsg(ierror);
    return (ierror);
  }

  /* Output to mathematica friendly file */
  if (ierror == 0)
    TCKMI_outmath(
      listelem, &Nelem, listspec, &Nspec, listreac, &Nreac, aunits, eunits);

  /* Close all files */
  if (mechin != 0)
    fclose(mechin);
  if (thermoin != 0)
    fclose(thermoin);
  if (filelist != 0)
    fclose(filelist);
  if (fileascii != 0)
    fclose(fileascii);
  if (filereacEqn != 0)
    fclose(filereacEqn);

  /* Garbage collection */
  free(periodictable);
  free(listelem);
  free(listspec);
  free(listreac);

  return (ierror);
}

/* ------------------------------------------------------------------------- */
/**
 *   \brief Read periodic table
 *   <ul>
 *    <li> First line contains two integer values: the total number
 *      of elements and the number of elements listed on each
 *      of the following lines
 *    <li> The following lines (an even number) contain lists of elements
 *      names (two characters separated by one of more spaces, and
 *      elemental masses (separated by spaces):
 *      <ol>
 *         <li> S1 S2 ... S(Nline)
 *         <li> M1 M2 ... M(Nline)
 *         <li> S(Nline+1) S(Nline+2) ....
 *         <li> M(Nline+1) M(Nline+2) ....
 *      </ol>
 *    </ul>
 */
void
TCKMI_setperiodictable(elemtable* periodictable, int* Natoms, int iflag)
{

  /*
    102 10
    H          HE         LI         BE         B          C          N O F NE
    1.00797    4.00260    6.93900    9.01220   10.81100   12.01115   14.00670   15.99940
    18.99840   20.18300 NA         MG         AL         SI         P          S
    CL         AR         K          CA
    22.98980   24.31200   26.98150   28.08600   30.97380   32.06400   35.45300   39.94800
    39.10200   40.08000 SC         TI         V          CR         MN FE CO NI
    CU         ZN
    44.95600   47.90000   50.94200   51.99600   54.93800   55.84700   58.93320   58.71000
    63.54000   65.37000 GA         GE         AS         SE         BR KR RB SR
    Y          ZR
    69.72000   72.59000   74.92160   78.96000   79.90090   83.80000   85.47000   87.62000
    88.90500   91.22000 NB         MO         TC         RU         RH PD AG CD
    IN         SN 92.90600   95.94000   99.00000  101.07000  102.90500 106.40000
    107.87000  112.40000  114.82000  118.69000 SB         TE         I XE CS BA
    LA         CE         PR         ND 121.75000  127.60000  126.90440
    131.30000  132.90500  137.34000  138.91000  140.12000  140.90700  144.24000
    PM         SM         EU         GD         TB         DY         HO ER TM
    YB 145.00000  150.35000  151.96000  157.25000  158.92400  162.50000
    164.93000  167.26000  168.93400  173.04000 LU         HF         TA W RE OS
    IR         PT         AU         HG 174.99700  178.49000  180.94800
    183.85000  186.20000  190.20000  192.20000  195.09000  196.96700  200.59000
    TL         PB         BI         PO         AT         RN         FR RA AC
    TH 204.37000  207.19000  208.98000  210.00000  210.00000  222.00000
    223.00000  226.00000  227.00000  232.03800 PA         U          NP PU AM CM
    BK         CF         ES         FM 231.00000  238.03000  237.00000
    242.00000  243.00000  247.00000  249.00000  251.00000  254.00000  253.00000
    D          E
    2.01410    5.45E-4
   */

  if ((iflag != 1) && (iflag != 2)) {
    printf("Unknown flag in setperiodictable : %d\n", iflag);
    exit(0);
  }

  if (iflag == 1) {
    *Natoms = 102;
  }
  if (iflag == 2) {
    const int natoms = 102;
    const char names[natoms][3] = {
      "H",  "HE", "LI", "BE", "B",  "C",  "N",  "O",  "F",  "NE", "NA", "MG",
      "AL", "SI", "P",  "S",  "CL", "AR", "K",  "CA", "SC", "TI", "V",  "CR",
      "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR",
      "RB", "SR", "Y",  "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
      "IN", "SN", "SB", "TE", "I",  "XE", "CS", "BA", "LA", "CE", "PR", "ND",
      "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF",
      "TA", "W",  "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO",
      "AT", "RN", "FR", "RA", "AC", "TH", "PA", "U",  "NP", "PU", "AM", "CM",
      "BK", "CF", "ES", "FM", "D",  "E "
    };

    const double values[natoms] = {
      1.00797,   4.00260,   6.93900,   9.01220,   10.81100,  12.01115,
      14.00670,  15.99940,  18.99840,  20.18300,  22.98980,  24.31200,
      26.98150,  28.08600,  30.97380,  32.06400,  35.45300,  39.94800,
      39.10200,  40.08000,  44.95600,  47.90000,  50.94200,  51.99600,
      54.93800,  55.84700,  58.93320,  58.71000,  63.54000,  65.37000,
      69.72000,  72.59000,  74.92160,  78.96000,  79.90090,  83.80000,
      85.47000,  87.62000,  88.90500,  91.22000,  92.90600,  95.94000,
      99.00000,  101.07000, 102.90500, 106.40000, 107.87000, 112.40000,
      114.82000, 118.69000, 121.75000, 127.60000, 126.90440, 131.30000,
      132.90500, 137.34000, 138.91000, 140.12000, 140.90700, 144.24000,
      145.00000, 150.35000, 151.96000, 157.25000, 158.92400, 162.50000,
      164.93000, 167.26000, 168.93400, 173.04000, 174.99700, 178.49000,
      180.94800, 183.85000, 186.20000, 190.20000, 192.20000, 195.09000,
      196.96700, 200.59000, 204.37000, 207.19000, 208.98000, 210.00000,
      210.00000, 222.00000, 223.00000, 226.00000, 227.00000, 232.03800,
      231.00000, 238.03000, 237.00000, 242.00000, 243.00000, 247.00000,
      249.00000, 251.00000, 254.00000, 253.00000, 2.01410,   5.45E-4
    };

    /* Second pass through the file, reading the entire table */
    for (int k = 0; k < natoms; ++k) {
      strncpy(periodictable[k].name, names[k], 3);
      periodictable[k].mass = values[k];
    }
  }

  return;
}
/*
      _____ _     _____ __  __ _____ _   _ _____ ____
     | ____| |   | ____|  \/  | ____| \ | |_   _/ ___|
     |  _| | |   |  _| | |\/| |  _| |  \| | | | \___ \
     | |___| |___| |___| |  | | |___| |\  | | |  ___) |
     |_____|_____|_____|_|  |_|_____|_| \_| |_| |____/

*/
/*  ------------------------------------------------------------------------- */
/**
 * \brief Returns index of an element in the list of elements:
 *  - the index goes from 0 to (Nelem-1);
 *  - the value of Nelem is returned if the element is not found
 */
void
TCKMI_checkeleminlist(char* elemname, element* listelem, int* Nelem, int* ipos)
{

  int i;

  (*ipos) = (*Nelem);

  for (i = 0; i < (*Nelem); i++)
    if (strcmp(elemname, listelem[i].name) == 0) {
      (*ipos) = i;
      break;
    }

  return;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Interprets a character string containing element names and
 *  possible their mass.
 */
int
TCKMI_getelements(char* linein,
                  char* singleword,
                  element** listelemaddr,
                  int* Nelem,
                  int* Nelemmax,
                  int* iread,
                  int* ierror)
{

  int i, lenstr;

  /* Exit immediately if the error flag is not zero */
  if (*ierror > 0)
    return (1);

  /* Eliminate leading and ending spaces */
  TCKMI_elimleads(linein);
  TCKMI_elimends(linein);
  lenstr = strlen(linein);

#ifdef DEBUGMSG
  printf("In TCKMI_getelements :\n");
  printf("              ->%s\n", linein);
#endif

  while (lenstr > 0) {

    if (strncmp(linein, "/", 1) == 0) {
      /* Possibly found element weight */

      /* Check if at least one element was found before this */
      if (*Nelem < 1) {
        *ierror = 105;
        return (1);
      }
      /* Check if string contained just the "/" */
      if (lenstr == 1) {
        *ierror = 110; /* string contains and odd number of slashes */
        return (1);
      }
      /* Check for the second slash */
      i = strcspn(&linein[1], "/");
      if (i == lenstr - 1) {
        *ierror = 110;
        return (1);
      }
      /* Found stuff between slashes, transform to number */
      strncpy(singleword, &linein[1], i);
      singleword[i] = 0;
      (*listelemaddr)[(*Nelem) - 1].mass = atof(singleword);
      (*listelemaddr)[(*Nelem) - 1].hasmass = 1;
      /* Then remove the "/.../" by shifting the remaining string to the left */
      memmove(linein, &linein[i + 2], lenstr - i - 2);
      memset(&linein[lenstr - i - 2], 0, i + 2);

#ifdef DEBUGMSG
      printf("-->:: %s\n", singleword);
      printf("-->:: %s\n", linein);
      printf("-->:: %e\n", (*listelemaddr)[*Nelem - 1].mass);
#endif

    } /* end if found "/" */
    else {

      int isduplicate;
      /* Possible found element name */
      TCKMI_extractWordLeft(linein, singleword);

#ifdef DEBUGMSG
      printf("TCKMI_getelements:singleword: %s %d\n", linein, strlen(linein));
      printf("TCKMI_getelements:singleword: %s %d %d\n",
             singleword,
             strlen(singleword),
             *Nelem);
#endif

      /* Check is elements has more than two characters in which case
      only keyword "end" is acceptable */
      if (strlen(singleword) > 2) {
        for (i = 0; i < 3; i++)
          singleword[i] = toupper(singleword[i]);
        if (strncmp(singleword, "END", 3) == 0) {
          *iread = 0;
          return (0);
        } else {
          printf("Element name : %s\n", singleword);
          *ierror = 100;
          return (1);
        }

      } /* end if length of string > 2 */

      /* Convert to upper case */
      for (i = 0; i < (int)strlen(singleword); i++)
        singleword[i] = toupper(singleword[i]);

#ifdef DEBUGMSG
      printf("TCKMI_getelements:singleword: %d\n", strlen(singleword));
#endif

      /* First, check if the element is a duplicate */
      isduplicate = 0;

      for (i = 0; i < (*Nelem); i++)
        if (strcmp((*listelemaddr)[i].name, singleword) == 0) {
          isduplicate = 1;
          printf("Error : Element %s is duplicate\n", singleword);
          fflush(stdout);
          exit(1);
        }

      /* Insert element name */
      if (isduplicate == 0) {
        strcpy((*listelemaddr)[*Nelem].name, singleword);
        TCKMI_resetelemdata(&(*listelemaddr)[*Nelem]);
        *Nelem += 1;
      }
    }

    lenstr = strlen(linein);

    if (*Nelem > (*Nelemmax) - 1) {
      /*
      *ierror = -100;
      return (0) ;
      */
      (*Nelemmax) += 1;
      *listelemaddr = (element*)realloc(*listelemaddr,
                                        (*Nelemmax) * sizeof(*listelemaddr[0]));
    }

#ifdef DEBUGMSG
    printf("Nelem = %d...%s\n", *Nelem, linein);
#endif
  }
  return 0;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Reset data for an element
 */

void
TCKMI_resetelemdata(element* currentelem)
{
  currentelem[0].hasmass = 0; /* Flag for element mass initialization */
  currentelem[0].mass = 0.0;  /* Element mass */

  return;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Set the mass for all entries in the list of elements
 *   based on the values found in the periodic table
 */
int
TCKMI_setelementmass(element* listelem,
                     int* Nelem,
                     elemtable* periodictable,
                     int* Natoms,
                     int* ierror)
{

  int i, j;

  /* Exit if the error flag is not zero */
  if (*ierror > 0)
    return (1);

  for (i = 0; i < (*Nelem); i++) {
    /* Only set mass of the property was not already set */
    if (listelem[i].hasmass == 0) {
      for (j = 0; j < (*Natoms); j++) {
        if (strcmp(listelem[i].name, periodictable[j].name) == 0) {
          listelem[i].mass = periodictable[j].mass;
          listelem[i].hasmass = 1;
          break;
        }
      }
      if (listelem[i].hasmass == 0) {
        printf("Element : %d %s\n", i, listelem[i].name);
        *ierror = 150;
      }
    } /* done if mass was already set */
  }   /* done loop over all elements */

  return (0);
}

/*
       ____  ____  _____ ____ ___ _____ ____
      / ___||  _ \| ____/ ___|_ _| ____/ ___|
      \___ \| |_) |  _|| |    | ||  _| \___ \
       ___) |  __/| |__| |___ | || |___ ___) |
      |____/|_|   |_____\____|___|_____|____/

*/
/* ------------------------------------------------------------------------- */
/**
 * \brief Interprets a character string containing species names
 */
int
TCKMI_getspecies(char* linein,
                 char* singleword,
                 species** listspecaddr,
                 int* Nspec,
                 int* Nspecmax,
                 int* iread,
                 int* ierror)
{

  int i, lenstr;

  /* Exit if the error flag is not zero */
  if (*ierror > 0)
    return (1);

  /* Retrieve species names from a line of characters */
  /* Eliminate leading and ending spaces */
  TCKMI_elimleads(linein);
  TCKMI_elimends(linein);
  lenstr = strlen(linein);

#ifdef DEBUGMSG
  printf("In TCKMI_getspecies :\n");
  printf("          ->%s\n", linein);
#endif

  while (lenstr > 0) {
    int len1, isduplicate;
    if (strncmp(linein, "/", 1) == 0) {
      /* Found illegal character on species line */
      *ierror = 210;
      return (1);
    }

    /* Possible found species name, check if duplicate */
    TCKMI_extractWordLeft(linein, singleword);

    if (strlen(singleword) > LENGTHOFSPECNAME) {
      *ierror = 220;
      return (1);
    }

    len1 = strlen(singleword);
    TCKMI_wordtoupper(singleword, singleword, len1);

    /* 20141108 Added check for END keyword */
    if (strncmp(singleword, "END", 3) == 0) {
      *iread = 0;
      return (0);
    }

    /* Check if duplicate */
    isduplicate = 0;
    i = -1;
    while ((isduplicate == 0) && (i < (*Nspec) - 1)) {
      i++;
      if (strcmp((*listspecaddr)[i].name, singleword) == 0)
        isduplicate = 1;
    }
#ifdef DEBUGMSG
    printf("isduplicate : %d\n", isduplicate);
#endif

    /* Insert element name */
    if (isduplicate == 0) {
      /* TCKMI_resetspecdata(&listspec[*Nspec]) ; */
      strcpy((*listspecaddr)[*Nspec].name, singleword);
      TCKMI_resetspecdata(&(*listspecaddr)[*Nspec]);
      (*Nspec) += 1;
    }

    TCKMI_elimleads(linein);
    lenstr = strlen(linein);

    if ((*Nspec) > ((*Nspecmax) - 1)) {
      /*
      *ierror = -100 ;
      return (0) ;
      */
      (*Nspecmax) += 1;
      *listspecaddr = (species*)realloc(*listspecaddr,
                                        (*Nspecmax) * sizeof(*listspecaddr[0]));
    }
  }

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Reset data for a species
 */
void
TCKMI_resetspecdata(species* currentspec)
{

  int i;

#ifdef DEBUGMSG
  printf("In TCKMI_resetspecdata : %p\n", &currentspec[0]);
#endif

  currentspec[0].hasthermo = 0;
  currentspec[0].hasmass = 0;
  currentspec[0].mass = 0.0;
  currentspec[0].charge = 0;
  currentspec[0].phase = 0;
  currentspec[0].numofelem = 0;

  for (i = 0; i < NUMBEROFELEMINSPEC; i++) {
    currentspec[0].elemindx[i] = 0;
    currentspec[0].elemcontent[i] = 0;
  }

  for (i = 0; i < 3; i++)
    currentspec[0].nasapoltemp[i] = -100.0;
  for (i = 0; i < 14; i++)
    currentspec[0].nasapolcoefs[i] = 0.0;

  return;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Set the mass for all entries in the list of elements
 *   based on the values found in the periodic table
 */
int
TCKMI_setspecmass(element* listelem,
                  int* Nelem,
                  species* listspec,
                  int* Nspec,
                  int* ierror)
{

  int i, j;

  /* Exit if the error flag is not zero */
  if (*ierror > 0)
    return (1);

  for (i = 0; i < (*Nspec); i++) {

    /* Only set mass if the value was not already set */
    if (listspec[i].hasmass == 0) {
      listspec[i].mass = 0;
      for (j = 0; j < listspec[i].numofelem; j++) {

        if (listspec[i].elemindx[j] < (*Nelem))
          listspec[i].mass +=
            listspec[i].elemcontent[j] * listelem[listspec[i].elemindx[j]].mass;
        else {
          *ierror = 240;
          break;
        }
      } /* Done loop over the number of elements in species */
      if (*ierror == 0)
        listspec[i].hasmass = 1;
    } /* Done if statement checking for mass */
  }   /* Done loop over number of species */

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 *  \brief Returns position of a species in the list of species
 *   The index goes from 0 to (Nspec-1); if the species is not found
 *   the value of Nspec is returned
 */
void
TCKMI_checkspecinlist(char* specname, species* listspec, int* Nspec, int* ipos)
{

  int i;

  (*ipos) = (*Nspec);

  for (i = 0; i < (*Nspec); i++)
    if (strcmp(specname, listspec[i].name) == 0) {
      (*ipos) = i;
      break;
    }

  return;
}
/*
             _____ _   _ _____ ____  __  __  ___
            |_   _| | | | ____|  _ \|  \/  |/ _ \
              | | | |_| |  _| | |_) | |\/| | | | |
              | | |  _  | |___|  _ <| |  | | |_| |
              |_| |_| |_|_____|_| \_\_|  |_|\___/

*/
/* ------------------------------------------------------------------------- */
/**
 * \brief Returns 1 if all species have thermodynamic properties set, 0
 * otherwise
 */
int
TCKMI_checkthermo(species* listspec, int* Nspec)
{
  int i;
  int allthermo = 1;
  for (i = 0; i < (*Nspec); i++)
    if (listspec[i].hasthermo == 0) {
      allthermo = 0;
      break;
    }

  return (allthermo);
}

void
TCMI_getnames(std::string& line, std::vector<std::string>& items ){
  std::string delimiter = "+";
  size_t pos = 0;
  while ((pos = line.find(delimiter)) != std::string::npos) {
    items.push_back(line.substr(0, pos));
    line.erase(0, pos + delimiter.length());
  }
    items.push_back(line);
}

void
TCMI_parseString(std::string& line, std::string& delimiter, std::vector<std::string>& items ){
  size_t pos = 0;
  while ((pos = line.find(delimiter)) != std::string::npos) {
    items.push_back(line.substr(0, pos));
    line.erase(0, pos + delimiter.length());
  }
    items.push_back(line);
}


void
TCMI_getReactansAndProductosFromEquation
(std::string& equationOriginal,
int& isRev,
std::map<std::string, real_type>& reactantsMap,
std::map<std::string, real_type>& productsMap
)
{
  std::vector<std::string> reactants_sp;
  std::vector<std::string> products_sp;
  std::string equation = equationOriginal;
  equation.erase(remove_if(equation.begin(),
   equation.end(), isspace), equation.end());

  std::transform(equation.begin(),
  equation.end(),equation.begin(), ::toupper);

  std::map<std::string, double> realCoefMap;

  // order is important here:
  std::string delimiter;
  if (equation.find("<=>") != std::string::npos ) {
      delimiter = "<=>";
  } else if   (equation.find("=>") != std::string::npos ) {
      delimiter = "=>";
      isRev = 0;
  } else if   (equation.find("=") != std::string::npos ) {
      delimiter = "=";
  }

  std::string toErase ="(+M)";
  size_t posErase = equation.find(toErase);
  //reactans
  if (posErase != std::string::npos)
  {
    // std::cout << "before +M!:" << equation << '\n';
    equation.erase(posErase,toErase.length());

    //check one more time
    //products
    posErase = equation.find(toErase);
    if (posErase != std::string::npos) {
      equation.erase(posErase,toErase.length());

    }
    // std::cout << "After +M!:" << equation << '\n';

  }

  // printf("equation %s\n",equation.c_str() );

  const size_t posp = equation.length();
  const size_t pos = equation.find(delimiter);
  std::string reactants = equation.substr(0, pos );

  std::string products = equation.substr(pos + delimiter.length() , posp   );

  // printf("reactants %s\n",reactants.c_str() );
  // printf("products %s\n",products.c_str() );
  bool hasRealCoef(false);
  int ipos = strcspn(&equation[0], ".");
  if (ipos < (int)strlen(&equation[0]))
    hasRealCoef = true;
  // if  (hasRealCoef)
    // printf("equation has real coefficients %d\n",hasRealCoef);

  std::vector<std::string> reactants_list;
  TCMI_getnames(reactants, reactants_list );
  double coeff(0);
  char stoicoeff[LENGTHOFSPECNAME];
  // int  ipos;
  for (size_t k = 0; k < reactants_list.size(); k++) {
    std::string  reactant = reactants_list[k] ;

    // printf("Reactants %s\n",reactant.c_str() );

    if (reactant != "M"){
    TCKMI_findnonnum(&reactant[0],&ipos);
    if (ipos > 0)
    {
      //there is a number in front of the species name
      memset(stoicoeff, 0, LENGTHOFSPECNAME);
      strncpy(stoicoeff, &reactant[0], ipos);
      reactant.erase(0,ipos);
          //check if number is integer or real
      ipos = strcspn(stoicoeff, ".");

      if (ipos < (int)strlen(stoicoeff))
      {
        coeff = atof(stoicoeff);
        reactants_sp.push_back(reactant);
      }
      else
      {
        coeff = atoi(stoicoeff);
        for (int i = 0; i < coeff; i++)
        {
          reactants_sp.push_back(reactant);
        }
      }

      }
    else
    {
      reactants_sp.push_back(reactant);
      coeff =1;
    }
    if (hasRealCoef){
      // realCoef.push_back(coeff);
      realCoefMap.insert(std::pair<std::string, real_type>(reactant, coeff));
          // printf("has real coef %d %f\n",hasRealCoef, coeff );
    }
    }

  }

  std::vector<std::string> products_list;

  TCMI_getnames(products, products_list );
  for (size_t k = 0; k < products_list.size(); k++)
  {
    std::string  product = products_list[k] ;
    // printf("Product %s\n",product.c_str() );
    if (product != "M"){
    TCKMI_findnonnum(&product[0],&ipos);
    if (ipos > 0)
    {
       //there is a number in front of the species name
       memset(stoicoeff, 0, LENGTHOFSPECNAME);
       strncpy(stoicoeff, &product[0], ipos);
       product.erase(0,ipos);

       //check if number is integer or real
       ipos = strcspn(stoicoeff, ".");
       // double coeff(0);
       if (ipos < (int)strlen(stoicoeff))
       {
         coeff = atof(stoicoeff);
         products_sp.push_back(product);
       }
       else
       {
         coeff = atoi(stoicoeff);
         for (int i = 0; i < coeff; i++)
         {
           products_sp.push_back(product);
          }

        }

        }
    else
    {
          products_sp.push_back(product);
          coeff=1;
          // if (hasRealCoef)
          //   realCoef.push_back(1);
        }

    if (hasRealCoef){
      // realCoef.push_back(coeff);
      realCoefMap.insert(std::pair<std::string, double>(product, coeff));
      // printf("has real coef %d %f\n",hasRealCoef, coeff );
    }
    }

  }

  for (auto & sp : products_sp)
  {
    auto result = productsMap.insert(std::pair<std::string, int>(sp, 1));
    if (result.second == false)
        result.first->second++;
  }

  for (auto & sp : reactants_sp)
  {
    auto result = reactantsMap.insert(std::pair<std::string, int>(sp, 1));
    if (result.second == false)
        result.first->second++;
  }
  // if there is a least one real coefficient in the reaction all integer coefficient become zero
  if (hasRealCoef)
  {
    for (auto & reac : reactantsMap )
    {
      reac.second = realCoefMap[reac.first];
      // realCoef.push_back(realCoefMap[reac.first]);
      // printf("sp: %s coef: %e\n",reac.first, realCoefMap[reac.first]  );
    }

    for (auto & prod : productsMap)
    {
     prod.second = realCoefMap[prod.first];
     // realCoef.push_back(realCoefMap[prod.first]);
     // printf("sp: %s coef: %f\n", prod.first.c_str(), realCoefMap[prod.first]  );
    }
  }


}

double
TCMI_unitFactorActivationEnergies(std::string unitsOriginal)
{
  /* Activation energies */
  double factor(1.0);
  std::string units = unitsOriginal;
  std::transform(units.begin(),
  units.end(),units.begin(), ::toupper);


  if (units != "KELV")
  {
  if (units == "CAL/MOL")
    /* Found calories for activation energies, need to rescale */
    factor = CALJO / RUNIV;
    /*    factor = 4.184/8.31451 ; */

  else if (units == "KCAL/MOL")
      /* Found kcalories for activation energies, need to rescale */
      factor = CALJO / RUNIV * 1000.0;

  else if (units == "JOUL/MOL")

      /* Found Joules for activation energies, need to rescale */
      factor = 1.0 / RUNIV;
  //
  else if (units == "J/MOL")

      /* Found Joules for activation energies, need to rescale */
      factor = 1.0 / RUNIV;

  else if (units ==  "KJOU/MOL")

      /* Found kJoules for activation energies, need to rescale */
      factor = 1000.0 / RUNIV;
  //
  else if (units ==  "KJ/MOL")

      /* Found kJoules for activation energies, need to rescale */
      factor = 1000.0 / RUNIV;

    else if (units == "EVOL/MOL")

      /* Found electron Volts for activation energies, need to rescale */
      factor = EVOLT / KBOLT;

    else {
      printf("!!! Found unknown activation energy units for reaction ");
      return 10;
    }
    }
    return factor;

}
/* ------------------------------------------------------------------------- */
/**
 * \brief Reads thermodynamic properties (NASA polynomials) from the
 *        mechanism input file or from a separate file
 */
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
                int* ierror)
{

  char linein1[200], linein2[200], linein3[200], linein4[200];
  double dtemp;
  int itemp;
  FILE* thermofile;
  int i, len1;

  /* Exit if the error flag is not zero */
  if (*ierror > 0)
    return (1);

  /* Exit if value of iread flag is not appropriate (3 or 4) */
  if ((*iread != 3) && (*iread != 4)) {
    *ierror = 300;
    return (1);
  }

  /* Check if all species have the thermodynamic properties set already */
  if (TCKMI_checkthermo(listspec, Nspec) == 1)
    return (0);

  /* Check if the thermodynamic properies are to be read from mechanism
  input file  */
  if ((*ithermo == 1) && (*iread == 4)) {
    /* previously encountered thermo all and not all species were
    provided thermodynamic properties in the input file */
    *ierror = 310;
    return (1);
  }

  /* Set up the input file */
  thermofile = NULL;
  if (*iread == 3)
    thermofile = mechin;
  else if (*iread == 4)
    thermofile = thermoin;

  /* Check if file pointer is null */
  if (thermofile == NULL) {
    *ierror = 320;
    return (1);
  }

  /* If reading from mechanism input see if "thermo all" */
  if (*iread == 3) {
    TCKMI_wordtoupper(linein, singleword, 3);
    /* If ALL is encountered need to read three temp values from
    the next line */
    if (strncmp(singleword, "ALL", 3) == 0) {
      if (feof(thermofile) > 0) {
        *ierror = 330;
        return (1);
      }

      /* Reading the next line */
      fgets(linein, 200, thermofile);
      if (feof(thermofile) > 0) {
        *ierror = 332;
        return 1;
      }
      /* replace tab characters and eliminate comments and leading spaces */
      TCKMI_cleancharstring(linein, &len1);
      if (len1 == 0) {
        *ierror = 334;
        return 1;
      }
      i = 0;
      while (i < 3) {
        i++;
        TCKMI_extractWordRight(linein, singleword);
        TCKMI_checkstrnum(singleword, &len1, ierror);
        if (*ierror > 0)
          return 1;
        if (len1 == 0) {
          *ierror = 336;
          return 1;
        }
        Tglobal[3 - i] = atof(singleword);
      } /* done loop over temperature values */
    }   /* done if for "ALL" keyword */
  } /* done with the if section corresponding to Tglobal from mechanism input */
  else {
    if (feof(thermofile) > 0) {
      *ierror = 338;
      return 1;
    }
    len1 = 0;
    while (len1 == 0) {
      /* Reading the first line in the file */
      fgets(linein1, 200, thermofile);
      if (feof(thermofile) > 0) {
        *ierror = 340;
        return 1;
      }
      TCKMI_cleancharstring(linein1, &len1);
    }

    len1 = 0;
    while (len1 == 0) {

      /* Reading the second line in the file, hopefully the temperature range */
      fgets(linein1, 200, thermofile);
      if (feof(thermofile) > 0) {
        *ierror = 342;
        return 1;
      }
      TCKMI_cleancharstring(linein1, &len1);
    }
    /* replace tab characters and eliminate comments and leading spaces */
#ifdef DEBUGMSG
    printf("|%s|\n", linein1);
#endif
    TCKMI_cleancharstring(linein1, &len1);
    if (len1 == 0) {
      *ierror = 344;
      return 1;
    }

    i = 0;

    while (i < 3) {
      i++;
      TCKMI_extractWordRight(linein1, singleword);

      TCKMI_checkstrnum(singleword, &len1, ierror);

      if (*ierror > 0)
        return 1;

      if (len1 == 0) {
        *ierror = 346;
        return 1;
      }

      Tglobal[3 - i] = atof(singleword);

    } /* done loop over temperature values */

  } /* done with the if section corresponding to Tglobal from thermo input */

#ifdef DEBUGMSG
  printf("Trange: %e %e %e\n", Tglobal[0], Tglobal[1], Tglobal[2]);
#endif
  /* Update Trange.mid for the species that need it */
  for (i = 0; i < (*Nspec); i++)
    if (listspec[i].nasapoltemp[2] < 0.0)
      listspec[i].nasapoltemp[2] = Tglobal[1];

  /* Start reading properties */
  while (feof(thermofile) == 0) {

    int ipos, ipos1;
    /* Clear memory locations */
    memset(linein1, 0, 200);
    memset(linein2, 0, 200);
    memset(linein3, 0, 200);
    memset(linein4, 0, 200);

    len1 = 0;
    while ((len1 == 0) && (feof(thermofile) == 0)) {
      fgets(linein1, 200, thermofile);
      TCKMI_cleancharstring(linein1, &len1);
    }

    /* If end of file exit */
    if (feof(thermofile) > 0) {
      if (len1 == 0)
        return 0;
      return 1;
    }

    /* Test for keyword */
    strncpy(singleword, linein1, 4);
    TCKMI_wordtoupper(singleword, singleword, 4);

    /* If "END" is encountered */
    if (strncmp(singleword, "END", 3) == 0) {
      if (*iread == 3)
        (*iread) = 0;
      return 0;
    }

    if ((strncmp(singleword, "REAC", 4) == 0) && (*iread == 3)) {
      strncpy(linein, linein1, 200);
      TCKMI_extractWordLeft(linein, singleword);
      (*iread) = 4;
      return 0;
    }

    if (feof(thermofile) > 0)
      *ierror = 360;
    if (*ierror == 0)
      fgets(linein2, 200, thermofile);
    /* TCKMI_cleancharstring(linein2,&len1) ; */
    if (feof(thermofile) > 0)
      *ierror = 360;
    if (*ierror == 0)
      fgets(linein3, 200, thermofile);
    /* TCKMI_cleancharstring(linein3,&len1) ; */
    if (feof(thermofile) > 0)
      *ierror = 360;
    if (*ierror == 0)
      fgets(linein4, 200, thermofile);
    /* TCKMI_cleancharstring(linein4,&len1) ; */

    if (*ierror > 0)
      return (1);

    /* Get species name and convert it to uppercase */
    len1 = strcspn(linein1, " ");
    strncpy(singleword, linein1, len1);
    singleword[len1] = 0;
    TCKMI_cleancharstring(singleword, &len1);
    TCKMI_wordtoupper(singleword, singleword, len1);

    /* Check if species name is not null */
    if (len1 == 0) {
      *ierror = 370;
      return 1;
    }

    /* Check if species is of interest */
    TCKMI_checkspecinlist(singleword, listspec, Nspec, &ipos);

    if (ipos < (*Nspec)) {
      /* Found species of interest, check if species thermodata was initialized
       * before */
      if (listspec[ipos].hasthermo == 0) {
        int j;
        double polcoeff[5];
        char* currentline;

        /* Initialize temperature ranges */
#ifdef DEBUGMSG
        printf("found species %18s Trange: ", listspec[ipos].name);
#endif
        for (i = 0; i < 3; i++) {
          itemp = 45 + i * 10;
          len1 = 10;
          if (i == 2)
            len1 = 8;
          strncpy(singleword, &linein1[itemp], len1);
          singleword[len1] = 0;
          TCKMI_checkstrnum(singleword, &len1, ierror);

          if (*ierror > 0)
            return 1;

          if (len1 > 0) {
            dtemp = atof(singleword);
            if (dtemp > 0.0)
              listspec[ipos].nasapoltemp[i] = dtemp;
          }
#ifdef DEBUGMSG
          printf("%e ", listspec[ipos].nasapoltemp[i]);
#endif
        }
#ifdef DEBUGMSG
        printf("\n");
#endif

        /* Done with temperature range, check element content */
        for (i = 0; i < 5; i++) {
          itemp = 24 + i * 5;
          if (i == 4)
            itemp = 73;

          /* Extract element name */
          strncpy(singleword, &linein1[itemp], 2);
          singleword[2] = 0;
          if ((isdigit(singleword[0]) > 0) || (isdigit(singleword[0]) > 0))
            break;
          TCKMI_cleancharstring(singleword, &len1);

          /* Break from loop if element position is blank */
          if (len1 == 0)
            break;

          /* Determine position in the list of elements */
          TCKMI_wordtoupper(singleword, singleword, len1);
          TCKMI_checkeleminlist(singleword, listelem, Nelem, &ipos1);
          if (ipos1 == (*Nelem)) {
            *ierror = 380;
            return 1;
          }

          listspec[ipos].elemindx[i] = ipos1;

          /* Good element, find numbers */
          strncpy(singleword, &linein1[itemp + 2], 3);
          singleword[3] = 0;
          TCKMI_checkstrnum(singleword, &len1, ierror);
          if (*ierror > 0)
            return 1;
          itemp = (int)atof(singleword);

          listspec[ipos].elemcontent[i] = itemp;

          /* See if charge needs to be updated */
          if (strncmp(listelem[ipos1].name, "E", 1) == 0) {
            listspec[ipos].charge += listspec[ipos].elemcontent[i] * (-1);
          }

          /* Update mass */
          listspec[ipos].mass +=
            listelem[ipos1].mass * listspec[ipos].elemcontent[i];
        }

        listspec[ipos].hasmass = 1;
        listspec[ipos].numofelem = i;

        /* Get species phase */
        itemp = 44;
        if ((strncmp(&linein[itemp], "l", 1) == 0) ||
            (strncmp(&linein[itemp], "L", 1) == 0))
          listspec[ipos].phase = 1;
        else if ((strncmp(&linein[itemp], "s", 1) == 0) ||
                 (strncmp(&linein[itemp], "S", 1) == 0))
          listspec[ipos].phase = -1;

        /*
          Done with element content and charge, check polynomial coefficients
          Lines 2,3,4
        */
        for (i = 0; i < 3; i++) {

          int nummax = 5;
          /* Set current line */
          if (i == 0)
            currentline = linein2;
          else if (i == 1)
            currentline = linein3;
          else
            currentline = linein4;

          if (i == 2)
            nummax = 4;

          // printf("%d:%s\n",i,currentline);
          for (j = 0; j < nummax; j++) {
            itemp = j * 15;
            strncpy(singleword, &currentline[itemp], 15);
            singleword[15] = 0;
            len1 = 15;
            TCKMI_charfixespc(singleword, &len1); /* 20111101 */
            TCKMI_cleancharstring(singleword, &len1);
            TCKMI_checkstrnum(singleword, &len1, ierror);
            if ((len1 == 0) || (*ierror > 0)) {
              *ierror = 390;
              return 1;
            }
            polcoeff[j] = atof(singleword);
          }

          if (i == 0) {
            for (j = 0; j < nummax; j++)
              listspec[ipos].nasapolcoefs[7 + j] = polcoeff[j];
          } else if (i == 1) {
            for (j = 0; j < 2; j++)
              listspec[ipos].nasapolcoefs[12 + j] = polcoeff[j];
            for (j = 0; j < 3; j++)
              listspec[ipos].nasapolcoefs[j] = polcoeff[2 + j];
          } else {
            for (j = 0; j < nummax; j++)
              listspec[ipos].nasapolcoefs[3 + j] = polcoeff[j];
          }

        } /* Done with for over lines 2,3,4 */

        listspec[ipos].hasthermo = 1;

      } /* Done with if the species had thermo data already initialized */

    } /* Done with if the species is of interest */
  }
  return 0;
}
/*
       ____  _____    _    ____ _____ ___ ___  _   _ ____
      |  _ \| ____|  / \  / ___|_   _|_ _/ _ \| \ | / ___|
      | |_) |  _|   / _ \| |     | |  | | | | |  \| \___ \
      |  _ <| |___ / ___ \ |___  | |  | | |_| | |\  |___) |
      |_| \_\_____/_/   \_\____| |_| |___\___/|_| \_|____/

*/
/* ------------------------------------------------------------------------- */
/**
 * \brief Resets the current entry in the list of reactions
 */
void
TCKMI_resetreacdata(reaction* currentreac, char* aunits, char* eunits)
{

  int i;

  currentreac[0].isdup = -2;
  currentreac[0].isreal = 0;
  currentreac[0].isrev = 0;
  currentreac[0].isfall = 0;
  currentreac[0].specfall = 0;
  currentreac[0].isthrdb = 0;
  currentreac[0].nthrdb = 0;
  currentreac[0].iswl = 0;
  currentreac[0].isbal = 0;
  currentreac[0].iscomp = 0;
  currentreac[0].inreac = 0;
  currentreac[0].inprod = 0;
  currentreac[0].ismome = 0;
  currentreac[0].isxsmi = 0;
  currentreac[0].isford = 0;
  currentreac[0].isrord = 0;
  currentreac[0].islowset = 0;
  currentreac[0].ishighset = 0;
  currentreac[0].istroeset = 0;
  currentreac[0].isplogset = 0;
  currentreac[0].issriset = 0;
  currentreac[0].isrevset = 0;
  currentreac[0].isltset = 0;
  currentreac[0].isrltset = 0;
  currentreac[0].ishvset = 0;
  currentreac[0].istdepset = 0;
  currentreac[0].isexciset = 0;
  currentreac[0].isjanset = 0;
  currentreac[0].isfit1set = 0;

  for (i = 0; i < 2 * NSPECREACMAX; i++)
    currentreac[0].spec[i] = -1;
  for (i = 0; i < 2 * NSPECREACMAX; i++)
    currentreac[0].nuki[i] = 0;
  for (i = 0; i < 2 * NSPECREACMAX; i++)
    currentreac[0].rnuki[i] = 0.0;

  for (i = 0; i < 3; i++)
    currentreac[0].arhenfor[i] = 0.0;
  for (i = 0; i < 3; i++)
    currentreac[0].arhenrev[i] = 0.0;

  for (i = 0; i < NTHRDBMAX; i++)
    currentreac[0].ithrdb[i] = -1;
  for (i = 0; i < NTHRDBMAX; i++)
    currentreac[0].rthrdb[i] = -1.0;
  for (i = 0; i < 8; i++)
    currentreac[0].fallpar[i] = 0.0;

  for (i = 0; i < 2; i++)
    currentreac[0].ltpar[i] = 0.0;
  for (i = 0; i < 2; i++)
    currentreac[0].rltpar[i] = 0.0;

  currentreac[0].hvpar = 0.0;

  strncpy(currentreac[0].aunits, aunits, 4);
  strncpy(currentreac[0].eunits, eunits, 4);

  currentreac[0].tdeppar = -1;
  currentreac[0].excipar = 0;

  for (i = 0; i < 9; i++)
    currentreac[0].optfit[i] = 0.0;

  for (i = 0; i < 4 * NSPECREACMAX; i++)
    currentreac[0].arbspec[i] = -1;
  for (i = 0; i < 4 * NSPECREACMAX; i++)
    currentreac[0].arbnuki[i] = 0;

  return;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Sets units for the pre-exponential factor and for the activation
 *   energy
 */
void
TCKMI_checkunits(char* linein, char* singleword, char* aunits, char* eunits)
{
  int len1;

  /* Reset memory locations */
  memset(aunits, 0, 10);
  memset(eunits, 0, 10);
  memset(singleword, 0, 10);

  TCKMI_elimleads(linein);

  len1 = strlen(linein);

  while (len1 > 0) {

    strncpy(singleword, linein, 5);
    TCKMI_wordtoupper(singleword, singleword, 5);

    if (strlen(eunits) == 0) {

      if (strncmp(singleword, "CAL/", 4) == 0)
        strncpy(eunits, singleword, 4);
      else if (strncmp(singleword, "KCAL", 4) == 0)
        strncpy(eunits, singleword, 4);
      else if (strncmp(singleword, "JOUL", 4) == 0)
        strncpy(eunits, singleword, 4);
      else if (strncmp(singleword, "KJOU", 4) == 0)
        strncpy(eunits, singleword, 4);
      else if (strncmp(singleword, "KELV", 4) == 0)
        strncpy(eunits, singleword, 4);
      else if (strncmp(singleword, "EVOL", 4) == 0)
        strncpy(eunits, singleword, 4);
    }
    if (strlen(aunits) == 0) {

      if (strncmp(singleword, "MOLE", 4) == 0) {
        if (strncmp(&singleword[4], "C", 1) == 0)
          strncpy(aunits, "MOLC", 4);
        else
          strncpy(aunits, singleword, 4);
      }
    }

    TCKMI_extractWordLeftNoslash(linein, singleword);
    len1 = strlen(linein);
  }

  if (strlen(eunits) == 0)
    strncpy(eunits, "CAL/", 4);

  if (strlen(aunits) == 0)
    strncpy(aunits, "MOLE", 4);

#ifdef DEBUGMSG
  printf("In check units : %s  %s\n", aunits, eunits);
#endif

  return;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Interprets a character string containing reaction description
 *   (equation + forward Arrhenius parameters)
 */
int
TCKMI_getreacline(char* linein,
                  char* singleword,
                  species* listspec,
                  int* Nspec,
                  reaction* listreac,
                  int* Nreac,
                  int* ierror)
{

  char specname[LENGTHOFSPECNAME], stoicoeff[LENGTHOFSPECNAME];
  char reac[lenstr02], prod[lenstr02];
  char *reacprod, *reacfall, *prodfall;
  int ithrdb;
  int ipos = 0, ipos1 = 0, ipos2 = 0;
  int i, len1;

  int *aplus, iplus;

  int j;
  int ireacthrdb = -2;
  int iprodthrdb = -2;

  /* Return immediately if error flag is not zero */
  if (*ierror > 0)
    return 1;

    //#define DEBUGMSG
#ifdef DEBUGMSG
  printf("%d |%s|\n", i, linein);
#endif
  //#undef DEBUGMSG

  /* Extract forward Arrhenius coefficients */
  for (i = 0; i < 3; i++) {

    TCKMI_extractWordRight(linein, singleword);
    TCKMI_checkstrnum(singleword, &len1, ierror);

    //#define DEBUGMSG
#ifdef DEBUGMSG
    printf("%d |%s|\n", i, linein);
    printf("%d |%s|\n", i, singleword);
#endif
    //#undef DEBUGMSG

    if ((len1 == 0) || (*ierror > 0)) {
      *ierror = 610;
      return 1;
    }
    listreac[*Nreac].arhenfor[2 - i] = atof(singleword);
  }

  /* Eliminate spaces from the rest of the string */
  TCKMI_elimspaces(linein);

  /* Check for delimiters between reactants and products */
  len1 = strlen(linein);
  if (len1 < 3) {
    printf("|%s|\n", linein);
    *ierror = 620;
    return 1;
  }

  for (i = 0; i < len1; i++) {
    if (strncmp(&linein[i], "<=>", 3) == 0) {
      ipos = i;
      ipos1 = i + 2;
      listreac[*Nreac].isrev = 1;
    } else if (strncmp(&linein[i], "=>", 2) == 0) {
      ipos = i;
      ipos1 = i + 1;
      listreac[*Nreac].isrev = -1;
    } else if ((i > 0) && (strncmp(&linein[i], "=", 1) == 0) &&
               (strncmp(&linein[i - 1], "=", 1) != 0)) {
      ipos = i;
      ipos1 = i;
      listreac[*Nreac].isrev = 1;
    }
    if (ipos > 0)
      break;

  } /* Done with the for checking for delimiters */

  if (ipos == 0) {
    *ierror = 630;
    return 1;
  }

#ifdef DEBUGMSG
  printf("---> %d  %d %d\n", listreac[*Nreac].isrev, ipos, ipos1);
#endif

  /* split into reactants and products */
  memset(reac, 0, lenstr02);
  memset(prod, 0, lenstr02);
  strncpy(reac, linein, ipos);
  strncpy(prod, &linein[ipos1 + 1], len1 - ipos1 - 1);

  /* Check if both reactants and products are present */
  if ((strlen(reac) == 0) || (strlen(prod) == 0)) {
    *ierror = 640;
    return 1;
  }
#ifdef DEBUGMSG
  printf("---> %d  |%s| |%s|\n", listreac[*Nreac].isrev, reac, prod);
#endif
  /* Check for fall-of reactions */
  reacfall = strstr(reac, "(+");
  prodfall = strstr(prod, "(+");
  if ((reacfall != 0) || (prodfall != 0)) {
    /* Possible found fall-off reaction, investigate */
    for (j = 0; j < 2; j++) {
      reacprod = reac;
      if (j == 1)
        reacprod = prod;
      ithrdb = -2;

      len1 = strlen(reacprod);
      i = -1;
      while (i < len1 - 2) {
        i++;
        if (strncmp(&reacprod[i], "(+", 2) == 0) {
          /* found location of starting paranthesis */
          ipos = i;

          /* check if enough space for the other info */
          if (ipos > len1 - 3) {
            *ierror = 650;
            return 1;
          }

          /* determine the closing paranthesis */
          ipos1 = strcspn(&reacprod[ipos + 2], ")");
          ipos1 += ipos + 2;

          /* check if ) was actually found */
          if (ipos1 >= len1) {
            *ierror = 650;
            return 1;
          }

          /* check if (+) was found */
          if (ipos1 == ipos + 2) {
            *ierror = 660;
            return 1;
          }

          /* See which is the third body */
          if (strncmp(&reacprod[ipos + 2], "M", ipos1 - ipos - 2) == 0) {
            if (ithrdb != -2) {
              *ierror = 670;
              return 1;
            }
            ithrdb = -1;
          } else {
            memset(specname, 0, LENGTHOFSPECNAME);
            strncpy(specname, &reacprod[ipos + 2], ipos1 - ipos - 2);
            TCKMI_checkspecinlist(specname, listspec, Nspec, &ipos2);

            /* Check if species exist */
            if (ipos2 == (*Nspec)) {
              *ierror = 680;
              return 1;
            }

            if (ithrdb != -2) {
              *ierror = 690;
              return 1;
            }
            ithrdb = ipos2;

          } /* Done if over which is the third body */

          if (ithrdb > -2) {

            /* Eliminate the fall-off indicator */
            if (len1 - ipos1 - 1 > 0) {
              memmove(&reacprod[ipos], &reacprod[ipos1 + 1], len1 - ipos1 - 1);
              memset(&reacprod[len1 - ipos1 - 1 + ipos], 0, ipos1 + 1 - ipos);
            } else {
              memset(&reacprod[ipos], 0, ipos1 + 1 - ipos);
            }
            len1 = strlen(reacprod);
            if (j == 0)
              ireacthrdb = ithrdb;
            else
              iprodthrdb = ithrdb;
          }

        } /* Done if statement for location of (+ */

      } /* End while statement for string sweep */

    } /* End loop for reactants and products */

#ifdef DEBUGMSG
    printf(
      "---> %d  |%s| |%s| %d %d\n", *Nreac, reac, prod, ireacthrdb, iprodthrdb);
#endif

    /* Check what type of third body */
    if ((ireacthrdb != -2) || (iprodthrdb != -2)) {
      if ((ireacthrdb < 0) && (iprodthrdb < 0)) {
        /* Generic third body */
        listreac[*Nreac].isfall = 1;
        listreac[*Nreac].specfall = -1;

        listreac[*Nreac].isthrdb = 1;

      } else if (ireacthrdb == iprodthrdb) {
        /* Third-body is a specific species */
        listreac[*Nreac].isfall = 1;
        listreac[*Nreac].specfall = iprodthrdb;
      } else {
        printf("Found discrepancy in reaction %d: third body species different "
               "on reactants/products sides: %d vs %d\n",
               *Nreac + 1,
               ireacthrdb,
               iprodthrdb);
        exit(1);
      }

    } /* Done with the if identifying the third body */

  } /* Done if fall-off indicator found */

  /* Check if reactant and product strings are still valid */
  if ((strlen(reac) == 0) || (strlen(prod) == 0)) {
    *ierror = 605;
    return 1;
  }

  /* Check if the reaction has real coeffs -> look for decimal points */
  /* make all coef real, I changed the type for int coef to real OD
   There was a differences in TChem C version between int and real coeffs in how we compute power,
   however, we do not have this difference in TChem++, thus, we decided to delete the int coef.
  */
  ipos = strcspn(reac, ".");
  if (ipos < (int)strlen(reac))
    listreac[*Nreac].isreal = 0;
  ipos = strcspn(prod, ".");
  if (ipos < (int)strlen(prod))
    listreac[*Nreac].isreal = 0;

  /* Start looking for species in the strings of reactants and products */
  for (j = 0; j < 2; j++) {
    int icheckTB = 0, icheckWL = 0;
    reacprod = reac;
    if (j == 1)
      reacprod = prod;

    len1 = strlen(reacprod);
    aplus = (int*)malloc(len1 * sizeof(int));
    iplus = 0;

    /* Identify the "+" locations */
    for (i = 0; i < len1; i++)
      if (strncmp(&reacprod[i], "+", 1) == 0) {
        aplus[iplus] = i;
        iplus++;
      }

    /* Check if any "+"'es are adjacent, if yes, eliminate the ones at the left
     */
    i = 1;
    while (i < iplus - 1) {
      if (aplus[i] - aplus[i - 1] == 1) {
        for (j = i - 1; j < iplus - 1; j++)
          aplus[i] = aplus[i + 1];
        iplus--;
        i--;
      }
      i++;
    }

    /* Check if any "+"'es appear at the end */
    while ((iplus > 0) && (aplus[iplus - 1] == len1 - 1))
      iplus--;

#ifdef DEBUGMSG
    printf("--->%d %d\n", j, iplus);
#endif

    /* Start identifying species */
    for (i = 0; i < iplus + 1; i++) {

      /* Checking species by species */

      int ispec1, ispec2;
      len1 = strlen(reacprod);
      if ((i == 0) && (iplus == 0)) {
        ispec1 = 0;
        ispec2 = len1;
      } else if ((i == 0) && (iplus > 0)) {
        ispec1 = 0;
        ispec2 = aplus[i];
      } else if ((i == iplus) && (i != 0)) {
        ispec1 = aplus[i - 1] + 1;
        ispec2 = len1;
      } else {
        ispec1 = aplus[i - 1] + 1;
        ispec2 = aplus[i];
      }

#ifdef DEBUGMSG
      printf("|%s| %d %d\n", reacprod, ispec1, ispec2);
#endif

      memset(specname, 0, LENGTHOFSPECNAME);
      strncpy(specname, &reacprod[ispec1], ispec2 - ispec1);

      //#define DEBUGMSG
#ifdef DEBUGMSG
      printf("|%s|\n", specname);
#endif
      //#undef DEBUGMSG
      /* Check */
      if (strlen(specname) == 0) {
        *ierror = 615;
        return 1;
      }

#ifdef DEBUGMSG
      printf("--->%d |%s|\n", listreac[*Nreac].isthrdb, reacprod);
#endif
      /* Check for third body indicator */
      if ((strlen(specname) == 1) && (strncmp(specname, "M", 1) == 0)) {
        /* Found third body indicator, check for consistency */
        if (listreac[*Nreac].isfall == 1) {
          *ierror = 635;
          return 1;
        }
        if (icheckTB == 0) {
          listreac[*Nreac].isthrdb = 1;
          icheckTB = 1;
        } else {
          *ierror = 625;
          return 1;
        }
      }
      /* Check for wavelength */
      else if ((strlen(specname) == 2) && (strncmp(specname, "HV", 2) == 0)) {
        if (icheckWL == 0) {
          listreac[*Nreac].iswl = 1;
          if (j == 0)
            listreac[*Nreac].iswl = -1;
          icheckWL = 1;
        } else {
          *ierror = 636;
          return 1;
        }

      } /* Done if section for wavelength */
      else {
        double istcoeff = 1.0;
        double rstcoeff = 1.0;
        /* Check for numbers */
        len1 = strlen(specname);
        TCKMI_findnonnum(specname, &ipos);

        /* identify stoichiometric coefficients */
        if (ipos > 0) {
          memset(stoicoeff, 0, LENGTHOFSPECNAME);
          strncpy(stoicoeff, specname, ipos);
          /* found number */
          // if (listreac[*Nreac].isreal == 1)
          //   rstcoeff = atof(stoicoeff);
          // else
          //   istcoeff = atoi(stoicoeff);
          istcoeff = atof(stoicoeff);
        }

        if (j == 0) {
          /* reactants */
          if (listreac[*Nreac].inreac == NSPECREACMAX) {
            *ierror = 645;
            return 1;
          }
          if (listreac[*Nreac].isreal == 1)
            listreac[*Nreac].rnuki[listreac[*Nreac].inreac] = -rstcoeff;
          else
            listreac[*Nreac].nuki[listreac[*Nreac].inreac] = -istcoeff;

        } else {
          /* products */
          if (listreac[*Nreac].inprod == NSPECREACMAX) {
            printf("%d\n", *Nreac);
            *ierror = 655;
            return 1;
          }
          if (listreac[*Nreac].isreal == 1)
            listreac[*Nreac].rnuki[NSPECREACMAX + listreac[*Nreac].inprod] =
              rstcoeff;
          else
            listreac[*Nreac].nuki[NSPECREACMAX + listreac[*Nreac].inprod] =
              istcoeff;
        }

        /* Check species name */
        /* CS: modified on 2011/01/24 to avoid overlaps between source and and
         * destination arrays */
        if (ipos > 0) {
          char sntmp[LENGTHOFSPECNAME];
          strncpy(sntmp, &specname[ipos], len1 - ipos);
          strncpy(specname, sntmp, len1 - ipos);
        }
        memset(&specname[len1 - ipos], 0, ipos);
        TCKMI_checkspecinlist(specname, listspec, Nspec, &ipos);

        if (ipos == *Nspec) {
          printf("%s\n", specname);
          *ierror = 665;
          return 1;
        }

        if (j == 0) {
          int irep = -1;
          int kspec;
          for (kspec = 0; kspec < listreac[*Nreac].inreac; kspec++) {
            if (ipos == listreac[*Nreac].spec[kspec]) {
              irep = kspec;
              break;
            }
          }

          if (irep >= 0) {
            /* found repeat species, combine info */
            if (listreac[*Nreac].isreal == 1) {
              listreac[*Nreac].rnuki[irep] +=
                listreac[*Nreac].rnuki[listreac[*Nreac].inreac];
              listreac[*Nreac].rnuki[listreac[*Nreac].inreac] = 0.0;
            } else {
              listreac[*Nreac].nuki[irep] +=
                listreac[*Nreac].nuki[listreac[*Nreac].inreac];
              listreac[*Nreac].nuki[listreac[*Nreac].inreac] = 0;
            }
          } else {
            /* found new species in the current reaction */
            listreac[*Nreac].spec[listreac[*Nreac].inreac] = ipos;
            listreac[*Nreac].inreac += 1;
          }

        } else {
          int irep = -1;
          int kspec;
          for (kspec = 0; kspec < listreac[*Nreac].inprod; kspec++) {
            if (ipos == listreac[*Nreac].spec[NSPECREACMAX + kspec]) {
              irep = kspec;
              break;
            }
          }

          if (irep >= 0) {
            /* found repeat species, combine info */
            if (listreac[*Nreac].isreal == 1) {
              listreac[*Nreac].rnuki[NSPECREACMAX + irep] +=
                listreac[*Nreac].rnuki[NSPECREACMAX + listreac[*Nreac].inprod];
              listreac[*Nreac].rnuki[NSPECREACMAX + listreac[*Nreac].inprod] =
                0.0;
            } else {
              listreac[*Nreac].nuki[NSPECREACMAX + irep] +=
                listreac[*Nreac].nuki[NSPECREACMAX + listreac[*Nreac].inprod];
              listreac[*Nreac].nuki[NSPECREACMAX + listreac[*Nreac].inprod] = 0;
            }
          } else {
            listreac[*Nreac].spec[NSPECREACMAX + listreac[*Nreac].inprod] =
              ipos;
            listreac[*Nreac].inprod += 1;
          }
        }

      } /* Done if section for real species */

#ifdef DEBUGMSG
      printf("Done with this location %d\n", i);
#endif
    } /* Done for loop over all inter "+" spaces */

    free(aplus);

  } /* Done for loop over reactants and products */

  linein[0] = 0;

  return (0);
}

/* -------------------------------------------------------------------------  */
/**
 * \brief Interprets a character string containing reaction description
 *   (auxiliary information)
 */
int
TCKMI_getreacauxl(char* linein,
                  char* singleword,
                  species* listspec,
                  int* Nspec,
                  reaction* listreac,
                  int* Nreac,
                  int* ierror)
{
  int len1;
  int ilenkey = 20, ilenval = 200;
  int inum, ipos, i, iswitch, indx, ireac;
  double dvalues[20];
  char* wordkey = (char*)malloc(ilenkey * sizeof(char));
  char* wordval = (char*)malloc(ilenval * sizeof(char));

  /* Return immediately if error flag is not zero */
  if (*ierror > 0)
    return 1;

  /* Store reaction location in the list */
  ireac = (*Nreac) - 1;

  while (strlen(linein) > 0) {
    memset(wordkey, 0, ilenkey);
    memset(wordval, 0, ilenval);
#ifdef DEBUGMSG
    printf(
      "TCKMI_getreacauxl: String: |%s|, Length: %d \n", linein, strlen(linein));
    for (i = 0; i < strlen(linein); i++)
      printf(
        "TCKMI_getreacauxl: %d: %c vs %d \n", i, linein[i], (int)linein[i]);
#endif
    TCKMI_extractWordLeftauxline(linein, wordkey, wordval, &inum, ierror);
#ifdef DEBUGMSG
    printf("TCKMI_getreacauxl: String: |%s|, Length: %d \n",
           wordkey,
           strlen(wordkey));
#endif

    if (*ierror > 0)
      return 1;

    /* Check for inconsistencies */
    if (inum == 0) {
      *ierror = 710;
      return 1;
    }

    if ((inum == 1) && ((strncmp(wordkey, "DUP", 3) != 0) && /* duplicate */
                        (strncmp(wordkey, "MOME", 4) !=
                         0) && /* momentum-transfer collision frequency */
                        (strncmp(wordkey, "XSMI", 4) !=
                         0) /* ion momentum-transfer collision frequency */
                        )) {
      *ierror = 711;
      return 1;
    }

    if ((inum == 2) && ((strncmp(wordkey, "DUP", 3) == 0) ||
                        (strncmp(wordkey, "MOME", 4) == 0) ||
                        (strncmp(wordkey, "XSMI", 4) == 0))) {
      *ierror = 712;
      return 1;
    }

    if (strncmp(wordkey, "DUP", 3) == 0) {
      listreac[ireac].isdup = -1;

    } /* End test for duplicate keyword */

    else if (strncmp(wordkey, "MOME", 4) == 0) {
      listreac[ireac].ismome = 1;

    } /* End test for MOME keyword */

    else if (strncmp(wordkey, "XSMI", 4) == 0) {
      listreac[ireac].isxsmi = 1;

    } /* End test for XSMI keyword */

    else if (strncmp(wordkey, "LOW", 3) == 0) {
#ifdef DEBUGMSG
      printf("TCKMI_getreacauxl: found LOW\n");
#endif
      if ((listreac[ireac].islowset > 0) || (listreac[ireac].ishighset > 0)) {
        *ierror = 715;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 716;
        return 1;
      }
      if (inum != 3) {
        *ierror = 717;
        return 1;
      }
      for (i = 0; i < 3; i++)
        listreac[ireac].fallpar[i] = dvalues[i];
      listreac[ireac].islowset = 1;

    } /* End test for LOW keyword */

    else if (strncmp(wordkey, "HIGH", 4) == 0) {
      if ((listreac[ireac].islowset > 0) || (listreac[ireac].ishighset > 0)) {
        *ierror = 780;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 781;
        return 1;
      }
      if (inum != 3) {
        *ierror = 782;
        return 1;
      }
      for (i = 0; i < 3; i++)
        listreac[ireac].fallpar[i] = dvalues[i];
      listreac[ireac].ishighset = 1;

    } /* End test for HIGH keyword */

    else if (strncmp(wordkey, "TROE", 4) == 0) {
      if (listreac[ireac].istroeset > 0) {
        *ierror = 720;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 721;
        return 1;
      }
      if ((inum != 3) && (inum != 4)) {
        *ierror = 722;
        return 1;
      }
      for (i = 0; i < inum; i++)
        listreac[ireac].fallpar[3 + i] = dvalues[i];
      listreac[ireac].istroeset = inum;

    } /* End test for TROE keyword */

    else if (strncmp(wordkey, "PLOG", 4) == 0) {

      // printf("TCKMI_getreacauxl: error parsing PLOG numbers:%s\n",wordval) ;
      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        printf("TCKMI_getreacauxl: error parsing PLOG numbers!\n");
        // for (i=0;i<inum;i++)
        //  printf("TCKMI_getreacauxl: error parsing PLOG numbers:%d\n",
        //  dvalues[i]) ;
        exit(1);
      }
      if (inum != 4) {
        printf("TCKMI_getreacauxl: could not find 4 number on PLOG line!\n");
        exit(1);
      }
      for (i = 0; i < inum; i++)
        listreac[ireac].plog[4 * listreac[ireac].isplogset + i] = dvalues[i];
      listreac[ireac].isplogset += 1;

    } /* End test for PLOG keyword */

    else if (strncmp(wordkey, "SRI", 3) == 0) {
      if ((listreac[ireac].istroeset > 0) || (listreac[ireac].issriset > 0)) {
        *ierror = 725;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 726;
        return 1;
      }
      if ((inum != 3) && (inum != 5)) {
        *ierror = 727;
        return 1;
      }

      if (inum == 3) {
        dvalues[3] = 1.0;
        dvalues[4] = 0.0;
      }
      for (i = 0; i < 5; i++)
        listreac[ireac].fallpar[3 + i] = dvalues[i];
      listreac[ireac].issriset = inum;

    } /* End test for SRI keyword */
    else if (strncmp(wordkey, "REV", 3) == 0) {
      if (listreac[ireac].isrevset > 0) {
        *ierror = 730;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 731;
        return 1;
      }
      if (inum != 3) {
        *ierror = 732;
        return 1;
      }

      for (i = 0; i < 3; i++)
        listreac[ireac].arhenrev[i] = dvalues[i];
      listreac[ireac].isrevset = 1;

    } /* End test for REV keyword */
    else if (strncmp(wordkey, "LT", 2) == 0) {
      if (listreac[ireac].isltset > 0) {
        *ierror = 735;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 736;
        return 1;
      }
      if (inum != 2) {
        *ierror = 737;
        return 1;
      }
      for (i = 0; i < 2; i++)
        listreac[ireac].ltpar[i] = dvalues[i];
      listreac[ireac].isltset = 1;

    } /* End test for LT keyword */

    else if (strncmp(wordkey, "RLT", 3) == 0) {
      if (listreac[ireac].isrltset > 0) {
        *ierror = 740;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 741;
        return 1;
      }
      if (inum != 2) {
        *ierror = 742;
        return 1;
      }
      for (i = 0; i < 2; i++)
        listreac[ireac].rltpar[i] = dvalues[i];
      listreac[ireac].isrltset = 1;

    } /* End test for RLT keyword */

    else if (strncmp(wordkey, "HV", 2) == 0) {
      if (listreac[ireac].ishvset > 0) {
        *ierror = 745;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 746;
        return 1;
      }
      if (inum != 1) {
        *ierror = 747;
        return 1;
      }
      listreac[ireac].hvpar = dvalues[0] * listreac[ireac].iswl;
      listreac[ireac].ishvset = 1;

    } /* End test for HV keyword */

    else if (strncmp(wordkey, "TDEP", 4) == 0) {
      if (listreac[ireac].istdepset > 0) {
        *ierror = 770;
        return 1;
      }

      TCKMI_elimleads(wordval);
      TCKMI_elimends(wordval);

      TCKMI_checkspecinlist(wordval, listspec, Nspec, &ipos);
      if (ipos == *Nspec) {
        *ierror = 771;
        return 1;
      }

      listreac[ireac].tdeppar = ipos;
      listreac[ireac].istdepset = 1;

    } /* End test for TDEP */

    else if (strncmp(wordkey, "EXCI", 4) == 0) {
      if (listreac[ireac].isexciset > 0) {
        *ierror = 775;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 776;
        return 1;
      }
      if (inum != 1) {
        *ierror = 777;
        return 1;
      }
      listreac[ireac].excipar = dvalues[0];
      listreac[ireac].isexciset = 1;

    } /* End test for EXCI */

    else if (strncmp(wordkey, "JAN", 3) == 0) {
      if (listreac[ireac].isjanset > 0) {
        *ierror = 785;
        return 1;
      }
      if (listreac[ireac].isfit1set > 0) {
        *ierror = 786;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 787;
        return 1;
      }
      if (inum != 9) {
        *ierror = 788;
        return 1;
      }
      for (i = 0; i < 9; i++)
        listreac[ireac].optfit[i] = dvalues[i];
      listreac[ireac].isjanset = 1;

    } /* End test for JAN */

    else if (strncmp(wordkey, "FIT1", 4) == 0) {
      if (listreac[ireac].isfit1set > 0) {
        *ierror = 790;
        return 1;
      }
      if (listreac[ireac].isjanset > 0) {
        *ierror = 791;
        return 1;
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 792;
        return 1;
      }
      if (inum != 4) {
        *ierror = 793;
        return 1;
      }
      for (i = 0; i < 4; i++)
        listreac[ireac].optfit[i] = dvalues[i];
      listreac[ireac].isfit1set = 1;

    } /* End test for FIT1 */

    else if (strncmp(wordkey, "UNITS", 5) == 0) {
      int spacefind;
      TCKMI_elimleads(wordval);
      TCKMI_elimends(wordval);
      len1 = strlen(wordval);
      if (len1 == 0) {
        *ierror = 760;
        return 1;
      }
      spacefind = strcspn(wordval, " ");

      if (spacefind != len1) {
        /* found two keywords */
        inum = 2;
        strncpy(wordkey, wordval, spacefind);
        wordkey[spacefind] = 0;
        memmove(wordval, &wordval[spacefind], len1 - spacefind);
        memset(&wordval[len1 - spacefind], 0, spacefind);
        TCKMI_elimleads(wordval);
        len1 = strlen(wordval);

      } else {
        inum = 1;
        strncpy(wordkey, wordval, len1);
      }

      for (i = 0; i < inum; i++) {
        if (i == 1) {
          strncpy(wordkey, wordval, len1);
          wordkey[len1] = 0;
        }
        if (strncmp(wordkey, "CAL/", 4) == 0)
          strncpy(listreac[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "KCAL", 4) == 0)
          strncpy(listreac[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "JOUL", 4) == 0)
          strncpy(listreac[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "KJOU", 4) == 0)
          strncpy(listreac[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "KELV", 4) == 0)
          strncpy(listreac[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "EVOL", 4) == 0)
          strncpy(listreac[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "MOLE", 4) == 0) {
          if (strncmp(&wordkey[4], "C", 1) == 0)
            strncpy(listreac[ireac].aunits, "MOLC", 4);
          else
            strncpy(listreac[ireac].aunits, wordkey, 4);
        } else {
          *ierror = 761;
          return 1;
        }
      }

    } /* End test for UNITS keyword */

    else if ((strncmp(wordkey, "FORD", 4) == 0) ||
             (strncmp(wordkey, "RORD", 4) == 0)) {

      int spacefind;

      if (strncmp(wordkey, "FORD", 4) == 0)
        iswitch = -1;
      else
        iswitch = 1;
      if (listreac[ireac].isford == 0) {
        listreac[ireac].isford = listreac[ireac].inreac;
        listreac[ireac].isrord = listreac[ireac].inprod;
#ifdef DEBUG
        printf("FORD:%d %d\n", listreac[ireac].inreac, listreac[ireac].inprod);
#endif
        /* transfer reactant info */
        for (i = 0; i < listreac[ireac].isford; i++) {
          listreac[ireac].arbspec[i] = listreac[ireac].spec[i];
          if (listreac[ireac].isreal == 1)
            listreac[ireac].arbnuki[i] = fabs(listreac[ireac].rnuki[i]);
          else
            listreac[ireac].arbnuki[i] = fabs(listreac[ireac].nuki[i]);
        }

        /* transfer product info */
        for (i = 0; i < listreac[ireac].isrord; i++) {
          listreac[ireac].arbspec[2 * NSPECREACMAX + i] =
            listreac[ireac].spec[NSPECREACMAX + i];
          if (listreac[ireac].isreal == 1)
            listreac[ireac].arbnuki[2 * NSPECREACMAX + i] =
              listreac[ireac].rnuki[NSPECREACMAX + i];
          else
            listreac[ireac].arbnuki[2 * NSPECREACMAX + i] =
              listreac[ireac].nuki[NSPECREACMAX + i];
        }

      } /* Done setting-up the arbitrary order */

      /* Extract species name */
      TCKMI_elimleads(wordval);
      TCKMI_elimends(wordval);
      len1 = strlen(wordval);
      if (len1 == 0) {
        *ierror = 920;
        return 1;
      }
      spacefind = strcspn(wordval, " ");

      if (spacefind != len1) {
        /* found two keywords */
        strncpy(wordkey, wordval, spacefind);
        wordkey[spacefind] = 0;
        memmove(wordval, &wordval[spacefind], len1 - spacefind);
        memset(&wordval[len1 - spacefind], 0, spacefind);
        TCKMI_elimleads(wordval);
        len1 = strlen(wordval);
      } else {
        *ierror = 921;
        return 1;
      }

      TCKMI_checkspecinlist(wordkey, listspec, Nspec, &ipos);
      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (ipos == *Nspec) {
        *ierror = 922;
        return 1;
      }

      if (*ierror > 0) {
        *ierror = 923;
        return (1);
      }

      if (inum != 1) {
        *ierror = 924;
        return (1);
      }

      if (iswitch == -1) {
        iswitch = -1;
        for (i = 0; i < listreac[ireac].isford; i++) {
          if (ipos == listreac[ireac].arbspec[i]) {
            iswitch = i;
            listreac[ireac].arbnuki[i] = dvalues[0];
            break;
          }
        }

        if (iswitch == -1) {
          if (listreac[ireac].isford == 2 * NSPECREACMAX) {
            *ierror = 925;
            return (1);
          }
          indx = listreac[ireac].isford;
          listreac[ireac].arbspec[indx] = ipos;
          listreac[ireac].arbnuki[indx] = dvalues[0];
          listreac[ireac].isford += 1;
        }
      } /* End update for FORD */

      else {
        iswitch = -1;
        for (i = 0; i < listreac[ireac].isrord; i++) {
          indx = 2 * NSPECREACMAX + i;
          if (ipos == listreac[ireac].arbspec[indx]) {
            iswitch = i;
            listreac[ireac].arbnuki[indx] = dvalues[0];
            break;
          }
        }

        if (iswitch == -1) {
          if (listreac[ireac].isrord == 2 * NSPECREACMAX) {
            *ierror = 926;
            return (1);
          }
          indx = 2 * NSPECREACMAX + listreac[ireac].isrord;
          listreac[ireac].arbspec[indx] = ipos;
          listreac[ireac].arbnuki[indx] = dvalues[0];
          listreac[ireac].isrord += 1;
        }
      } /* End update for RORD */

    } /* End test for arbitrary reaction order */
    else {
      /* Possible found third-body */
      TCKMI_checkspecinlist(wordkey, listspec, Nspec, &ipos);

      if (ipos == *Nspec) {
        printf("%s\n", wordkey);
        *ierror = 750;
        return 1;
      }

      if (listreac[ireac].nthrdb == NTHRDBMAX) {
        *ierror = 751;
        return 1;
      }

      for (i = 0; i < listreac[ireac].nthrdb; i++) {
        if (ipos == listreac[ireac].ithrdb[i]) {
          *ierror = 752;
          return 1;
        }
      }

      TCKMI_extractdouble(wordval, dvalues, &inum, ierror);
      if (*ierror > 0) {
        *ierror = 753;
        return 1;
      }
      if (inum != 1) {
        *ierror = 754;
        return 1;
      }
      listreac[ireac].ithrdb[listreac[ireac].nthrdb] = ipos;
      listreac[ireac].rthrdb[listreac[ireac].nthrdb] = dvalues[0];
      listreac[ireac].nthrdb += 1;
    }

  } /* End loop sweeping the length of linein */

  /* Garbage collection */
  free(wordkey);
  free(wordval);

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Interprets a character string containing reaction description
 *    - decides if the character string describes a reaction or the
 *      auxiliary information associated with one
 */
int
TCKMI_getreactions(char* linein,
                   char* singleword,
                   species* listspec,
                   int* Nspec,
                   reaction* listreac,
                   int* Nreac,
                   char* aunits,
                   char* eunits,
                   int* ierror)
{
  int i, len1, iloc;

  //#define DEBUGMSG
#ifdef DEBUGMSG
  printf("In getreactions: |%s|\n", linein);
#endif
  //#undef DEBUGMSG

  /* Return immediately if error flag is not zero */
  if (*ierror > 0)
    return 1;

  /* Return if character string is null */
  len1 = strlen(linein);
  if (len1 == 0)
    return 0;

  /* Transform everything to uppercase */
  for (i = 0; i < len1; i++)
    linein[i] = toupper(linein[i]);

  /* Check is character string contains auxiliary data */
  iloc = -1;
  /* 1. look for DUPLICATE keyword */
  for (i = 0; i < len1 - 2; i++)
    if (strncmp(&linein[i], "DUP", 3) == 0)
      iloc = i;
  /* 2. look for XSMI keyword */
  for (i = 0; i < len1 - 2; i++)
    if (strncmp(&linein[i], "XSMI", 4) == 0)
      iloc = i;
  /* 3. look for MOME keyword */
  for (i = 0; i < len1 - 2; i++)
    if (strncmp(&linein[i], "MOME", 4) == 0)
      iloc = i;
  /* 4. look for numbers (automatically involving /) */
  for (i = 0; i < len1; i++)
    if (strncmp(&linein[i], "/", 1) == 0)
      iloc = i;
#ifdef DEBUGMSG
  printf("TCKMI_getreactions: %d \n", iloc);
#endif

  if ((iloc > -1) && (*Nreac == 0)) {
    /* Error, cannot have auxiliary data before reading at least one reaction */
    *ierror = 500;
    return 1;
  }

  if (iloc > -1) {
    /* Character string contains auxiliary info */
#ifdef DEBUGMSG
    printf("TCKMI_getreactions: entering TCKMI_getreacauxl\n");
#endif
    TCKMI_getreacauxl(
      linein, singleword, listspec, Nspec, listreac, Nreac, ierror);
#ifdef DEBUGMSG
    printf("TCKMI_getreactions: done with TCKMI_getreacauxl\n");
#endif
  } else {
    /* Character string contains reaction description */
    /* First, reset the info for the current reaction */
    TCKMI_resetreacdata(&listreac[*Nreac], aunits, eunits);
    /* Next, interpret the string, and increment the current reaction line */
    TCKMI_getreacline(
      linein, singleword, listspec, Nspec, listreac, Nreac, ierror);
    (*Nreac)++;
  }

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Verifies corectness and completness for all reactions in the list
 */

int
TCKMI_verifyreac(element* listelem,
                 int* Nelem,
                 species* listspec,
                 int* Nspec,
                 reaction* listreac,
                 int* Nreac,
                 int* ierror)
{
  int icount = 0, icheck;
  int i, j, k, l, i1, j1;
  int ispec;
  int isum = 0;
  double rsum = 0.0;
  int icheckvec[NSPECREACMAX];

  /* Zero-out the error flag */
  (*ierror) = 0;

  /*
  -------------------Check if all species have thermodynamic
  data------------------
  */
  for (i = 0; i < *Nspec; i++) {
    if (listspec[i].hasthermo == 0) {
      printf("!!! Could not find thermodynamic data for : %s\n",
             listspec[i].name);
      icount += 1;
      /* return 1 ; */
    }
  }

  /*
  ------------------------------Check
  balance--------------------------------------
  */

  for (i = 0; i < *Nreac; i++) {
    for (j = 0; j < *Nelem; j++) {
      if (listreac[i].isreal == 1)
        rsum = 0.0;
      else
        isum = 0;

      /* Sum reactants */
      for (k = 0; k < listreac[i].inreac; k++) {
        ispec = listreac[i].spec[k];
        for (l = 0; l < listspec[ispec].numofelem; l++) {
          if (listspec[ispec].elemindx[l] == j) {
            if (listreac[i].isreal == 1)
              rsum += listspec[ispec].elemcontent[l] * listreac[i].rnuki[k];
            else
              isum += listspec[ispec].elemcontent[l] * listreac[i].nuki[k];
          }
        }

      } /* Done summing contributions from reactants */

#ifdef DEBUGMSG
      if (i == (*Nreac) - 1)
        printf("-------------> %d %d %d\n", i, j, isum);

      if (i == (*Nreac) - 1)
        printf("-------------> %d %d %e\n", i, j, rsum);
#endif

      /* Sum products */
      for (k = 0; k < listreac[i].inprod; k++) {
        ispec = listreac[i].spec[NSPECREACMAX + k];
        for (l = 0; l < listspec[ispec].numofelem; l++) {
          if (listspec[ispec].elemindx[l] == j) {
            if (listreac[i].isreal == 1)
              rsum += listspec[ispec].elemcontent[l] *
                      listreac[i].rnuki[NSPECREACMAX + k];
            else
              isum += listspec[ispec].elemcontent[l] *
                      listreac[i].nuki[NSPECREACMAX + k];
          }
        }
      } /* Done summing contributions from products */

#ifdef DEBUGMSG
      if (i == (*Nreac) - 1)
        printf("-------------> %d %d %d\n", i, j, isum);

      if (i == (*Nreac) - 1)
        printf("-------------> %d %d %d\n", i, j, rsum);
#endif

      if (listreac[i].isreal == 1) {
        if (fabs(rsum) > REACBALANCE()) {
          printf("!!! Error: Reaction #%d does not balance for element %s\n",
                 i + 1,
                 listelem[j].name);
          icount += 1;
          /* return 1 ; */
        }
      } else {
        if (isum != 0) {
          printf("!!! Error: Reaction #%d does not balance for element %s\n",
                 i + 1,
                 listelem[j].name);
          icount += 1;
          return 1;
        }
      }

    } /* Done loop over number of elements */

  } /* Done loop over number of reactions */

  /*
  ------------------------------Check
  charge--------------------------------------
  */
  for (i = 0; i < *Nreac; i++) {
    if (listreac[i].isreal == 1) {
      rsum = 0.0;
      for (k = 0; k < listreac[i].inreac; k++) {
        ispec = listreac[i].spec[k];
        rsum += listreac[i].rnuki[k] * listspec[ispec].charge;
      }
      for (k = 0; k < listreac[i].inprod; k++) {
        ispec = listreac[i].spec[NSPECREACMAX + k];
        rsum += listreac[i].rnuki[NSPECREACMAX + k] * listspec[ispec].charge;
      }
      if (fabs(rsum) > REACBALANCE()) {
        printf(
          "!!! Error: Electron charge does not balance for reaction #%d ! \n",
          i + 1);
        icount += 1;
        /* return 1 ; */
      }
    } else {
      isum = 0;
      for (k = 0; k < listreac[i].inreac; k++) {
        ispec = listreac[i].spec[k];
        isum += listreac[i].nuki[k] * listspec[ispec].charge;
        /* if (abs(isum)>0) { */
        /*   printf("%d %d %d
         * %d\n",i,ispec,listreac[i].nuki[k],listspec[ispec].charge,listspec[ispec].name);
         */
        /* } */
      }
      for (k = 0; k < listreac[i].inprod; k++) {
        ispec = listreac[i].spec[NSPECREACMAX + k];
        isum += listreac[i].nuki[NSPECREACMAX + k] * listspec[ispec].charge;
        /* if (abs(isum)>0) { */
        /*   printf("%d %d %d %d:
         * %s\n",i,ispec,listreac[i].nuki[NSPECREACMAX+k],listspec[ispec].charge,listspec[ispec].name);
         */
        /* } */
      }
      if (isum != 0) {
        printf(
          "!!! Error: Electron charge does not balance for reaction #%d ! \n",
          i + 1);
        icount += 1;
        /* return 1 ; */
      }
    }
  }

  /*
  ---------------------Check parameter
  compatibilities----------------------------
  */

  for (i = 0; i < *Nreac; i++) {
    /* pressure-dependent consistencies */
    if (listreac[i].isfall == 0) {
      if ((listreac[i].islowset > 0) || (listreac[i].ishighset > 0) ||
          (listreac[i].istroeset > 0) || (listreac[i].issriset > 0)) {
        printf("!!! Error: Reaction #%d is not pressure-dependent but has "
               "LOW/HIGH/TROE/SRI parameters\n",
               i + 1);
        icount += 1;
      }
    }

    if (listreac[i].isfall > 0) {
      if ((listreac[i].islowset == 0) && (listreac[i].ishighset == 0)) {
        printf("!!! Error: Reaction #%d is pressure-dependent but does not "
               "have LOW/HIGH parameters\n",
               i + 1);
        icount += 1;
      }

      if (listreac[i].isltset > 0) {
        printf("!!! Error: Reaction #%d is pressure-dependent and has "
               "Landau-Teller parameters\n",
               i + 1);
        icount += 1;
      }

      if (listreac[i].isrevset > 0) {
        printf("!!! Error: Reaction #%d is pressure-dependent and has reverse "
               "parameters\n",
               i + 1);
        icount += 1;
      }
    }

    /* Forbidden pressure-dependent combinations */
    if ((listreac[i].islowset > 0) && (listreac[i].ishighset > 0)) {
      printf("!!! Error: Reaction #%d has both LOW and HIGH parameters\n",
             i + 1);
      icount += 1;
    }
    if ((listreac[i].istroeset > 0) && (listreac[i].issriset > 0)) {
      printf("!!! Error: Reaction #%d has both TROE and SRI parameters\n",
             i + 1);
      icount += 1;
    }

    /* Landau-Teller consistencies */
    if (listreac[i].isltset > 0) {
      if ((listreac[i].isrevset > 0) && (listreac[i].isrltset == 0)) {
        printf("!!! Error: Reaction #%d has LT,REV parameters but misses "
               "reverse Landau-Teller \n",
               i + 1);
        icount += 1;
      }

    } /* Done checking LT consistencies */

    /* reverse Landau-Teller consistencies */
    if (listreac[i].isrltset > 0) {
      if (listreac[i].isltset > 0) {
        printf("!!! Error: Reaction #%d has reverse LT parameters and but no "
               "LT parameters\n",
               i + 1);
        icount += 1;
      }

    } /* Done checking reverse LT consistencies */

    /* reverse consistencies */
    if (listreac[i].isrevset > 0) {

      if (listreac[i].isrev < 0) {
        printf("!!! Error: Reaction #%d has reverse parameters but is "
               "irreversible\n",
               i + 1);
        icount += 1;
      }

    } /* Done checking reverse consistencies */

    /* check XSMI and MOME */
    if ((listreac[i].ismome > 0) || (listreac[i].isxsmi > 0)) {
      if (listreac[i].inreac != 2) {
        printf("!!! Error: Reaction #%d has more or less than 2 reactants and "
               "is MOME/XSMI\n",
               i + 1);
        icount += 1;
      }
      if ((listreac[i].isxsmi > 0) &&
          (listspec[listreac[i].spec[0]].charge == 0) &&
          (listspec[listreac[i].spec[1]].charge == 0)) {
        printf(
          "!!! Error: XSMI Reaction #%d needs at least one ionic reactant\n",
          i + 1);
        icount += 1;
      }
      if ((listreac[i].ismome > 0) &&
          (strncmp(listspec[listreac[i].spec[0]].name, "E", 1) != 0) &&
          (strncmp(listspec[listreac[i].spec[1]].name, "E", 1) != 0)) {
        printf(
          "!!! Error: MOME Reaction #%d needs at least one electron reactant\n",
          i + 1);
        icount += 1;
      }
    }

  } /* Done looping over all reactions */

  for (i = 0; i < (*Nreac) - 1; i++) {

    for (j = i + 1; j < (*Nreac); j++) {

      while (j > i) {

        /* Check for same number of reactants and products */
        if (listreac[i].inreac != listreac[j].inreac)
          break;
        if (listreac[i].inprod != listreac[j].inprod)
          break;

        /* Check for same type of stoichiometric coefficients */
        if (listreac[i].isreal != listreac[j].isreal)
          break;

        /* Check for third body */
        if (listreac[i].isthrdb != listreac[j].isthrdb)
          break;

        /* Check for number third body */
        if (listreac[i].nthrdb != listreac[j].nthrdb)
          break;

        /* Check for fall-off */
        if (listreac[i].isfall != listreac[j].isfall)
          break;

        /* For fall-off check if third body is the same */
        if ((listreac[i].isfall == 1) &&
            (listreac[i].specfall != listreac[j].specfall))
          break;

        /* check reactants one by one */
        icheck = 0;
        for (i1 = 0; i1 < NSPECREACMAX; i1++)
          icheckvec[i1] = 0;

        for (j1 = 0; j1 < listreac[j].inreac; j1++) {
          for (i1 = 0; i1 < listreac[i].inreac; i1++) {
            if (listreac[i].isreal == 1) {
              if ((listreac[j].spec[j1] == listreac[i].spec[i1]) &&
                  ((fabs(listreac[j].rnuki[j1] - listreac[i].rnuki[i1]) <
                    REACBALANCE()) &&
                   (icheckvec[i1] == 0))) {
                icheck += 1;
                icheckvec[i1] += 1;
              }
            } else {
              if ((listreac[j].spec[j1] == listreac[i].spec[i1]) &&
                  ((listreac[j].nuki[j1] - listreac[i].nuki[i1] == 0) &&
                   (icheckvec[i1] == 0))) {
                icheck += 1;
                icheckvec[i1] += 1;
              }
            }
          }
        }
        if (icheck != listreac[i].inreac)
          break;

        /* check products one by one */
        icheck = 0;
        for (i1 = 0; i1 < NSPECREACMAX; i1++)
          icheckvec[i1] = 0;

        for (j1 = 0; j1 < listreac[j].inprod; j1++) {
          for (i1 = 0; i1 < listreac[i].inprod; i1++) {
            if (listreac[i].isreal == 1) {
              if ((listreac[j].spec[NSPECREACMAX + j1] ==
                   listreac[i].spec[NSPECREACMAX + i1]) &&
                  ((fabs(listreac[j].rnuki[NSPECREACMAX + j1] -
                         listreac[i].rnuki[NSPECREACMAX + i1]) < REACBALANCE()) &&
                   (icheckvec[i1] == 0))) {
                icheck += 1;
                icheckvec[i1] += 1;
                break;
              }
            } else {
              if ((listreac[j].spec[NSPECREACMAX + j1] ==
                   listreac[i].spec[NSPECREACMAX + i1]) &&
                  ((listreac[j].nuki[NSPECREACMAX + j1] -
                      listreac[i].nuki[NSPECREACMAX + i1] ==
                    0) &&
                   (icheckvec[i1] == 0))) {
                icheck += 1;
                icheckvec[i1] += 1;
                break;
              }
            }
          }
        }
        if (icheck != listreac[i].inprod)
          break;

        /* Found duplicate reactions */
        if (listreac[i].isdup == -2) {
          printf(
            "!!! Error: Reaction #%d is a duplicate but was not declared so\n",
            i + 1);
          icount += 1;
        }
        if (listreac[j].isdup == -2) {
          printf(
            "!!! Error: Reaction #%d is a duplicate but was not declared so\n",
            j + 1);
          icount += 1;
        }
        if (listreac[i].isdup > -1)
          listreac[j].isdup = listreac[i].isdup;
        else if (listreac[i].isdup == -1)
          listreac[j].isdup = i;

        break;
      }

    } /* Done inner loop */

  } /* Done outer loop */

#ifdef VERBOSE
  printf("!!! There are %d errors in the kinetic model \n", icount);
#endif

  if (icount > 0)
    (*ierror) = 1000;

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Verifies corectness and completness for all reactions in the list
 */

int
TCKMI_rescalereac(reaction* listreac, int* Nreac)
{

  int i, j;
  double factor;

  for (i = 0; i < (*Nreac); i++) {

    /* Pre-exponential factor */
    factor = 1.0;
    if (strncmp(listreac[i].aunits, "MOLC", 4) == 0) {
      /* Found molecules for pre-exponential factor, need to rescale */
      double nureac = 0, nuprod = 0;
      if (listreac[i].isford > 0) {
        /* Sum arbitrary order coefficients */
        for (j = 0; j < listreac[i].isford; j++)
          nureac += listreac[i].arbnuki[j];
        for (j = 0; j < listreac[i].isrord; j++)
          nuprod += listreac[i].arbnuki[2 * NSPECREACMAX + j];
      } else if (listreac[i].isreal == 0) {
        /* Sum the (integer) stoichiometric coefficients */
        for (j = 0; j < listreac[i].inreac; j++)
          nureac += listreac[i].nuki[j];
        for (j = 0; j < listreac[i].inprod; j++)
          nuprod += listreac[i].nuki[NSPECREACMAX + j];
      } else {
        /* Sum the (real) stoichiometric coefficients */
        for (j = 0; j < listreac[i].inreac; j++)
          nureac += listreac[i].nuki[j];
        for (j = 0; j < listreac[i].inprod; j++)
          nuprod += listreac[i].nuki[NSPECREACMAX + j];
      }

      /* Scale only if not MOME or XSMI */
      if ((listreac[i].ismome == 0) && (listreac[i].isxsmi == 0)) {
        if (listreac[i].isfall > 0) {
          /* Pressure-dependent */
          factor = pow(NAVOG, nureac - 1.0);
          listreac[i].arhenfor[2] *= factor; /* (+M) does not count */
          factor *= NAVOG;
          listreac[i].fallpar[2] *= factor; /* (+M) does count */
        } else if (listreac[i].isthrdb > 0) {
          /* Third-body present */
          factor = pow(NAVOG, nureac);
          listreac[i].arhenfor[2] *= factor; /* (+M) does count */
          if (listreac[i].isrev > 0) {
            factor = pow(NAVOG, nuprod);
            listreac[i].arhenrev[2] *= factor; /* (+M) does count */
          }
        } else {
          factor = pow(NAVOG, nureac - 1);
          listreac[i].arhenfor[2] *= factor;
          if (listreac[i].isrev > 0) {
            factor = pow(NAVOG, nuprod - 1);
            listreac[i].arhenrev[2] *= factor;
          }
        }
      }

    } /* Done with the pre-exponential factor */

    /* Activation energies */
    factor = 1.0;
    if (strncmp(listreac[i].eunits, "KELV", 4) != 0) {

      if (strncmp(listreac[i].eunits, "CAL/", 4) == 0)

        /* Found calories for activation energies, need to rescale */
        factor = CALJO / RUNIV;
      /*    factor = 4.184/8.31451 ; */

      else if (strncmp(listreac[i].eunits, "KCAL", 4) == 0)

        /* Found kcalories for activation energies, need to rescale */
        factor = CALJO / RUNIV * 1000.0;

      else if (strncmp(listreac[i].eunits, "JOUL", 4) == 0)

        /* Found Joules for activation energies, need to rescale */
        factor = 1.0 / RUNIV;

      else if (strncmp(listreac[i].eunits, "KJOU", 4) == 0)

        /* Found kJoules for activation energies, need to rescale */
        factor = 1000.0 / RUNIV;

      else if (strncmp(listreac[i].eunits, "EVOL", 4) == 0)

        /* Found electron Volts for activation energies, need to rescale */
        factor = EVOLT / KBOLT;

      else {
        printf("!!! Found unknown activation energy units for reaction #%d", i);
        return 10;
      }
    }

    /* Scale the forward activation energy */
    listreac[i].arhenfor[2] *= factor;

    /* If reverse parameter were provided, then scale it */
    if (listreac[i].isrevset > 0)
      listreac[i].arhenrev[2] *= factor;

    /* If pressure-dependent, scale the fall-off value */
    if (listreac[i].isfall > 0)
      listreac[i].fallpar[2] *= factor;

    /* If PLOG, scale the list of activation energies */
    if (listreac[i].isplogset > 0) {
      for (j = 0; j < listreac[i].isplogset; j++)
        listreac[i].plog[4 * j + 3] *= factor;
    }
  }

  return 0;

} /* Done with "TCKMI_rescalereac" */

/*
                 ___    _____
                |_ _|  / / _ \
                 | |  / / | | |
                 | | / /| |_| |
                |___/_/  \___/

*/
/* -------------------------------------------------------------------------  */
/**
 * \brief Outputs reaction data to ascii file
 */

int
TCKMI_saveRectionEquations(species* listspec,
                     reaction* listreac,
                     int* Nreac,
                     FILE* fileascii)
{
  for (int i = 0; i < *Nreac; i++) {
    /* Output reactant string */
    for (int j = 0; j < listreac[i].inreac; j++) {
      if (-listreac[i].nuki[j] != 1){
        const int value(listreac[i].nuki[j]);
        if (value == listreac[i].nuki[j] ){
          fprintf(fileascii, "%d", -value);
        } else {
          fprintf(fileascii, "%.4f", -listreac[i].nuki[j]);
        }
      }

      fprintf(fileascii, "%s", listspec[listreac[i].spec[j]].name);

      if (j != listreac[i].inreac - 1)
        fprintf(fileascii, "%s", " + ");
    }

    /* Check for fall-off */
    if ((listreac[i].isfall > 0) && (listreac[i].specfall < 0))
      fprintf(fileascii, "%s", " (+M) ");
    else if ((listreac[i].isfall > 0) && (listreac[i].specfall >= 0))
      fprintf(
        fileascii, "%s%s%s", " (+", listspec[listreac[i].specfall].name, ") ");

    /* Check for third-body */
    if ((listreac[i].isfall == 0) && (listreac[i].isthrdb > 0))
      fprintf(fileascii, "%s", " +M ");

    /* Check for photon activation */
    if (listreac[i].iswl < 0)
      fprintf(fileascii, "%s", " +HV ");

    /* Output delimiter */
    if (listreac[i].isrev > 0)
      fprintf(fileascii, "%s", " <=> ");
    else
      fprintf(fileascii, "%s", " => ");

    /* Output product string */
    for (int j = 0; j < listreac[i].inprod; j++) {
        if (listreac[i].nuki[NSPECREACMAX + j] != 1){
          const int value(listreac[i].nuki[NSPECREACMAX + j]);
          if ( value == listreac[i].nuki[NSPECREACMAX + j] ){
            fprintf(fileascii, "%d", value);
          } else {
            fprintf(fileascii, "%.4f", listreac[i].nuki[NSPECREACMAX + j]);
          }

        }
      fprintf(
        fileascii, "%s", listspec[listreac[i].spec[NSPECREACMAX + j]].name);

      if (j != listreac[i].inprod - 1)
        fprintf(fileascii, "%s", " + ");
    }

    fprintf(fileascii, ",\n");
  }
  return 0;
}
int
TCKMI_outform(element* listelem,
              int* Nelem,
              species* listspec,
              int* Nspec,
              reaction* listreac,
              int* Nreac,
              char* aunits,
              char* eunits,
              FILE* fileascii)
{
  int i, j, k, ipos;

  /* Output elements */
  fprintf(fileascii, "Number of elements in the mechanism : %d\n", *Nelem);
  fprintf(fileascii, "-------------------------------------------\n");
  fprintf(fileascii, "  No.    Name       Mass \n");
  for (i = 0; i < *Nelem; i++)
    fprintf(fileascii,
            "  %3d     %3s     %9.3e\n",
            i + 1,
            listelem[i].name,
            listelem[i].mass);
  fprintf(fileascii, "\n");

  /* Output species */
  fprintf(fileascii, "Number of species in the mechanism : %d\n", *Nspec);
  fprintf(fileascii, "-------------------------------------------");
  for (i = 0; i < *Nelem; i++)
    fprintf(fileascii, "----");
  fprintf(fileascii, "\n");
  fprintf(fileascii,
          "  No.           Name            Mass        Element content\n");
  fprintf(fileascii, "                                          ");
  for (i = 0; i < *Nelem; i++)
    fprintf(fileascii, " %2s ", listelem[i].name);
  fprintf(fileascii, "\n");
  fprintf(fileascii, "-------------------------------------------");
  for (i = 0; i < *Nelem; i++)
    fprintf(fileascii, "----");
  fprintf(fileascii, "\n");

  for (i = 0; i < *Nspec; i++) {
    fprintf(fileascii,
            "  %3d  %18s     %9.3e   ",
            i + 1,
            listspec[i].name,
            listspec[i].mass);

    for (j = 0; j < *Nelem; j++) {
      ipos = *Nelem;
      for (k = 0; k < listspec[i].numofelem; k++)
        if (listspec[i].elemindx[k] == j) {
          ipos = j;
          fprintf(fileascii, "%3d ", listspec[i].elemcontent[k]);
        }
      if (ipos == *Nelem)
        fprintf(fileascii, "%3d ", 0);
    }
    fprintf(fileascii, "\n");
  }
  fprintf(fileascii, "\n");

  /* Output reactions */
  fprintf(fileascii, "Number of reactions in the mechanism : %d\n", *Nreac);
  if (*Nreac == 0)
    return 0;
  fprintf(
    fileascii,
    "---------------------------------------------------------------------\n");
  fprintf(fileascii, "Pre-exponential factor and activation energy units : ");
  if (strncmp(aunits, "MOLE", 4) == 0)
    fprintf(fileascii, "moles & ");
  else
    fprintf(fileascii, "molecules & ");

  if (strncmp(eunits, "CAL/", 4) == 0)
    fprintf(fileascii, "cal/mole\n");
  else if (strncmp(eunits, "KCAL", 4) == 0)
    fprintf(fileascii, "kcal/mole\n");
  else if (strncmp(eunits, "JOUL", 4) == 0)
    fprintf(fileascii, "Joules/mole\n");
  else if (strncmp(eunits, "KJOU", 4) == 0)
    fprintf(fileascii, "kJoules/mole\n");
  else if (strncmp(eunits, "KELV", 4) == 0)
    fprintf(fileascii, "Kelvins\n");
  else if (strncmp(eunits, "EVOL", 4) == 0)
    fprintf(fileascii, "electron Volts\n");

  fprintf(
    fileascii,
    "---------------------------------------------------------------------\n");

  for (i = 0; i < *Nreac; i++) {
    fprintf(fileascii, "%4d%s ", i + 1, "->");

    /* Output reactant string */
    for (j = 0; j < listreac[i].inreac; j++) {
      if (-listreac[i].nuki[j] != 1){
        const int value(listreac[i].nuki[j]);
        if (value == listreac[i].nuki[j])
          fprintf(fileascii, "%d", -value);
        else
          fprintf(fileascii, "%f", -listreac[i].nuki[j]);
      }

      fprintf(fileascii, "%s", listspec[listreac[i].spec[j]].name);

      if (j != listreac[i].inreac - 1)
        fprintf(fileascii, "%s", " + ");
    }

    /* Check for fall-off */
    if ((listreac[i].isfall > 0) && (listreac[i].specfall < 0))
      fprintf(fileascii, "%s", " (+M) ");
    else if ((listreac[i].isfall > 0) && (listreac[i].specfall >= 0))
      fprintf(
        fileascii, "%s%s%s", " (+", listspec[listreac[i].specfall].name, ") ");

    /* Check for third-body */
    if ((listreac[i].isfall == 0) && (listreac[i].isthrdb > 0))
      fprintf(fileascii, "%s", " +M ");

    /* Check for photon activation */
    if (listreac[i].iswl < 0)
      fprintf(fileascii, "%s", " +HV ");

    /* Output delimiter */
    if (listreac[i].isrev > 0)
      fprintf(fileascii, "%s", " <=> ");
    else
      fprintf(fileascii, "%s", " => ");

    /* Output product string */
    for (j = 0; j < listreac[i].inprod; j++) {

      if (listreac[i].nuki[NSPECREACMAX + j] != 1){
        const int value(listreac[i].nuki[NSPECREACMAX + j]);
        if (value == listreac[i].nuki[NSPECREACMAX + j])
          fprintf(fileascii, "%d", value);
        else
          fprintf(fileascii, "%f", listreac[i].nuki[NSPECREACMAX + j]);

      }

      fprintf(
        fileascii, "%s", listspec[listreac[i].spec[NSPECREACMAX + j]].name);

      if (j != listreac[i].inprod - 1)
        fprintf(fileascii, "%s", " + ");
    }
    /* Check for fall-off */
    if ((listreac[i].isfall > 0) && (listreac[i].specfall < 0))
      fprintf(fileascii, "%s", " (+M) ");
    else if ((listreac[i].isfall > 0) && (listreac[i].specfall >= 0))
      fprintf(
        fileascii, "%s%s%s", " (+", listspec[listreac[i].specfall].name, ") ");

    /* Check for third-body */
    if ((listreac[i].isfall == 0) && (listreac[i].isthrdb > 0))
      fprintf(fileascii, "%s", " +M ");

    /* Check for photon activation */
    if (listreac[i].iswl > 0)
      fprintf(fileascii, "%s", " +HV ");

    fprintf(fileascii, "\n");

    /* Duplicate */
    if (listreac[i].isdup > -2) {
      fprintf(fileascii,
              "    --> This reaction is a duplicate %d",
              listreac[i].isdup);
      fprintf(fileascii, "\n");
    }
    /* Arrhenius parameters forward */
    fprintf(fileascii, "    --> Arrhenius coefficients (forward) : ");
    fprintf(fileascii,
            "%11.4e  %5.2f  %11.4e\n",
            listreac[i].arhenfor[0],
            listreac[i].arhenfor[1],
            listreac[i].arhenfor[2]);

    /* Arrhenius parameters reverse */
    if (listreac[i].isrevset > 0) {
      fprintf(fileascii, "    --> Arrhenius coefficients (reverse) : ");
      fprintf(fileascii,
              "%11.4e  %5.2f  %11.4e\n",
              listreac[i].arhenrev[0],
              listreac[i].arhenrev[1],
              listreac[i].arhenrev[2]);
    }

    /* LOW parameters */
    if (listreac[i].islowset > 0) {
      fprintf(fileascii, "    --> LOW parameters  : ");
      for (j = 0; j < 3; j++)
        fprintf(fileascii, "%11.4e  ", listreac[i].fallpar[j]);
      fprintf(fileascii, "\n");
    }

    /* HIGH parameters */
    if (listreac[i].ishighset > 0) {
      fprintf(fileascii, "    --> HIGH parameters : ");
      for (j = 0; j < 3; j++)
        fprintf(fileascii, "%11.4e  ", listreac[i].fallpar[j]);
      fprintf(fileascii, "\n");
    }

    /* TROE parameters */
    if (listreac[i].istroeset > 0) {
      fprintf(fileascii, "    --> TROE parameters : ");
      for (j = 0; j < listreac[i].istroeset; j++)
        fprintf(fileascii, "%11.4e  ", listreac[i].fallpar[3 + j]);
      fprintf(fileascii, "\n");
    }

    /* SRI parameters */
    if (listreac[i].issriset > 0) {
      fprintf(fileascii, "    --> SRI  parameters : ");
      for (j = 0; j < listreac[i].issriset; j++)
        fprintf(fileascii, "%11.4e  ", listreac[i].fallpar[3 + j]);
      fprintf(fileascii, "\n");
    }

    /* PLOG parameters */
    if (listreac[i].isplogset > 0) {
      fprintf(fileascii, "    --> PLOG parameters : \n");
      for (j = 0; j < listreac[i].isplogset; j++) {
        fprintf(fileascii, "        ");
        for (k = 0; k < 4; k++)
          fprintf(fileascii, "%11.4e  ", listreac[i].plog[4 * j + k]);
        fprintf(fileascii, "\n");
      }
    }

    /* Landau-Teller parameters */
    if (listreac[i].isltset > 0) {
      fprintf(fileascii, "    --> LT  parameters  : ");
      for (j = 0; j < 2; j++)
        fprintf(fileascii, "%11.4e  ", listreac[i].ltpar[j]);
      fprintf(fileascii, "\n");
    }

    /* reverse Landau-Teller parameters */
    if (listreac[i].isrltset > 0) {
      fprintf(fileascii, "    --> RLT  parameters : ");
      for (j = 0; j < 2; j++)
        fprintf(fileascii, "%11.4e  ", listreac[i].rltpar[j]);
      fprintf(fileascii, "\n");
    }

    /* specific species temperature dependency */
    if (listreac[i].istdepset > 0) {
      fprintf(fileascii, "    --> Rates based on temperature of    : ");
      fprintf(fileascii, "%s  ", listspec[listreac[i].tdeppar].name);
      fprintf(fileascii, "\n");
    }

    /* energy loss parameter */
    if (listreac[i].isexciset > 0) {
      fprintf(fileascii, "    --> Energy loss parameter            : ");
      fprintf(fileascii, "%11.4e", listreac[i].excipar);
      fprintf(fileascii, "\n");
    }

    /* optional rate-fit parameters */
    if (listreac[i].isjanset > 0) {
      fprintf(fileascii, "    --> JAN parameters  : ");
      for (j = 0; j < 9; j++) {
        fprintf(fileascii, "%9.2e  ", listreac[i].optfit[j]);
        if ((j == 3) || (j == 7))
          fprintf(fileascii, "\n                          ");
      }
      fprintf(fileascii, "\n");
    } else if (listreac[i].isfit1set > 0) {
      fprintf(fileascii, "    --> FIT1 parameters  : ");
      for (j = 0; j < 4; j++)
        fprintf(fileascii, "%9.2e  ", listreac[i].optfit[j]);
      fprintf(fileascii, "\n");
    }

    /* Arbitrary reaction orders */
    if (listreac[i].isford > 0) {
      fprintf(fileascii, "    --> Change in reaction order    : Forward");
      fprintf(fileascii, "\n                                      ");
      for (j = 0; j < listreac[i].isford; j++) {
        if (j < listreac[i].inreac) {
          if (listreac[i].isreal > 0) {
            if (fabs(listreac[i].rnuki[j] - listreac[i].arbnuki[j]) >
                REACBALANCE()) {
              fprintf(fileascii,
                      " %-10.18s : %5.2f",
                      listspec[listreac[i].arbspec[j]].name,
                      listreac[i].arbnuki[j]);
              fprintf(fileascii, "\n                                      ");
            }
          } else {
            if (listreac[i].nuki[j] != (int)listreac[i].arbnuki[j]) {
              fprintf(fileascii,
                      " %-10.18s : %5.2f",
                      listspec[listreac[i].arbspec[j]].name,
                      listreac[i].arbnuki[j]);
              fprintf(fileascii, "\n                                      ");
            }
          }
        } else {
          fprintf(fileascii,
                  " %-10.18s : %5.2f",
                  listspec[listreac[i].arbspec[j]].name,
                  listreac[i].arbnuki[j]);
          fprintf(fileascii, "\n                                      ");
        }
      }

      fprintf(fileascii, "Reverse");
      fprintf(fileascii, "\n                                      ");

      for (j = 0; j < listreac[i].isrord; j++) {
        if (j < listreac[i].inprod) {
          if (listreac[i].isreal > 0) {
            if (fabs(listreac[i].rnuki[NSPECREACMAX + j] -
                     listreac[i].arbnuki[2 * NSPECREACMAX + j]) > REACBALANCE()) {
              fprintf(fileascii,
                      " %-10.18s : %5.2f",
                      listspec[listreac[i].arbspec[2 * NSPECREACMAX + j]].name,
                      listreac[i].arbnuki[2 * NSPECREACMAX + j]);
              fprintf(fileascii, "\n                                      ");
            }
          } else {
            if (listreac[i].nuki[NSPECREACMAX + j] !=
                (int)listreac[i].arbnuki[2 * NSPECREACMAX + j]) {
              fprintf(fileascii,
                      " %-10.18s : %5.2f",
                      listspec[listreac[i].arbspec[2 * NSPECREACMAX + j]].name,
                      listreac[i].arbnuki[2 * NSPECREACMAX + j]);
              fprintf(fileascii, "\n                                      ");
            }
          }
        } else {
          fprintf(fileascii,
                  " %-10.18s : %5.2f",
                  listspec[listreac[i].arbspec[2 * NSPECREACMAX + j]].name,
                  listreac[i].arbnuki[2 * NSPECREACMAX + j]);
          fprintf(fileascii, "\n                                      ");
        }
      }

      fprintf(fileascii, "\n");

    } /* Done checking arbitrary reaction orders */

    /* Enhanced third-body efficiencies */
    if (listreac[i].nthrdb > 0) {
      fprintf(fileascii, "    --> Enhanced third-body efficiencies : \n");
      fprintf(fileascii, "        ");
      for (j = 0; j < listreac[i].nthrdb; j++) {
        if (j == 5) {
          fprintf(fileascii, "\n");
          fprintf(fileascii, "        ");
        }
        fprintf(fileascii,
                "%s: %5.3f,  ",
                listspec[listreac[i].ithrdb[j]].name,
                listreac[i].rthrdb[j]);
      }
      fprintf(fileascii, "\n");
    }

    if ((strncmp(listreac[i].eunits, eunits, 4) != 0) ||
        (strncmp(listreac[i].aunits, aunits, 4) != 0)) {
      fprintf(fileascii, "    --> Units : ");
      if (strncmp(listreac[i].aunits, "MOLE", 4) == 0)
        fprintf(fileascii, "moles & ");
      else
        fprintf(fileascii, "molecules & ");

      if (strncmp(listreac[i].eunits, "CAL/", 4) == 0)
        fprintf(fileascii, "cal/mole\n");
      else if (strncmp(listreac[i].eunits, "KCAL", 4) == 0)
        fprintf(fileascii, "kcal/mole\n");
      else if (strncmp(listreac[i].eunits, "JOUL", 4) == 0)
        fprintf(fileascii, "Joules/mole\n");
      else if (strncmp(listreac[i].eunits, "KJOU", 4) == 0)
        fprintf(fileascii, "kJoules/mole\n");
      else if (strncmp(listreac[i].eunits, "KELV", 4) == 0)
        fprintf(fileascii, "Kelvins\n");
      else if (strncmp(listreac[i].eunits, "EVOL", 4) == 0)
        fprintf(fileascii, "electron Volts\n");
    }

    fprintf(fileascii, "\n");
  }

  return (0);

} /* Done with "TCKMI_outform" */

/* -------------------------------------------------------------------------  */
/**
 * \brief Outputs reaction data to an unformatted ascii file
 */
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
                int* ierror)
{
  int i, j, k;

  /* integer parameters */
  int nIonEspec, electrIndx, nIonSpec, maxSpecInReac, maxTbInReac, maxOrdPar,
    nFallPar, maxTpRange;
  int nLtReac, nRltReac, nFallReac, nPlogReac, nThbReac, nRevReac, nHvReac,
    nTdepReac, nJanReac, nFit1Reac, nExciReac;
  int nMomeReac, nXsmiReac, nRealNuReac, nOrdReac, nNASAinter, nCpCoef,
    nNASAfit, nArhPar;
  int nLtPar, nJanPar, nFit1Par, nNASA9coef;

  if (*ierror == 0)
    fprintf(filelist, "SUCCESS\n");
  else {
    fprintf(filelist, "ERROR  \n");
    return (1);
  }

  /* compute kinetic model summary */
  TCKMI_kmodsum(listelem,
                Nelem,
                listspec,
                Nspec,
                listreac,
                Nreac,
                &nIonEspec,
                &electrIndx,
                &nIonSpec,
                &maxSpecInReac,
                &maxTbInReac,
                &maxOrdPar,
                &nFallPar,
                &maxTpRange,
                &nLtReac,
                &nRltReac,
                &nFallReac,
                &nPlogReac,
                &nThbReac,
                &nRevReac,
                &nHvReac,
                &nTdepReac,
                &nJanReac,
                &nFit1Reac,
                &nExciReac,
                &nMomeReac,
                &nXsmiReac,
                &nRealNuReac,
                &nOrdReac,
                &nNASAinter,
                &nCpCoef,
                &nNASAfit,
                &nArhPar,
                &nLtPar,
                &nJanPar,
                &nFit1Par,
                &nNASA9coef);

  fprintf(filelist,
          "%12d\n",
          maxSpecInReac); /* Maximum number of species in a reaction */
  fprintf(
    filelist,
    "%12d\n",
    maxTbInReac); /* Maximum number of third-body efficiencies in a reaction */
  fprintf(filelist,
          "%12d\n",
          nNASAinter); /* Number of temperature regions for thermo fits */
  fprintf(filelist,
          "%12d\n",
          nCpCoef); /* Number of polynomial coefficients for thermo fits */
  fprintf(filelist, "%12d\n", nArhPar); /* Number of Arrhenius parameters */
  fprintf(filelist,
          "%12d\n",
          nLtPar); /* Number of parameters for Landau-Teller reactions */
  fprintf(filelist,
          "%12d\n",
          nFallPar); /* Number of parameters for pressure-dependent reactions */
  fprintf(filelist,
          "%12d\n",
          nJanPar); /* Number of parameters for Jannev-Langer fits (JAN) */
  fprintf(filelist,
          "%12d\n",
          maxOrdPar); /* Number of parameters for arbitrary order reactions */
  fprintf(
    filelist, "%12d\n", nFit1Par); /* Number of parameters for FIT1 fits */

  fprintf(filelist, "%12d\n", *Nelem); /* Number of elements */
  fprintf(filelist, "%12d\n", *Nspec); /* Number of species */
  fprintf(filelist, "%12d\n", *Nreac); /* Number of reactions */
  fprintf(
    filelist, "%12d\n", nRevReac); /* Number of reactions with REV given */
  fprintf(
    filelist, "%12d\n", nFallReac); /* Number of pressure-dependent reactions */
  fprintf(filelist,
          "%12d\n",
          nPlogReac); /* Number of reactions with PLOG expressions */
  fprintf(filelist,
          "%12d\n",
          nThbReac); /* Number of reactions using third-body efficiencies */
  fprintf(filelist, "%12d\n", nLtReac); /* Number of Landau-Teller reactions */
  fprintf(filelist,
          "%12d\n",
          nRltReac); /* Number of Landau-Teller reactions with RLT given */
  fprintf(filelist, "%12d\n", nHvReac);  /* Number of reactions with HV */
  fprintf(filelist, "%12d\n", nIonSpec); /* Number of ion species */
  fprintf(filelist, "%12d\n", nJanReac); /* Number of reactions with JAN fits */
  fprintf(
    filelist, "%12d\n", nFit1Reac); /* Number of reactions with FIT1 fits */
  fprintf(filelist, "%12d\n", nExciReac); /* Number of reactions with EXCI */
  fprintf(filelist, "%12d\n", nMomeReac); /* Number of reactions with MOME */
  fprintf(filelist, "%12d\n", nXsmiReac); /* Number of reactions with XSMI */
  fprintf(filelist, "%12d\n", nTdepReac); /* Number of reactions with TDEP */
  fprintf(
    filelist, "%12d\n", nRealNuReac); /* Number of reactions with non-integer
                                         stoichiometric coefficients */
  fprintf(filelist,
          "%12d\n",
          nOrdReac); /* Number of reactions with arbitrary order */
  fprintf(filelist, "%12d\n", electrIndx); /* Index of the electron species */
  fprintf(filelist,
          "%12d\n",
          nIonEspec); /* Number of ion species excluding the electron species */
  fprintf(
    filelist,
    "%12d\n",
    nNASA9coef); /* Number of species with 9-coefficients polynomial fits */

  /* Tolerance for reaction balance */
  fprintf(filelist, "%24.16e\n", REACBALANCE());

  /* Elements' name and weights */
  for (i = 0; i < (*Nelem); i++)
    fprintf(filelist, "%-16s\n", listelem[i].name);
  for (i = 0; i < (*Nelem); i++)
    fprintf(filelist, "%24.16e\n", listelem[i].mass);

  /* Species' name and weights */
  for (i = 0; i < (*Nspec); i++)
    fprintf(filelist, "%-32s\n", listspec[i].name);
  for (i = 0; i < (*Nspec); i++)
    fprintf(filelist, "%24.16e\n", listspec[i].mass);

  /* Species elemental compositions */
  for (i = 0; i < (*Nspec); i++) {
    for (j = 0; j < (*Nelem); j++) {
      int ipos = (*Nelem);
      for (k = 0; k < listspec[i].numofelem; k++)
        if (listspec[i].elemindx[k] == j) {
          ipos = j;
          fprintf(filelist, "%12d\n", listspec[i].elemcontent[k]);
        }
      if (ipos == *Nelem)
        fprintf(filelist, "%12d\n", 0);

    } /* Done loop over elements */

  } /* Done loop over species */

  /* Species charges */
  for (i = 0; i < (*Nspec); i++)
    fprintf(filelist, "%12d\n", listspec[i].charge);

  /* Number of temperature regions for thermo fits (2 for now) */
  for (i = 0; i < (*Nspec); i++)
    fprintf(filelist, "%12d\n", 2);

  /* Species phase (-1=solid,0=gas,+1=liquid) */
  for (i = 0; i < (*Nspec); i++)
    fprintf(filelist, "%12d\n", listspec[i].phase);

  /* Temperature for thermo fits ranges */
  for (i = 0; i < (*Nspec); i++) {
    fprintf(
      filelist, "%24.16e\n", listspec[i].nasapoltemp[0]); /* lower  bound */
    fprintf(
      filelist, "%24.16e\n", listspec[i].nasapoltemp[2]); /* middle bound */
    fprintf(
      filelist, "%24.16e\n", listspec[i].nasapoltemp[1]); /* upper  bound */
  }

  /* Polynomial coeffs for thermo fits */
  for (i = 0; i < (*Nspec); i++)
    for (j = 0; j < nNASAinter; j++)
      for (k = 0; k < nNASAfit; k++)
        fprintf(
          filelist, "%24.16e\n", listspec[i].nasapolcoefs[j * nNASAfit + k]);

  /* Polynomial coeffs for 9-term thermo fits */
  if (nNASA9coef > 0) {
    fprintf(filelist, "%12d\n", nNASA9coef);
    for (i = 0; i < (*Nspec); i++) {
      if (listspec[i].hasthermo == 2) {
        fprintf(filelist, "%12d\n", i + 1);
        fprintf(filelist, "%12d\n", listspec[i].Nth9rng);
        for (j = 0; j < listspec[i].Nth9rng; j++) {

          fprintf(filelist,
                  "%24.16e\n%24.16e\n",
                  listspec[i].Nth9Temp[2 * j],
                  listspec[i].Nth9Temp[2 * j + 1]);
          for (k = 0; k < 9; k++)
            fprintf(filelist, "%24.16e\n", listspec[i].Nth9coefs[j * 9 + k]);

        } /* done loop over temperature ranges */
      }   /* done if species has 9-coefficients thermo props */
    }     /* done loop over all species */
  }       /* done if for species with 9-coefficients thermo props */

  /* Ionic species */
  if (nIonEspec > 0) {
    fprintf(filelist, "%12d\n", nIonEspec);
    for (i = 0; i < (*Nspec); i++) {
      if ((listspec[i].charge != 0) && (strncmp(listspec[i].name, "E", 1) != 0))
        fprintf(filelist, "%12d\n", i + 1);
    }
  }

  /* Basic reaction info */
  if ((*Nreac) > 0) {
    /* No. of reactants+products (negative values for irreversible reactions) */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isrev < 0)
        fprintf(filelist, "%12d\n", -(listreac[i].inreac + listreac[i].inprod));
      else
        fprintf(filelist, "%12d\n", listreac[i].inreac + listreac[i].inprod);
    }

    /* No. of reactants only */
    for (i = 0; i < (*Nreac); i++)
      fprintf(filelist, "%12d\n", listreac[i].inreac);

    /* Stoichiometric coefficients */
    for (i = 0; i < (*Nreac); i++) {
      double nusumk = 0;
      for (j = 0; j < maxSpecInReac; j++) {
        if (j < maxSpecInReac / 2) {
          fprintf(filelist, "%12f\n", listreac[i].nuki[j]);
          fprintf(filelist, "%12d\n", listreac[i].spec[j] + 1);
          nusumk += listreac[i].nuki[j];
        } else {
          int j1;
          j1 = j - maxSpecInReac / 2;
          fprintf(filelist, "%12f\n", listreac[i].nuki[NSPECREACMAX + j1]);
          fprintf(filelist, "%12d\n", listreac[i].spec[NSPECREACMAX + j1] + 1);
          nusumk += listreac[i].nuki[NSPECREACMAX + j1];
        }
      }
      fprintf(filelist, "%12f\n", nusumk);

    } /* Done with stoichimetric coefficients */

    /* Arrhenius parameters */
    for (i = 0; i < (*Nreac); i++) {
      fprintf(filelist, "%24.16e\n", listreac[i].arhenfor[0]);
      fprintf(filelist, "%24.16e\n", listreac[i].arhenfor[1]);
      fprintf(filelist, "%24.16e\n", listreac[i].arhenfor[2]);
    }

    /* If reactions are duplicates or not */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isdup > -2)
        fprintf(filelist, "1\n"); /* yes */
      else
        fprintf(filelist, "0\n"); /* no  */
    }

  } /* Done if Nreac > 0 */

  /* Reactions with reversible Arrhenius parameters given */
  if (nRevReac > 0) {
    /* No. of such reactions */
    fprintf(filelist, "%12d\n", nRevReac);

    /* Their indices */
    for (i = 0; i < (*Nreac); i++)
      if (listreac[i].isrevset > 0)
        fprintf(filelist, "%12d\n", i + 1);

    /* reverse Arrhenius parameters */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isrevset > 0) {
        fprintf(filelist, "%24.16e\n", listreac[i].arhenrev[0]);
        fprintf(filelist, "%24.16e\n", listreac[i].arhenrev[1]);
        fprintf(filelist, "%24.16e\n", listreac[i].arhenrev[2]);
      }
    }

  } /* Done if nRevReac > 0 */

  /* Pressure-dependent reactions */
  if (nFallReac > 0) {
    fprintf(filelist, "%12d\n", nFallReac);
    fprintf(filelist, "%12d\n", nFallPar);

    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isfall > 0) {
        /* Reaction index */
        fprintf(filelist, "%12d\n", i + 1);

        /* Type for fall-off reaction */
        if ((listreac[i].istroeset == 0) && (listreac[i].issriset == 0))
          /* Lindemann form */
          fprintf(filelist, "%12d\n", 1);
        else if (listreac[i].issriset > 0)
          /* SRI form */
          fprintf(filelist, "%12d\n", 2);
        else if (listreac[i].istroeset > 0)
          /* TROE form */
          fprintf(filelist, "%12d\n", listreac[i].istroeset);
        else {
          printf("Unknown pressure dependent type for reaction %d\n", i);
          exit(-1);
        }

        /* Type of low/high */
        if (listreac[i].islowset == 1)
          /* LOW */
          fprintf(filelist, "%12d\n", 0);
        else
          /* HIGH */
          fprintf(filelist, "%12d\n", 1);

        /* fall-off species */
        fprintf(filelist, "%12d\n", listreac[i].specfall + 1);
      }

    } /* Done printing fall-off indices */

    /* Print fall-off parameters */
    for (i = 0; i < (*Nreac); i++) {
      int maxpars = 0;
      if (listreac[i].isfall > 0) {
        if ((listreac[i].islowset > 0) || (listreac[i].ishighset > 0)) {
          /* LOW or HIGH */
          maxpars = 3;
          for (j = 0; j < 3; j++)
            fprintf(filelist, "%24.16e\n", listreac[i].fallpar[j]);

          if (listreac[i].istroeset > 0) {
            /* TROE */
            maxpars = 3 + listreac[i].istroeset;
            for (j = 0; j < listreac[i].istroeset; j++)
              fprintf(filelist, "%24.16e\n", listreac[i].fallpar[3 + j]);
          } else if (listreac[i].issriset > 0) {
            /* SRI */
            maxpars = 8;
            for (j = 0; j < 5; j++)
              fprintf(filelist, "%24.16e\n", listreac[i].fallpar[3 + j]);
          }
        } /* Done LOW/HIGH */
        else {
          printf("Unknown pressure dependent type for reaction %d\n", i);
          exit(-1);
        }

        for (j = maxpars; j < nFallPar; j++)
          fprintf(filelist, "%24.16e\n", 0.0);
      }

    } /* Done printing fall-off data */

  } /* Done if fall-off reactions */

  /* Third-body reactions */
  if (nThbReac > 0) {
    fprintf(filelist, "%12d\n%12d\n", nThbReac, maxTbInReac);

    /* Reaction index and # of third body spec */
    for (i = 0; i < (*Nreac); i++)
      if (listreac[i].isthrdb > 0)
        fprintf(filelist, "%12d\n%12d\n", i + 1, listreac[i].nthrdb);

    /* Species indices */
    for (i = 0; i < (*Nreac); i++)
      if (listreac[i].isthrdb > 0) {
        for (j = 0; j < maxTbInReac; j++) {
          if (j < listreac[i].nthrdb)
            fprintf(filelist, "%12d\n", listreac[i].ithrdb[j] + 1);
          else
            fprintf(filelist, "%12d\n", 0);
        }
      }

    /* Species enhanced efficiency */
    for (i = 0; i < (*Nreac); i++)
      if (listreac[i].isthrdb > 0) {
        for (j = 0; j < maxTbInReac; j++) {
          fprintf(filelist, "%24.16e\n", listreac[i].rthrdb[j]);
        }
      }
  }

  /* Reactions with real stoichiometric coefficients */
  if (nRealNuReac > 0) {
    fprintf(filelist, "%12d\n", nRealNuReac);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isreal > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    /* Stoichiometric coefficients */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isreal > 0) {
        double nusumk = 0;
        for (j = 0; j < maxSpecInReac; j++) {
          if (j < listreac[i].inreac) {
            fprintf(filelist, "%24.16e\n", listreac[i].rnuki[j]);
            nusumk += listreac[i].rnuki[j];
          } else if (j < listreac[i].inreac + listreac[i].inprod) {
            int j1;
            j1 = j - listreac[i].inreac;
            fprintf(
              filelist, "%24.16e\n", listreac[i].rnuki[NSPECREACMAX + j1]);
            nusumk += listreac[i].rnuki[NSPECREACMAX + j1];
          } else
            fprintf(filelist, "%24.16e\n", 0.0);
        }
        fprintf(filelist, "%24.16e\n", nusumk);
      }

    } /* Done with stoichimetric coefficients */
  }

  /* Arbitrary reaction orders */
  if (nOrdReac > 0) {
    fprintf(filelist, "%12d\n%12d\n", nOrdReac, maxOrdPar);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isford > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    /* Species indices */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isford > 0) {
        for (j = 0; j < listreac[i].isford; j++)
          fprintf(filelist, "%12d\n", -(listreac[i].arbspec[j] + 1));
        for (j = 0; j < listreac[i].isrord; j++)
          fprintf(
            filelist, "%12d\n", listreac[i].arbspec[2 * NSPECREACMAX + j] + 1);
        for (j = listreac[i].isford + listreac[i].isrord; j < maxOrdPar; j++)
          fprintf(filelist, "%12d\n", 0);
      }
    }
    /* Species orders */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isford > 0) {
        for (j = 0; j < listreac[i].isford; j++)
          fprintf(filelist, "%24.16e\n", listreac[i].arbnuki[j]);
        for (j = 0; j < listreac[i].isrord; j++)
          fprintf(
            filelist, "%24.16e\n", listreac[i].arbnuki[2 * NSPECREACMAX + j]);
        for (j = listreac[i].isford + listreac[i].isrord; j < maxOrdPar; j++)
          fprintf(filelist, "%24.16e\n", 0.0);
      }
    }
  }

  /* Landau-Teller reactions */
  if (nLtReac > 0) {
    fprintf(filelist, "%12d\n%12d\n", nLtReac, nLtPar);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isltset > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isltset > 0) {
        fprintf(filelist, "%24.16e\n", listreac[i].ltpar[0]);
        fprintf(filelist, "%24.16e\n", listreac[i].ltpar[1]);
      }
    }
  }

  /* reverse Landau-Teller reactions */
  if (nRltReac > 0) {
    fprintf(filelist, "%12d\n%12d\n", nRltReac, nLtPar);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isrltset > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isrltset > 0) {
        fprintf(filelist, "%24.16e\n", listreac[i].rltpar[0]);
        fprintf(filelist, "%24.16e\n", listreac[i].rltpar[1]);
      }
    }
  }

  /* radiation wavelength */
  if (nHvReac > 0) {
    fprintf(filelist, "%12d\n", nHvReac);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].iswl > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].iswl > 0)
        fprintf(filelist, "%24.16e\n", listreac[i].hvpar);
    }
  }

  /* JAN fits */
  if (nJanReac > 0) {
    fprintf(filelist, "%12d\n%12d\n", nJanReac, nJanPar);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isjanset > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isjanset > 0)
        for (j = 0; j < nJanPar; j++)
          fprintf(filelist, "%24.16e\n", listreac[i].optfit[j]);
    }
  }

  /* FIT1 fits */
  if (nFit1Reac > 0) {
    fprintf(filelist, "%12d\n%12d\n", nFit1Reac, nFit1Par);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isfit1set > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isfit1set > 0)
        for (j = 0; j < nFit1Par; j++)
          fprintf(filelist, "%24.16e\n", listreac[i].optfit[j]);
    }
  }

  /* Excitation reactions */
  if (nExciReac > 0) {
    fprintf(filelist, "%12d\n", nExciReac);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isexciset > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isexciset > 0)
        fprintf(filelist, "%24.16e\n", listreac[i].excipar);
    }
  }

  /* Plasma Momentum transfer collision */
  if (nMomeReac > 0) {
    fprintf(filelist, "%12d\n", nMomeReac);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].ismome > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].ismome > 0) {
        for (j = 0; j < listreac[i].inreac; j++)
          if (listreac[i].spec[j] != (electrIndx - 1)) {
            fprintf(filelist, "%12d\n", listreac[i].spec[j] + 1);
          }
      }
    }
  }

  /* Ion Momentum transfer collision */
  if (nXsmiReac > 0) {
    fprintf(filelist, "%12d\n", nXsmiReac);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isxsmi > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    /* Ion indices */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isxsmi > 0) {
        for (j = 0; j < listreac[i].inreac; j++)
          if (listspec[listreac[i].spec[j]].charge != 0) {
            fprintf(filelist, "%12d\n", listreac[i].spec[j] + 1);
          }
      }
    }
    /* Ion partner indices */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isxsmi > 0) {
        for (j = 0; j < listreac[i].inreac; j++)
          if (listspec[listreac[i].spec[j]].charge == 0) {
            fprintf(filelist, "%12d\n", listreac[i].spec[j] + 1);
          }
      }
    }
  }

  /* Species temperature dependency */
  if (nTdepReac > 0) {
    fprintf(filelist, "%12d\n", nTdepReac);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].istdepset > 0)
        fprintf(filelist, "%12d\n", i + 1);
    }

    /* Species index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].istdepset > 0)
        fprintf(filelist, "%12d\n", listreac[i].tdeppar + 1);
    }
  }

  /* Reactions with PLOG formulation */
  if (nPlogReac > 0) {
    /* No. of such reactions */
    fprintf(filelist, "%12d\n", nPlogReac);

    /* Their indices and no of PLOG intervals */
    for (i = 0; i < (*Nreac); i++)
      if (listreac[i].isplogset > 0) {
        fprintf(filelist, "%12d\n", i + 1);
        fprintf(filelist, "%12d\n", listreac[i].isplogset);
      }

    for (i = 0; i < (*Nreac); i++)
      if (listreac[i].isplogset > 0)
        for (j = 0; j < listreac[i].isplogset; j++)
          for (k = 0; k < 4; k++)
            fprintf(filelist, "%24.16e\n", listreac[i].plog[4 * j + k]);

  } /* Done if nPlogReac > 0 */

  return (0);

} /* Done with "TCKMI_outunform" */

/* ------------------------------------------------------------------------- */
/**
 * \brief Outputs error messages
 */
void
TCKMI_errormsg(int ierror)
{
  if (ierror == 10) {
    printf("File bug -> Encountered ELEM keyword after SPEC, THERM,\n ");
    printf("            or REAC\n");
  } else if (ierror == 20) {
    printf("File bug -> Encountered SPEC keyword after THERM or REAC\n ");
  }

  printf("Found error : %d\n ", ierror);

  /* Element related errors */
  if (ierror == 100)
    printf("Error %d: Element name is too long !\n", ierror);
  else if (ierror == 105)
    printf("Error %d: Found element mass before any element was specified !\n",
           ierror);
  else if (ierror == 110)
    printf(
      "Error %d: Found an odd number of / separators on an element line !\n",
      ierror);
  else if (ierror == 150)
    printf("Error %d: Could not find element in the periodic table!\n", ierror);

  /* Species related errors */
  if (ierror == 210)
    printf("Error %d: Species keyword line contains / character !\n", ierror);
  if (ierror == 220)
    printf("Error %d: Species name longer than 18 characters !\n", ierror);
  if (ierror == 240)
    printf("Error %d: Species contains elements that were undeclared !\n",
           ierror);

  /* Thermo data related errors */
  if (ierror == 300)
    printf("Error %d: Reading thermo data with the wrong flag !\n", ierror);
  else if (ierror == 310)
    printf(
      "Error %d: Encountered \"thermo all\" but thermo data was incomplete !\n",
      ierror);
  else if (ierror == 320)
    printf("Error %d: Could not open thermodynamic source file !\n", ierror);
  else if (ierror == 330)
    printf("Error %d: Incomplete thermodynamic data, need at least temperature "
           "range !\n",
           ierror);
  else if (ierror == 332)
    printf("Error %d: Incomplete thermodynamic data, need at least temperature "
           "range !\n",
           ierror);
  else if (ierror == 334)
    printf("Error %d: Incomplete thermodynamic data, need at least temperature "
           "range !\n",
           ierror);
  else if (ierror == 336)
    printf("Error %d: Incomplete temperature range for thermodynamic "
           "coefficients!\n",
           ierror);
  else if (ierror == 338)
    printf("Error %d: Premature end of the thermodynamic file !\n", ierror);
  else if (ierror == 340)
    printf("Error %d: Premature end of the thermodynamic file !\n", ierror);
  else if (ierror == 342)
    printf("Error %d: Premature end of the thermodynamic file !\n", ierror);
  else if (ierror == 344)
    printf("Error %d: Premature end of the thermodynamic file !\n", ierror);
  else if (ierror == 346)
    printf("Error %d: Illegal characters on the line containing the "
           "temperature range !\n",
           ierror);
  else if (ierror == 360)
    printf("Error %d: Encountered end if thermodynamic file -> ncomplete "
           "thermodynamic data !\n",
           ierror);
  else if (ierror == 370)
    printf("Error %d: Encountered null species name in the thermodynamic data "
           "card !\n",
           ierror);
  else if (ierror == 380)
    printf("Error %d: Species contains undeclared elements !\n", ierror);
  else if (ierror == 390)
    printf("Error %d: Illegal characters on in the thermo data lines !\n",
           ierror);

  /* Reaction errors */
  if (ierror == 500) {
    printf("Error %d: Cannot have auxiliary reaction data before reading the "
           "equation \n",
           ierror);
    printf("          for at least one reaction !\n");
  }
  if (ierror == 605)
    printf("Error %d: Encountered null strings for reactants and/or products\n",
           ierror);
  else if (ierror == 610)
    printf("Error %d: Could not find all Arrhenius parameters!\n", ierror);
  else if (ierror == 620)
    printf("Error %d: Reaction equation is too small!\n", ierror);
  else if (ierror == 630)
    printf(
      "Error %d: Could not find delimiter between reactants and products!\n",
      ierror);

  if (ierror == 630)
    printf(
      "Error %d: Could not find delimiter between reactants and products!\n",
      ierror);
  else if (ierror == 640)
    printf(
      "Error %d: Encountered null strings for reactants and/or products!\n",
      ierror);
  else if (ierror == 650)
    printf("Error %d: Third-body indicator is incomplete!\n", ierror);
  else if (ierror == 660)
    printf("Error %d: Third-body indicator is incomplete!\n", ierror);

  if (ierror == 670)
    printf(
      "Error %d: Found more than one third-body indicator in a reaction!\n",
      ierror);
  else if (ierror == 680)
    printf("Error %d: Third-body indicator in a species that is not declared "
           "in the list of species!\n",
           ierror);
  else if (ierror == 690)
    printf(
      "Error %d: Found more than one third-body indicator in a reaction!\n",
      ierror);
  else if (ierror == 615)
    printf("Error %d: Found null species name !\n", ierror);

  if (ierror == 625)
    printf(
      "Error %d: Found more than one third-body indicator in a reaction!\n",
      ierror);
  else if (ierror == 635)
    printf(
      "Error %d: Found more than one third-body indicator in a reaction!\n",
      ierror);
  else if (ierror == 636)
    printf("Error %d: Found more than one HV keywords in the reactants or "
           "products!\n",
           ierror);
  else if (ierror == 645)
    printf("Error %d: Number of reactants is more than the maximum allowed!\n",
           ierror);
  else if (ierror == 655)
    printf("Error %d: Number of products is more than the maximum allowed!\n",
           ierror);
  else if (ierror == 665)
    printf("Error %d: Reactant/Product species was not found in the list of "
           "species!\n",
           ierror);

  if (ierror == 710)
    printf(
      "Error %d: Could not find any info on the auxiliary reaction line!\n",
      ierror);
  else if (ierror == 711)
    printf(
      "Error %d: Incomplete keyword info on the auxiliary reaction line!\n",
      ierror);
  else if (ierror == 712)
    printf("Error %d: Incorrect DUPLICATE/MOME/XSMI info!\n", ierror);
  else if (ierror == 715)
    printf(
      "Error %d: Encountered more than one LOW/HIGH keywords for a reaction!\n",
      ierror);
  else if (ierror == 716)
    printf("Error %d: Encountered when parsing LOW parameters!\n", ierror);

  if (ierror == 717)
    printf("Error %d: Encountered less or more than 3 LOW parameters!\n",
           ierror);
  else if (ierror == 780)
    printf(
      "Error %d: Encountered more than one LOW/HIGH keywords for a reaction!\n",
      ierror);
  else if (ierror == 781)
    printf("Error %d: Encountered when parsing HIGH parameters!\n", ierror);
  else if (ierror == 782)
    printf("Error %d: Encountered less or more than 3 HIGH parameters!\n",
           ierror);
  else if (ierror == 720)
    printf(
      "Error %d: Encountered more than one TROE keywords for a reaction!\n",
      ierror);
  else if (ierror == 721)
    printf("Error %d: Encountered when parsing TROE parameters!\n", ierror);
  else if (ierror == 722)
    printf(
      "Error %d: Encountered less than 3 or more than 4 TROE parameters!\n",
      ierror);

  if (ierror == 725)
    printf(
      "Error %d: Encountered more than one SRI/TROE keywords for a reaction!\n",
      ierror);
  else if (ierror == 726)
    printf("Error %d: Encountered when parsing SRI parameters!\n", ierror);
  else if (ierror == 727)
    printf("Error %d: Encountered a number of SRI parameters different than 3 "
           "or 5!\n",
           ierror);
  else if (ierror == 730)
    printf("Error %d: Encountered more than one REV keywords for a reaction!\n",
           ierror);
  else if (ierror == 731)
    printf("Error %d: Encountered when parsing REV parameters!\n", ierror);
  else if (ierror == 732)
    printf("Error %d: Encountered less or more than 3 REV parameters!\n",
           ierror);

  if (ierror == 735)
    printf("Error %d: Encountered more than one LT keywords for a reaction!\n",
           ierror);
  else if (ierror == 736)
    printf("Error %d: Encountered when parsing LT parameters!\n", ierror);
  else if (ierror == 737)
    printf("Error %d: Encountered less or more than 2 LT parameters!\n",
           ierror);
  else if (ierror == 740)
    printf("Error %d: Encountered more than one RLT keywords for a reaction!\n",
           ierror);
  else if (ierror == 741)
    printf("Error %d: Encountered when parsing RLT parameters!\n", ierror);
  else if (ierror == 742)
    printf("Error %d: Encountered less or more than 3 LT parameters!\n",
           ierror);

  if (ierror == 745)
    printf("Error %d: Encountered more than one HV keywords for a reaction!\n",
           ierror);
  else if (ierror == 746)
    printf("Error %d: Encountered when parsing HV parameters!\n", ierror);
  else if (ierror == 747)
    printf("Error %d: Encountered less or more than 1 HV parameters!\n",
           ierror);
  else if (ierror == 750)
    printf(
      "Error %d: Could not find third body species in the list of species!\n",
      ierror);
  else if (ierror == 751)
    printf(
      "Error %d: Found more enhanced efficiencies than the maximum allowed!\n",
      ierror);
  else if (ierror == 752)
    printf(
      "Error %d: Found repeated species in the list of third body species!\n",
      ierror);
  else if (ierror == 753)
    printf("Error %d: Encountered when parsing third body efficiency!\n",
           ierror);
  else if (ierror == 754)
    printf("Error %d: Found more than one value for a species third body "
           "efficiency!\n",
           ierror);

  if (ierror == 701)
    printf("Error %d: Encountered when looking for numeric values on auxiliary "
           "reaction lines!\n",
           ierror);
  else if (ierror == 702)
    printf("Error %d: Encountered when looking for numeric values on auxiliary "
           "reaction lines!\n",
           ierror);
  else if (ierror == 703)
    printf("Error %d: Encountered when looking for numeric values on auxiliary "
           "reaction lines!\n",
           ierror);
  else if (ierror == 760)
    printf("Error %d: Description field empty for units keyword!\n", ierror);
  else if (ierror == 761)
    printf("Error %d: Unknown units encountered for units keyword!\n", ierror);

  if (ierror == 770)
    printf("Error %d: Duplicate TDEP keyword found!\n", ierror);
  else if (ierror == 771)
    printf("Error %d: TDEP species not declared in the list of species!\n",
           ierror);
  else if (ierror == 775)
    printf("Error %d: Duplicate EXCI keyword found!\n", ierror);
  else if (ierror == 776)
    printf("Error %d: Error when extracting EXCI value!\n", ierror);
  else if (ierror == 777)
    printf("Error %d: Found more than one EXCI value!\n", ierror);

  if (ierror == 785)
    printf("Error %d: Found duplicate JAN keyword!\n", ierror);
  else if (ierror == 786)
    printf("Error %d: Found both JAN and FIT1 keywords in one reaction!\n",
           ierror);
  else if (ierror == 787)
    printf("Error %d: Error when extracting JAN parameter values!\n", ierror);
  else if (ierror == 788)
    printf("Error %d: Number of JAN parameters is not equal to 9!\n", ierror);

  if (ierror == 790)
    printf("Error %d: Found duplicate FIT1 keyword!\n", ierror);
  else if (ierror == 791)
    printf("Error %d: Found both FIT1 and JAN keywords in one reaction!\n",
           ierror);
  else if (ierror == 792)
    printf("Error %d: Error when extracting FIT1 parameter values!\n", ierror);
  else if (ierror == 793)
    printf("Error %d: Number of FIT1 parameters is not equal to 4!\n", ierror);

  if (ierror == 800)
    printf("Error %d: Encountered end of file while reading continuation "
           "reaction line!\n",
           ierror);
  else if (ierror == 810)
    printf(
      "Error %d: Reaction line is too long (modify lenstr01 and recompile )!\n",
      ierror);

  if (ierror == 920)
    printf("Error %d: Empty field for FORD/RORD keyword!\n", ierror);
  else if (ierror == 921)
    printf("Error %d: Found only one word inside FORD/RORD field!\n", ierror);
  else if (ierror == 922)
    printf("Error %d: Could not find FORD/RORD species in the list of declared "
           "species!\n",
           ierror);
  else if (ierror == 923)
    printf("Error %d: When extrating FORD/RORD order value!\n", ierror);
  else if (ierror == 924)
    printf("Error %d: More then one order value in FORD/RORD field!\n", ierror);
  else if (ierror == 925)
    printf("Error %d: Too many FORD entries field!\n", ierror);
  else if (ierror == 926)
    printf("Error %d: Too many RORD entries field!\n", ierror);

  return;
}
/*
            _                 __
        ___| |__   __ _ _ __ / _|_   _ _ __   ___
       / __| '_ \ / _` | '__| |_| | | | '_ \ / __|
      | (__| | | | (_| | |  |  _| |_| | | | | (__
       \___|_| |_|\__,_|_|  |_|  \__,_|_| |_|\___|

*/
/* ------------------------------------------------------------------------- */
/**
 * \brief Checks if a string of characters contains leading spaces
 * Then shifts the string left over the leading spaces, and marks the
 * remaining space at the right with null characters
 */
int
TCKMI_elimleads(char* linein)
{
  int exclfind;
  int len1 = strlen(linein);

  if (len1 == 0)
    return (0);

  exclfind = strspn(linein, " ");

  if ((exclfind != 0) && (exclfind < len1)) {
    memmove(linein, &linein[exclfind], len1 - exclfind);
    memset(&linein[len1 - exclfind], 0, exclfind);
  }

  return (exclfind);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Checks if a string of characters contains trailing spaces
 *   Marks all those positions with null characters
 */
int
TCKMI_elimends(char* linein)
{
  int i;
  int len1 = strlen(linein);

  if (len1 == 0)
    return (0);

  i = len1 - 1;
  while (isspace(linein[i]) > 0) {
    linein[i] = 0;
    if (i > 0)
      i--;
  }

  return (i);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Eliminates space characters from a string
 */
int
TCKMI_elimspaces(char* linein)
{
  int exclfind;
  int len1 = strlen(linein);

  if (len1 == 0)
    return (0);

  exclfind = strcspn(linein, " ");

  while ((len1 > 0) && (exclfind < len1)) {

    memmove(&linein[exclfind], &linein[exclfind + 1], len1 - exclfind - 1);
    memset(&linein[len1 - 1], 0, 1);

    len1 = strlen(linein);
    exclfind = strcspn(linein, " ");
  }

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Eliminate comments: advances through a line of characters, determines
 * the first occurence of "!" (if any), then nulls out all the positions
 *   downstream (including the "!")
 *
 */
int
TCKMI_elimcomm(char* linein)
{
  char* comment = 0;
  int len1 = strlen(linein);

  if (len1 == 0)
    return 0;

  comment = strstr(linein, "!");

  if (comment != 0)
    memset(comment, 0, strlen(comment));

  if (strlen(linein) > 0)
    TCKMI_elimends(linein);

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Replaces all horizontal tab, vertical tab, line feed, and carriage
 * return characters in a line with spaces
 *       - ASCII code for a hortizontal TAB is    9
 *       - ASCII code for a vertical TAB is       11
 *       - ASCII code for a SPACE is              32
 *       - ASCII code for line feed (new line) is 10
 *       - ASCII code for carriage return is      15
 */
int
TCKMI_tab2space(char* linein)
{
  int i;
  int len1 = strlen(linein);

  if (len1 == 0)
    return 0;

  for (i = 0; i < len1; i++)
    if ((linein[i] == 9) || (linein[i] == 11))
      linein[i] = 32;

  /*
  Test if the last characted is a carriage return or line feed
  If yes, replace it with a null character
  */
  i = len1 - 1;
  if ((linein[i] == 10) || (linein[i] == 15))
    linein[i] = 0;

  return 0;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Extracts the left most word from a character string
 *     - this function assumes words are separated by spaces or "/"
 *     - the initial character string is shifted to the left starting with the
 *       sepatation character
 *     - the corrensponding positions left at the right are filled with null
 * characters
 */
int
TCKMI_extractWordLeft(char* linein, char* oneword)
{

  int spacefind = 0;
  int len1 = strlen(linein);

  if (len1 == 0)
    return 0;

  /* Eliminate leading spaces if any */
  TCKMI_elimleads(linein);
  len1 = strlen(linein);

  /* Return if remaining string is null */
  if (len1 == 0)
    return 0;

  /* Identify the location of the first "space" or "/" character */
  while ((isspace(linein[spacefind]) == 0) && (spacefind < len1 - 1) &&
         (strncmp(&linein[spacefind], "/", 1) != 0))
    spacefind++;

  if (spacefind != len1 - 1) {
    /* Found separate word -> copy word to separate string */
    strncpy(oneword, linein, spacefind);
    oneword[spacefind] = 0;
    /* Then shift things over the leading word + null the leftover positions */
    memmove(linein, &linein[spacefind], len1 - spacefind);
    memset(&linein[len1 - spacefind], 0, spacefind);
  } else {
    /* String contained one word only */
    if (isspace(linein[spacefind]) > 0) {
      strncpy(oneword, linein, spacefind);
      oneword[spacefind] = 0;
    } else {
      strncpy(oneword, linein, spacefind + 1);
      oneword[spacefind + 1] = 0;
    }
    memset(linein, 0, len1);
  }

  TCKMI_elimleads(linein);

  return 0;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Extracts two strings from a character strings
 *     - the first string starts from the first position until a space or a "/"
 *     - the second string is starts after the first slash and ends at the
 * second "/"
 *      - the initial character string is shifted to the left starting with the
 *       first character after the second "/"
 *     - the corrensponding positions left at the right are filled with null
 * characters
 */
int
TCKMI_extractWordLeftauxline(char* linein,
                             char* oneword,
                             char* twoword,
                             int* inum,
                             int* ierror)
{

  int len1, ipos;
  int spacefind = 0;

  /* Eliminate leading spaces if any, then if string is null return */
  TCKMI_elimleads(linein);
  len1 = strlen(linein);
  if (len1 == 0) {
    *inum = 0;
    return 0;
  }

  /* Identify the location of the first "space" or "/" character */
  while ((isspace(linein[spacefind]) == 0) && (spacefind < len1 - 1) &&
         (strncmp(&linein[spacefind], "/", 1) != 0))
    spacefind++;

#ifdef DEBUGMSG
  printf("TCKMI_extractWordLeftauxline: spacefind=%d \n", spacefind);
#endif
  if (spacefind != len1 - 1) {

    /* Found space character -> copy word to separate string */
    strncpy(oneword, linein, spacefind);
    oneword[spacefind] = 0;

    /* Then shift things over the leading word + null the leftover positions */
    memmove(linein, &linein[spacefind], len1 - spacefind);
    memset(&linein[len1 - spacefind], 0, spacefind);

  } else {

    /* String contained one word only */
    if (isspace(linein[spacefind]) > 0) {
      strncpy(oneword, linein, spacefind);
      oneword[spacefind] = 0;
    } else {
      strncpy(oneword, linein, spacefind + 1);
      oneword[spacefind + 1] = 0;
    }
    memset(linein, 0, len1);

    *inum = 1;

    return 0;
  }

  TCKMI_elimleads(linein);
  len1 = strlen(linein);

#ifdef DEBUGMSG
  printf("TCKMI_extractWordLeftauxline: String: |%s|, Length: %d \n",
         linein,
         strlen(linein));
#endif

  if (len1 == 0) {
    *inum = 0;
    return 0;
  }

  /*----------------------Get second word--------------------------- */
  if (strncmp(linein, "/", 1) != 0) {
    *ierror = 701;
    return 1;
  }
  ipos = strcspn(&linein[1], "/");
  ipos += 1;
  if (ipos == len1) {
    *ierror = 702;
    return 1;
  }

  /* Check if number between slashes contains at least one position */
  if (ipos == 1) {
    *ierror = 703;
    return 1;
  }

#ifdef DEBUGMSG
  printf("TCKMI_extractWordLeftauxline: String: |%s|, Length: %d \n",
         &linein[1],
         strlen(&linein[1]));
#endif

  strncpy(twoword, &linein[1], ipos - 1);
  memmove(linein, &linein[ipos + 1], len1 - ipos - 1);
  memset(&linein[len1 - ipos - 1], 0, ipos + 1);
  TCKMI_elimleads(linein);
  *inum = 2;

#ifdef DEBUGMSG
  printf("TCKMI_extractWordLeftauxline: exit \n");
#endif

  return 0;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief The only difference between this method and "TCKMI_extractWordLeft"
 *   is the absence of "/" as delimiter
 */
int
TCKMI_extractWordLeftNoslash(char* linein, char* oneword)
{

  int spacefind = 0;
  int len1 = strlen(linein);

  if (len1 == 0)
    return 0;

  /* Eliminate leading spaces if any */
  TCKMI_elimleads(linein);
  len1 = strlen(linein);

  /* Return if remaining string is null */
  if (len1 == 0)
    return 0;

  /* Identify the location of the first "space" character */
  while ((isspace(linein[spacefind]) == 0) && (spacefind < len1 - 1))
    spacefind++;

  if (spacefind != len1 - 1) {
    /* Found space character -> copy word to separate string */
    strncpy(oneword, linein, spacefind);
    oneword[spacefind] = 0;
    /* Then shift things over the leading word + null the leftover positions */
    memmove(linein, &linein[spacefind], len1 - spacefind);
    memset(&linein[len1 - spacefind], 0, spacefind);
  } else {
    /* String contained one word only */
    if (isspace(linein[spacefind]) > 0) {
      strncpy(oneword, linein, spacefind);
      oneword[spacefind] = 0;
    } else {
      strncpy(oneword, linein, spacefind + 1);
      oneword[spacefind + 1] = 0;
    }
    memset(linein, 0, len1);
  }

  TCKMI_elimleads(linein);

  return 0;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Extracts last word from a character strings:
 * (1) assumes the word is separated from the rest of the string by at least a
 * space; (2) the corrensponding positions in the initial string are filled with
 * null characters
 */
int
TCKMI_extractWordRight(char* linein, char* oneword)
{

  int spacefind;
  int len1 = strlen(linein);
  if (len1 == 0)
    return 0;

  /* Eliminate leading and ending spaces if any */
  TCKMI_elimleads(linein);
  TCKMI_elimends(linein);

  /* Return if remaining string is null */
  len1 = strlen(linein);
  if (len1 == 0)
    return 0;

  /* Identify the location of the last space character */
  spacefind = len1 - 1;
  while ((isspace(linein[spacefind]) == 0) && (spacefind > 0))
    spacefind--;

  /* Copy last word */
  strncpy(oneword, &linein[spacefind], len1 - spacefind);
  oneword[len1 - spacefind] = 0;
  TCKMI_elimleads(oneword);

  /* Zero last word in the original string */
  memset(&linein[spacefind], 0, len1 - spacefind);

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 *   \brief Extracts a double number from a character string:
 *   (1) the number is assumed to be the left most word in the string;
 *   (2) words are separated by spaces
 */
int
TCKMI_extractdouble(char* wordval, double* dvalues, int* inum, int* ierror)
{
  int len1;
  char numstr[20];

  *inum = 0;

  /* Exit if error is not zero */
  if (*ierror > 0)
    return (1);

  len1 = strlen(wordval);
  while (len1 > 0) {
    memset(numstr, 0, 20);
    TCKMI_extractWordLeftNoslash(wordval, numstr);
    TCKMI_checkstrnum(numstr, &len1, ierror);

    if (*ierror > 0)
      return 1;

    if (len1 == 0)
      return 0;

    dvalues[*inum] = atof(numstr);
    (*inum) += 1;
    len1 = strlen(wordval);
  }

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Converts all letters in a character string to uppercase
 */
void
TCKMI_wordtoupper(char* linein, char* oneword, int Npos)
{
  int i;
  for (i = 0; i < Npos; i++)
    oneword[i] = toupper(linein[i]);

  oneword[Npos] = 0;

  return;
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Performs various operations on a character strings (see explanations
 * for individual methods)
 */
void
TCKMI_cleancharstring(char* linein, int* len1)
{
  (*len1) = strlen(linein);

  if ((*len1) > 0) {

    TCKMI_tab2space(linein);
    TCKMI_elimcomm(linein);
    TCKMI_elimleads(linein);
    TCKMI_elimends(linein);

    (*len1) = strlen(linein);
  }

  return;
}

int
TCKMI_charfixespc(char* singleword, int* len1)
{

  int i;
  char estr[] = "Ee";
  char spc[] = " ";

  if (*len1 < 2)
    return (0);

  for (i = 1; i < (*len1); i++)
    if (strncmp(&singleword[i], spc, 1) == 0) {
      /* found space, test for e of E */
      if ((strncmp(&singleword[i - 1], &estr[0], 1) == 0) ||
          (strncmp(&singleword[i - 1], &estr[1], 1) == 0))
        strncpy(&singleword[i], "+", 1);
    }

  return (0);
}

/* -------------------------------------------------------------------------  */
/**
 *  \brief Checks if all components of a charcater string are valid number
 * characters (as described by the %f or %e formats)
 */
int
TCKMI_checkstrnum(char* singleword, int* len1, int* ierror)
{
  int i, j, isgood;
  int nchar = 15;
  char numch[] = "0123456789.+-Ee";
  char* schk;
  int iplus, iminus;
  int jplus = 11, jminus = 12;

  /* Return if error is not 0 */
  if (*ierror > 0)
    return 1;

  /* Clean character string */
  TCKMI_cleancharstring(singleword, len1);

  /* Check if there are spaces in the number string */
  schk = strstr(singleword, " "); // to remove warning
  if (schk != NULL) {
    printf("Error : Found space inside a number: %s\n", singleword);
    fflush(stdout);
    *ierror = 350;
    return (1);
  }

  if ((*len1) == 0)
    return 0;

  iplus = -1;
  iminus = -1;
  isgood = 0;
  for (i = 0; i < (*len1); i++) {
    isgood = 1;
    /* Check against 0123456789.+-Ee */
    for (j = 0; j < nchar; j++)
      if (strncmp(&singleword[i], &numch[j], 1) == 0)
        isgood = 0;

    if (strncmp(&singleword[i], &numch[jplus], 1) == 0)
      iplus = i;
    if (strncmp(&singleword[i], &numch[jminus], 1) == 0)
      iminus = i;

    if (strncmp(&singleword[i], "D", 1) == 0) {
      isgood = 0;
      strncpy(&singleword[i], "E", 1);
    }
    if (strncmp(&singleword[i], "d", 1) == 0) {
      isgood = 0;
      strncpy(&singleword[i], "e", 1);
    }
    if (isgood == 1)
      break;
  }

  if (isgood == 1) {
    (*ierror) = 350;
    return 1;
  }

  /* Check if an E or e precede +/- found inside the number */
  if (iplus > 0) {
    if ((strncmp(&singleword[iplus - 1], "E", 1) != 0) &&
        (strncmp(&singleword[iplus - 1], "e", 1) != 0)) {
      printf("Error : Found + inside a number without an exponent preceeding "
             "it: %s\n",
             singleword);
      fflush(stdout);
      (*ierror) = 350;
      return 1;
    }
  }
  if (iminus > 0) {
    if ((strncmp(&singleword[iminus - 1], "E", 1) != 0) &&
        (strncmp(&singleword[iminus - 1], "e", 1) != 0)) {
      printf("Error : Found - inside a number without an exponent preceeding "
             "it: %s\n",
             singleword);
      fflush(stdout);
      (*ierror) = 350;
      return 1;
    }
  }

  return (0);
}

/* -------------------------------------------------------------------------  */
/**
 *   \brief Identifies the first position in the character string that does
 *   not correspond to a positive %f format.
 *   (in other words, finds the first position that is not a digit or
 *   a decimal point)
 */
int
TCKMI_findnonnum(char* specname, int* ipos)
{
  int len1 = strlen(specname);
  int icheck = 0;

  if (len1 == 0)
    return 0;

  (*ipos) = 0;

  while (((*ipos) < len1) && (icheck == 0))
    if ((strncmp(&specname[*ipos], ".", 1) == 0) ||
        (isdigit(specname[*ipos]) > 0))
      (*ipos) += 1;
    else
      icheck = 1;

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 *  \brief kinetic model summary
 */
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
              int* nNASA9coef)
{

  /* adding +1 to all indices to account for differences between C and Fortran
   */
  int i;

  /*---------------------Species counters---------------------------- */
  (*nIonEspec) = 0;
  (*electrIndx) = 0;
  (*nNASA9coef) = 0;
  for (i = 0; i < (*Nspec); i++) {
    /* Counting number of ion species and store position of the electron species
     */
    if (listspec[i].charge != 0) {
      if (strncmp(listspec[i].name, "E", 1) == 0)
        (*electrIndx) = i + 1;
      else
        (*nIonEspec) += 1;
    }
    if (listspec[i].hasthermo == 2)
      (*nNASA9coef) += 1;
  }
  (*nIonSpec) = (*nIonEspec);
  if ((*electrIndx) != 0)
    (*nIonSpec) += 1;

  /* ---------------------Reaction counters---------------------------- */
  (*maxSpecInReac) = 0;
  (*maxTbInReac) = 0;
  (*maxOrdPar) = 0;
  (*nFallPar) = 1;
  (*nLtReac) = 0;
  (*nRltReac) = 0;
  (*nFallReac) = 0;
  (*nPlogReac) = 0;
  (*nThbReac) = 0;
  (*nRevReac) = 0;
  (*nHvReac) = 0;
  (*nTdepReac) = 0;
  (*nJanReac) = 0;
  (*nFit1Reac) = 0;
  (*nExciReac) = 0;
  (*nMomeReac) = 0;
  (*nXsmiReac) = 0;
  (*nRealNuReac) = 0;
  (*nOrdReac) = 0;
  /* Start counting ... */
  for (i = 0; i < (*Nreac); i++) {
    /* ...maximum number of species in a reaction */
    (*maxSpecInReac) = MAX(*maxSpecInReac, 2 * (listreac[i].inreac));
    (*maxSpecInReac) = MAX(*maxSpecInReac, 2 * (listreac[i].inprod));

    /* ...maximum number of pressure dependent reactions and coeffs */
    if (listreac[i].isfall > 0) {
      (*nFallReac) += 1;
      if (listreac[i].islowset > 0)
        (*nFallPar) = MAX((*nFallPar), 3);
      if (listreac[i].ishighset > 0)
        (*nFallPar) = MAX((*nFallPar), 3);
      if (listreac[i].istroeset > 0)
        (*nFallPar) = MAX((*nFallPar), 7);
      if (listreac[i].issriset > 0)
        (*nFallPar) = MAX((*nFallPar), 8);
    }

    if (listreac[i].isplogset > 0)
      (*nPlogReac) += 1;

    /* ...third-body data */
    if (listreac[i].isthrdb > 0) {
      (*nThbReac) += 1;
      (*maxTbInReac) = MAX(*maxTbInReac, listreac[i].nthrdb);
    }
    if (listreac[i].isltset > 0)
      (*nLtReac) += 1; /* ...Landau-Teller reactions         */
    if (listreac[i].isrltset > 0)
      (*nRltReac) += 1; /* ...reverse Landau-Teller reactions */
    if (listreac[i].isrevset > 0)
      (*nRevReac) += 1; /* ...reverse data reactions          */
    if (listreac[i].iswl > 0)
      (*nHvReac) += 1; /* ...radiation wevelength            */
    if (listreac[i].istdepset > 0)
      (*nTdepReac) += 1; /* ...species temperature dependency  */
    if (listreac[i].isjanset > 0)
      (*nJanReac) += 1; /* ...JAN fit reactions               */
    if (listreac[i].isfit1set > 0)
      (*nFit1Reac) += 1; /* ...FIT1 reactions                  */
    if (listreac[i].isexciset > 0)
      (*nExciReac) += 1; /* ...EXCI reactions                  */
    if (listreac[i].ismome > 0)
      (*nMomeReac) += 1; /* ...MOME reactions                  */
    if (listreac[i].isxsmi > 0)
      (*nXsmiReac) += 1; /* ...XSMI reactions                  */
    if (listreac[i].isreal > 0)
      (*nRealNuReac) += 1; /* ...real stoichiometry reactions    */

    /* ...arbitrary order reactions */
    if (listreac[i].isford > 0) {
      (*nOrdReac) += 1;
      (*maxOrdPar) = MAX((*maxOrdPar), listreac[i].isford + listreac[i].isrord);
    }

  } /* done loop over all reactions */

  /* Other definitions */
  (*maxTpRange) =
    3; /* number of temperatures defining ranges for thermo props */
  (*nNASAinter) = (*maxTpRange) - 1;
  (*nCpCoef) = 5; /* no. of coeffs for the specific heat polynomial */
  (*nNASAfit) = (*nCpCoef) + 2; /* no. of coeff for thermo props */
  (*nArhPar) = 3;               /* no. of Arrhenius parameters */
  (*nLtPar) = 2;                /* no. of Landau-Teller parameters */
  (*nJanPar) = 9;               /* no. of Jannev-Langer parameters */
  (*nFit1Par) = 4;              /* no. of FIT1 parameters */

  return (0);
}

/* -------------------------------------------------------------------------  */
/**
 * \brief Outputs reaction data to ascii file
 */
int
TCKMI_outmath(element* listelem,
              int* Nelem,
              species* listspec,
              int* Nspec,
              reaction* listreac,
              int* Nreac,
              char* aunits,
              char* eunits)
{
  int i, j, k, ipos;
  char fname[100];
  FILE* fout;

  int nIonEspec, electrIndx, nIonSpec, maxSpecInReac, maxTbInReac, maxOrdPar,
    nFallPar, maxTpRange;
  int nLtReac, nRltReac, nFallReac, nPlogReac, nThbReac, nRevReac, nHvReac,
    nTdepReac, nJanReac, nFit1Reac, nExciReac;
  int nMomeReac, nXsmiReac, nRealNuReac, nOrdReac, nNASAinter, nCpCoef,
    nNASAfit, nArhPar;
  int nLtPar, nJanPar, nFit1Par, nNASA9coef;

  /* compute kinetic model summary */
  TCKMI_kmodsum(listelem,
                Nelem,
                listspec,
                Nspec,
                listreac,
                Nreac,
                &nIonEspec,
                &electrIndx,
                &nIonSpec,
                &maxSpecInReac,
                &maxTbInReac,
                &maxOrdPar,
                &nFallPar,
                &maxTpRange,
                &nLtReac,
                &nRltReac,
                &nFallReac,
                &nPlogReac,
                &nThbReac,
                &nRevReac,
                &nHvReac,
                &nTdepReac,
                &nJanReac,
                &nFit1Reac,
                &nExciReac,
                &nMomeReac,
                &nXsmiReac,
                &nRealNuReac,
                &nOrdReac,
                &nNASAinter,
                &nCpCoef,
                &nNASAfit,
                &nArhPar,
                &nLtPar,
                &nJanPar,
                &nFit1Par,
                &nNASA9coef);

  /* Output elements */
  sprintf(fname, "math_elem.dat");
  if ((fout = fopen(fname, "w")))
    for (i = 0; i < (*Nelem); i++)
      fprintf(fout, "%3s %24.15e\n", listelem[i].name, listelem[i].mass);
  else {
    printf("TCKMI_outmath() : ERROR : could not open %s -> Abort\n", fname);
    fflush(stdout);
    exit(1);
  }
  if (fout != NULL)
    fclose(fout);
  fout = NULL;

  /* Output species */
  sprintf(fname, "math_spec.dat");
  if ((fout = fopen(fname, "w"))) {
    for (i = 0; i < (*Nspec); i++) {
      fprintf(fout, "%20s %24.15e", listspec[i].name, listspec[i].mass);

      for (j = 0; j < (*Nelem); j++) {
        ipos = (*Nelem);
        for (k = 0; k < listspec[i].numofelem; k++)
          if (listspec[i].elemindx[k] == j) {
            ipos = j;
            fprintf(fout, " %3d", listspec[i].elemcontent[k]);
          }
        if (ipos == *Nelem)
          fprintf(fout, " %3d", 0);
      }
      fprintf(fout, "\n");
    }
  } else {
    printf("TCKMI_outmath() : ERROR : could not open %s -> Abort\n", fname);
    fflush(stdout);
    exit(1);
  }
  if (fout != NULL)
    fclose(fout);
  fout = NULL;

  /* NASA polynomials */
  sprintf(fname, "math_nasapol7.dat");
  if ((fout = fopen(fname, "w"))) {
    /* Temperature for thermo fits ranges */
    for (i = 0; i < (*Nspec); i++) {
      fprintf(fout, "%20s", listspec[i].name);
      fprintf(fout, " %20.12e", listspec[i].nasapoltemp[0]); /* lower  bound */
      fprintf(fout, " %20.12e", listspec[i].nasapoltemp[2]); /* middle bound */
      fprintf(
        fout, " %20.12e\n", listspec[i].nasapoltemp[1]); /* upper  bound */
      for (j = 0; j < nNASAinter; j++) {
        for (k = 0; k < nNASAfit; k++)
          fprintf(fout, " %20.12e", listspec[i].nasapolcoefs[j * nNASAfit + k]);
      }
      fprintf(fout, "\n");
    }
  } else {
    printf("TCKMI_outmath() : ERROR : could not open %s -> Abort\n", fname);
    fflush(stdout);
    exit(1);
  }
  if (fout != NULL)
    fclose(fout);
  fout = NULL;

  /* Basic reaction info */
  sprintf(fname, "math_reac.dat");
  if (((*Nreac) > 0) && (fout = fopen(fname, "w"))) {
    fprintf(fout, "%10d %10d\n", (*Nreac), maxSpecInReac);
    /* No. of reactants+products (negative values for irreversible reactions) */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isrev < 0)
        fprintf(fout, "%10d", -(listreac[i].inreac + listreac[i].inprod));
      else
        fprintf(fout, "%10d", listreac[i].inreac + listreac[i].inprod);
      fprintf(fout, " %10d", listreac[i].inreac);

      /* Stoichiometric coefficients */
      for (j = 0; j < maxSpecInReac; j++) {
        if (j < maxSpecInReac / 2) {
          fprintf(fout, " %10f", listreac[i].nuki[j]);
          fprintf(fout, " %10d", listreac[i].spec[j] + 1);
        } else {
          int j1;
          j1 = j - maxSpecInReac / 2;
          fprintf(fout, " %10f", listreac[i].nuki[NSPECREACMAX + j1]);
          fprintf(fout, " %10d", listreac[i].spec[NSPECREACMAX + j1] + 1);
        }
      }

      fprintf(fout, " %24.16e", listreac[i].arhenfor[0]);
      fprintf(fout, "%24.16e", listreac[i].arhenfor[1]);
      fprintf(fout, " %24.16e\n", listreac[i].arhenfor[2]);
    }

  } /* Done if Nreac > 0 */
  if (fout != NULL)
    fclose(fout);
  fout = NULL;

  /* Reactions with reversible Arrhenius parameters given */
  sprintf(fname, "math_reacrev.dat");
  if ((nRevReac > 0) && (fout = fopen(fname, "w"))) {

    /* No. of such reactions */
    fprintf(fout, "%12d\n", nRevReac);

    /* Their indices */
    for (i = 0; i < (*Nreac); i++)
      if (listreac[i].isrevset > 0)
        fprintf(fout,
                "%10d  %24.16e %24.16e %24.16e\n",
                i + 1,
                listreac[i].arhenrev[0],
                listreac[i].arhenrev[1],
                listreac[i].arhenrev[2]);

  } /* Done if nRevReac > 0 */
  if (fout != NULL)
    fclose(fout);
  fout = NULL;

  /* Pressure-dependent reactions */
  sprintf(fname, "math_falloff.dat");
  if ((nFallReac > 0) && (fout = fopen(fname, "w"))) {
    fprintf(fout, "%10d %10d\n", nFallReac, nFallPar);

    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isfall > 0) {

        /* Reaction index */
        fprintf(fout, "%10d ", i + 1);

        /* Type for fall-off reaction */
        if ((listreac[i].istroeset == 0) && (listreac[i].issriset == 0))
          /* Lindemann form */
          fprintf(fout, "LIND ");
        else if (listreac[i].issriset > 0)
          /* SRI form */
          fprintf(fout, "SRI  ");
        else if (listreac[i].istroeset > 0)
          /* TROE form */
          fprintf(fout, "TROE%d", listreac[i].istroeset);
        else {
          printf("Unknown pressure dependent type for reaction %d\n", i);
          exit(-1);
        }

        /* Type of low/high */
        if (listreac[i].islowset == 1)
          /* LOW */
          fprintf(fout, " LOW ");
        else
          /* HIGH */
          fprintf(fout, " HIGH");

        /* fall-off species */
        fprintf(fout, " %10d\n", listreac[i].specfall + 1);
      }

    } /* Done printing fall-off indices */

    /* Print fall-off parameters */
    for (i = 0; i < (*Nreac); i++) {
      int maxpars = 0;
      if (listreac[i].isfall > 0) {
        if ((listreac[i].islowset > 0) || (listreac[i].ishighset > 0)) {
          /* LOW or HIGH */
          maxpars = 3;
          for (j = 0; j < 3; j++)
            fprintf(fout, " %24.16e", listreac[i].fallpar[j]);

          if (listreac[i].istroeset > 0) {
            /* TROE */
            maxpars = 3 + listreac[i].istroeset;
            for (j = 0; j < listreac[i].istroeset; j++)
              fprintf(fout, " %24.16e", listreac[i].fallpar[3 + j]);
          } else if (listreac[i].issriset > 0) {
            /* SRI */
            maxpars = 8;
            for (j = 0; j < 5; j++)
              fprintf(fout, " %24.16e", listreac[i].fallpar[3 + j]);
          }

        } /* Done LOW/HIGH */
        else {
          printf("Unknown pressure dependent type for reaction %d\n", i);
          exit(-1);
        }

        for (j = maxpars; j < nFallPar; j++)
          fprintf(fout, " %24.16e", 0.0);
        fprintf(fout, "\n");
      }

    } /* Done printing fall-off data */

  } /* Done if fall-off reactions */
  if (fout != NULL)
    fclose(fout);
  fout = NULL;

  /* Third-body reactions */
  sprintf(fname, "math_trdbody.dat");
  if ((nThbReac > 0) && (fout = fopen(fname, "w"))) {
    fprintf(fout, "%10d %10d\n", nThbReac, maxTbInReac);

    /* Reaction index and # of third body spec */
    for (i = 0; i < (*Nreac); i++)
      if (listreac[i].isthrdb > 0)
        fprintf(fout, "%12d %12d\n", i + 1, listreac[i].nthrdb);

    /* Species indices */
    for (i = 0; i < (*Nreac); i++)
      if (listreac[i].isthrdb > 0) {
        for (j = 0; j < maxTbInReac; j++) {
          if (j < listreac[i].nthrdb)
            fprintf(fout,
                    "%10d %24.16e ",
                    listreac[i].ithrdb[j] + 1,
                    listreac[i].rthrdb[j]);
          else
            fprintf(fout, "%10d %24.16e ", 0, 0.0);
        }
        fprintf(fout, "\n");
      }
  }
  if (fout != NULL)
    fclose(fout);
  fout = NULL;

  /* Reactions with real stoichiometric coefficients */
  sprintf(fname, "math_realnu.dat");
  if ((nRealNuReac > 0) && (fout = fopen(fname, "w"))) {
    fprintf(fout, "%10d\n", nRealNuReac);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isreal > 0) {
        double nusumk = 0;

        fprintf(fout, "%10d ", i + 1);

        /* Stoichiometric coefficients */
        for (j = 0; j < maxSpecInReac; j++) {
          if (j < listreac[i].inreac) {
            fprintf(fout, "%24.16e ", listreac[i].rnuki[j]);
            nusumk += listreac[i].rnuki[j];
          } else if (j < listreac[i].inreac + listreac[i].inprod) {
            int j1;
            j1 = j - listreac[i].inreac;
            fprintf(fout, "%24.16e ", listreac[i].rnuki[NSPECREACMAX + j1]);
            nusumk += listreac[i].rnuki[NSPECREACMAX + j1];
          } else
            fprintf(fout, "%24.16e ", 0.0);
        }
        fprintf(fout, "%24.16e\n", nusumk);
      }

    } /* Done with stoichimetric coefficients */
  }
  if (fout != NULL)
    fclose(fout);
  fout = NULL;

  /* Arbitrary reaction orders */
  sprintf(fname, "math_arbnu.dat");
  if ((nOrdReac > 0) && (fout = fopen(fname, "w"))) {
    fprintf(fout, "%10d %10d\n", nOrdReac, maxOrdPar);

    /* Reaction index */
    for (i = 0; i < (*Nreac); i++) {
      if (listreac[i].isford > 0) {
        fprintf(fout, "%10d ", i + 1);

        for (j = 0; j < listreac[i].isford; j++)
          fprintf(fout,
                  "%10d %24.16e ",
                  -(listreac[i].arbspec[j] + 1),
                  listreac[i].arbnuki[j]);
        for (j = 0; j < listreac[i].isrord; j++)
          fprintf(fout,
                  "%10d %24.16e ",
                  listreac[i].arbspec[2 * NSPECREACMAX + j] + 1,
                  listreac[i].arbnuki[2 * NSPECREACMAX + j]);
        for (j = listreac[i].isford + listreac[i].isrord; j < maxOrdPar; j++)
          fprintf(fout, "%10d %24.16e ", 0, 0.0);
        fprintf(fout, "\n");
      }
    }
  }
  if (fout != NULL)
    fclose(fout);
  fout = NULL;

  /* Landau-Teller reactions */
  if (nLtReac > 0)
    printf("No Landau-Teller reactions to mathematica yet\n");

  /* reverse Landau-Teller reactions */
  if (nRltReac > 0)
    printf("No reverse Landau-Teller reactions to mathematica yet\n");

  /* radiation wavelength */
  if (nHvReac > 0)
    printf("No radiation wavelength output to mathematica yet\n");

  /* JAN fits */
  if (nJanReac > 0)
    printf("No JAN fits output to mathematica yet\n");

  /* FIT1 fits */
  if (nFit1Reac > 0)
    printf("No FIT1 fits output to mathematica yet\n");

  /* Excitation reactions */
  if (nExciReac > 0)
    printf("No excitation reactions output to mathematica yet\n");

  /* Plasma Momentum transfer collision */
  if (nMomeReac > 0)
    printf("No plasma momentum transfer collision output to mathematica yet\n");

  /* Ion Momentum transfer collision */
  if (nXsmiReac > 0)
    printf("No ion momentum transfer collision output to mathematica yet\n");

  /* Species temperature dependency */
  if (nTdepReac > 0)
    printf("No species temperature dependency output to mathematica yet\n");

  return (0);

} /* Done with "TCKMI_outmath" */
