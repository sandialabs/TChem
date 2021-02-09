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


/*! \file TC_kmodint_surface.c

    \brief Collection of functions used to parse surface kinetics

    \details
       ...

*/

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "TC_kmodint_surface.hpp"
#define VERBOSE
//#include "TC_getthc9.c"
#define MAX(A, B) (((A) > (B)) ? (A) : (B))
/**
 * \def MIN
 * Minimum of two expressions
 */
#define MIN(A, B) (((A) < (B)) ? (A) : (B))

/* ---------------------------Main function----------------------------- */
/**
 * \ingroup init
 * \brief Surface kinetic model interpretor
 */
int
TC_kmodint_surface_(char* mechfile, char* thermofile, infoGasChem* kmGas)
{
  /**
   * \param mechfile : name of file containing kinetic model in chemkin format
   * \param lmech    : length of mechfile character string
   * \param thermofile : name of file containing coefficients for NASA
   * polynomials \param lthrm    : length of thermofile character string
   */

  /* Counters */
  int i, icline;

  /// int Natoms = 0;/// not used

  /* ----------Number of species, reactions----------------- */
  int NspecSurf, NspecmaxSurf;
  int NreacSurf, NreacmaxSurf;

  /* -----------Species, and reaction data structures-------- */
  speciesSurf* listspecSurf;
  reactionSurf* listreacSurf;

  /* --------Global temperature range (thermo data)----- */
  double TglobalSurf[3];

  /* ---Pre-exonential factor and activation energy units--- */
  char aunits[lenstr03], eunits[lenstr03];

  /* File I/O */
  FILE *mechin, *thermoin, *filelist, *fileascii,  *filereacEqn;
  char listfile[lenfile], asciifile[lenfile], reactfile[lenfile];

  /* Character strings */
  char linein[lenstr01], linein2[lenstr01], singleword[lenstr01], kwd[lenstr03];

  /* Integer flags */
  int ierror, iread, ithermo, iremove;

  double siteden;

#ifdef VERBOSE
  printf("Reading surface kinetic model from : %s\n", mechfile);
  printf("              and thermo data from : %s\n", thermofile);
#endif

  /* ----------Number of elements, species, reactions----------------- */
  NspecSurf = 0;
  NspecmaxSurf = nspecalloc;
  NreacSurf = 0;
  NreacmaxSurf = nreacalloc;

  /* -----------Element, species, and reaction data structures-------- */
  listspecSurf = (speciesSurf*)malloc(NspecmaxSurf * sizeof(listspecSurf[0]));
  listreacSurf = (reactionSurf*)malloc(NreacmaxSurf * sizeof(listreacSurf[0]));

  /* --------Global temperature range (thermo data)----- */
  for (i = 0; i < 3; i++)
    TglobalSurf[i] = -100.0;

  /* Integer flags */
  ierror = 0;
  iread = 0;
  ithermo = 0;
  iremove = 0;

  /* --------------------Input/Output file---------------------------- */
  strcpy(listfile, "kmodSurf.list"); /* Unformatted ASCII output file */
  strcpy(asciifile, "kmodSurf.out"); /* Formatted   ASCII output file */
  strcpy(reactfile, "kmodSurf.reactions"); /* Formatted   ASCII output file */

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
    printf("TC_kmodin_surf() : Could not open %s -> Abort !\n", mechfile);
    fflush(stdout);
    exit(1);
  }
  if (!thermoin) {
    printf("TC_kmodin_surf() : Could not open %s -> Abort !\n", thermofile);
    fflush(stdout);
    exit(1);
  }
  if (!filelist) {
    printf("TC_kmodin_surf() : Could not open %s -> Abort !\n", listfile);
    fflush(stdout);
    exit(1);
  }
  if (!fileascii) {
    printf("TC_kmodin_surf() : Could not open %s -> Abort !\n", asciifile);
    fflush(stdout);
    exit(1);
  }

  if (!filereacEqn) {
    printf("TC_kmodin() : Could not open %s -> Abort !\n", reactfile);
    fflush(stdout);
    exit(1);
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

// #define DEBUGMSG
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
    if (strncmp(kwd, "SITE", 4) == 0) {
      /* Found species keyword, set remove flag */
      iread = 1;
      iremove = 1;
    } else if (strncmp(kwd, "THER", 4) == 0) {
      /* Found thermo keyword, set remove flag */
      iread = 2;
      iremove = 1;
    } else if (strncmp(kwd, "REAC", 4) == 0) {
      /* Found reactions keyword */
      iread = 3;
      iremove = 1;
      /* Get thermodynamic species for all elements */
      int TC_7TCoefs = 1;
      if (TC_7TCoefs == 1)
        TCKMI_getthermosurf(linein,
                            singleword,
                            mechin,
                            thermoin,
                            kmGas->infoE,
                            &(kmGas->Nelem),
                            listspecSurf,
                            &NspecSurf,
                            TglobalSurf,
                            &ithermo,
                            &iread,
                            &ierror);
    } else if (strncmp(kwd, "END", 3) == 0) {
      if (iread == 3)
        break;
      iread = 0;
    } else if ((iread == 0) && (strlen(linein) > 0)) {
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

    if ((iremove == 1) && (iread == 1))
      TCKMI_sitedens(linein, singleword, &siteden);

    if ((iremove == 1) && (iread == 3))
      TCKMI_checkunits(linein, singleword, aunits, eunits);

    /* Route things as approriate */
    if (iread == 1) {
      /* In the site mode */
      len1 = strlen(linein);
      while (len1 > 0) {
        TCKMI_getsite(linein,
                      singleword,
                      (kmGas->Nelem),
                      &listspecSurf,
                      &NspecSurf,
                      &NspecmaxSurf,
                      &iread,
                      &ierror);
        if (ierror > 0) {
          TCKMI_errormsg(ierror);
          return (ierror);
        }
        len1 = strlen(linein);
      }
    } else if (iread == 2)
      /* In the thermo mode */
      TCKMI_getthermosurf(linein,
                          singleword,
                          mechin,
                          thermoin,
                          kmGas->infoE,
                          &(kmGas->Nelem),
                          listspecSurf,
                          &NspecSurf,
                          TglobalSurf,
                          &ithermo,
                          &iread,
                          &ierror);

    else if (iread == 3) {
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

      // loop until there is nothing to interpret on the current line
      while (len1 > 0) {
        //#define DEBUGMSG
#ifdef DEBUGMSG
        printf("1-Line #%d, String: |%s|, Length: %d \n",
               icline,
               linein,
               strlen(linein));
#endif
        TCKMI_getreactionssurf(linein,
                               singleword,
                               kmGas->infoS,
                               &(kmGas->Nspec),
                               listspecSurf,
                               &NspecSurf,
                               listreacSurf,
                               &NreacSurf,
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
      if (NreacSurf > NreacmaxSurf - nreacalloc) {
        NreacmaxSurf += nreacalloc;
        listreacSurf = (reactionSurf*)realloc(
          listreacSurf, NreacmaxSurf * sizeof(listreacSurf[0]));
      }

    } /* end if iread = 3 */

    if (ierror > 0) {
      TCKMI_errormsg(ierror);
      return (ierror);
    }

    /* Reset kwd */
    len1 = strlen(kwd);
    memset(kwd, 0, len1);

  } /* Done reading the file */

  if (NreacSurf == 0) {
    if (ierror > 0) {
      TCKMI_errormsg(ierror);
      return (ierror);
    }
  }

  // /* Verify completness and correctness for each reaction */
  // TCKMI_verifyreac(listelem,&Nelem,listspec,&Nspec,listreac,&Nreac,&ierror) ;
  // if (ierror > 0) {
  //     TCKMI_errormsg(ierror) ;
  //     return ( ierror )  ;
  // }

  /* Output to ascii file (formatted) */
  TCKMI_outformsurf(kmGas->infoE,
                    &(kmGas->Nelem),
                    kmGas->infoS,
                    &(kmGas->Nspec),
                    listspecSurf,
                    &NspecSurf,
                    listreacSurf,
                    &NreacSurf,
                    aunits,
                    eunits,
                    fileascii);

//
  TCKMI_saveSurfRectionEquations(kmGas->infoS,
                           listspecSurf,
                           listreacSurf,
                           &NreacSurf,
                            filereacEqn);
  /* Rescale pre-exponential factors and activation energies (if needed) */
  TCKMI_rescalereacsurf(listreacSurf, &NreacSurf);

  /* Output to unformatted ascii file */
  if (ierror == 0)
    TCKMI_outunformsurf(&(kmGas->Nelem),
                        siteden,
                        listspecSurf,
                        &NspecSurf,
                        listreacSurf,
                        &NreacSurf,
                        aunits,
                        eunits,
                        filelist,
                        &ierror);
  if (ierror > 0) {
    TCKMI_errormsg(ierror);
    return (ierror);
  }

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
  // free(listelem     ) ;
  // free(listspec     ) ;
  // free(listreac     ) ;

  return (ierror);
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
TCKMI_getsite(char* linein,
              char* singleword,
              int Nelem,
              speciesSurf** listspecaddr,
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
  printf("In TCKMI_getsite :\n");
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
      TC_KMI_resetspecsurfdata(&(*listspecaddr)[*Nspec], Nelem);
      (*Nspec) += 1;
      printf("%s %d\n", (*listspecaddr)[*Nspec - 1].name, *Nspec);
    }

    TCKMI_elimleads(linein);
    lenstr = strlen(linein);

    if ((*Nspec) > ((*Nspecmax) - 1)) {
      /*
      *ierror = -100 ;
      return (0) ;
      */
      (*Nspecmax) += 1;
      *listspecaddr = (speciesSurf*)realloc(
        *listspecaddr, (*Nspecmax) * sizeof(*listspecaddr[0]));
    }
  }

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Reset data for a species
 */
void
TC_KMI_resetspecsurfdata(speciesSurf* currentspec, int Nelem)
{

  int i;

#ifdef DEBUGMSG
  printf("In TCKMI_resetspecdata : %p\n", &currentspec[0]);
#endif

  currentspec[0].hasmass = 0;
  currentspec[0].mass = 0.0;
  currentspec[0].charge = 0;
  currentspec[0].phase = 0;

  // free(currentspec[0].elemcontent);
  for (i = 0; i < 200; i++)
    currentspec[0].elemcontent[i] = 0;

  currentspec[0].hasthermo = 0;
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
TCKMI_setspecsurfmass(element* listelem,
                      int* Nelem,
                      speciesSurf* listspec,
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
      for (j = 0; j < (*Nelem); j++) {
        listspec[i].mass += listspec[i].elemcontent[j] * listelem[j].mass;
      } /* Done loop over the number of elements in species */
      if (*ierror == 0)
        listspec[i].hasmass = 1;
    } /* Done if statement checking for mass */
  }   /* Done loop over number of species */

  return (0);
}

/* ------------------------------------------------------------------------- */
/**
 * \brief Outputs error messages
 */

void
TCKMIS_errormsg(int ierror)
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
    printf("Error %d: Reactant/Product (Surface) species was not found in the "
           "list of species!\n",
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

/* ------------------------------------------------------------------------- */
/**
 * \brief Sets units for the pre-exponential factor and for the activation
 *   energy
 */
void
TCKMI_sitedens(char* linein, char* singleword, double* sden)
{

  int i, len1;

  /* Reset memory locations */
  memset(singleword, 0, 10);

  TCKMI_elimleads(linein);
  len1 = strlen(linein);

  if (len1 > 0) {
    TCKMI_extractWordLeft(linein, singleword);
    len1 = strlen(linein);
    if (strncmp(singleword, "SDEN", 4) == 0) {
      if (strncmp(linein, "/", 1) == 0) {
        /* Check if string contained just the "/" */
        if (len1 == 1) {
          printf("SDEN line too short");
          exit(1);
        }
        /* Check for the second slash */
        i = strcspn(&linein[1], "/");
        if (i == len1 - 1) {
          printf("SDEN line too short");
          exit(1);
        }
        /* Found stuff between slashes, transform to number */
        strncpy(singleword, &linein[1], i);
        singleword[i] = 0;
        *sden = atof(singleword);
        memmove(linein, &linein[i + 2], len1 - i - 2);
        memset(&linein[len1 - i - 2], 0, i + 2);
        printf("%e\n", (*sden));
      }
    } /* end if found "/" */
    else {
      printf("Did not find SDEN keyword!");
      exit(1);
    }
  } else {
    printf("Did not find SDEN keyword!");
    exit(1);
  }

  return;
}

/* ------------------------------------------------------------------------- */
/**
 *  \brief Returns position of a species in the list of species
 *   The index goes from 0 to (Nspec-1); if the species is not found
 *   the value of Nspec is returned
 */
void
TCKMI_checkspecsurfinlist(char* specname,
                          speciesSurf* listspec,
                          int* Nspec,
                          int* ipos)
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
/* ------------------------------------------------------------------------- */
/**
 * \brief Returns 1 if all species have thermodynamic properties set, 0
 * otherwise
 */
int
TCKMI_checkthermosurf(speciesSurf* listspec, int* Nspec)
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

/* ------------------------------------------------------------------------- */
/**
 * \brief Reads thermodynamic properties (NASA polynomials) from the
 *        mechanism input file or from a separate file
 */
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
  if ((*iread != 2) && (*iread != 3)) {
    *ierror = 300;
    return (1);
  }

  /* Check if all species have the thermodynamic properties set already */
  if (TCKMI_checkthermosurf(listspec, Nspec) == 1)
    return (0);

  /* Check if the thermodynamic properies are to be read from mechanism
  input file  */
  if ((*ithermo == 1) && (*iread == 3)) {
    /* previously encountered thermo all and not all species were
    provided thermodynamic properties in the input file */
    *ierror = 310;
    return (1);
  }

  /* Set up the input file */
  thermofile = NULL;
  if (*iread == 2)
    thermofile = mechin;
  else if (*iread == 3)
    thermofile = thermoin;

  /* Check if file pointer is null */
  if (thermofile == NULL) {
    *ierror = 320;
    return (1);
  }

  /* If reading from mechanism input see if "thermo all" */
  if (*iread == 2) {
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
    // printf("%s\n",linein1);

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
      if (*iread == 2)
        (*iread) = 0;
      return 0;
    }

    if ((strncmp(singleword, "REAC", 4) == 0) && (*iread == 3)) {
      strncpy(linein, linein1, 200);
      TCKMI_extractWordLeft(linein, singleword);
      (*iread) = 3;
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
    // printf("Species %s\n",singleword) ;
    TCKMI_checkspecsurfinlist(singleword, listspec, Nspec, &ipos);

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

          /* Good element, find numbers */
          strncpy(singleword, &linein1[itemp + 2], 3);
          singleword[3] = 0;
          TCKMI_checkstrnum(singleword, &len1, ierror);
          if (*ierror > 0)
            return 1;
          itemp = (int)atof(singleword);

          listspec[ipos].elemcontent[ipos1] = itemp;

          /* See if charge needs to be updated */
          if (strncmp(listelem[ipos1].name, "E", 1) == 0) {
            listspec[ipos].charge += listspec[ipos].elemcontent[i] * (-1);
          }

          /* Update mass */
          // printf("%s: %s
          // %d\n",listspec[ipos].name,listelem[ipos1].name,listspec[ipos].elemcontent[ipos1]);
          listspec[ipos].mass +=
            listelem[ipos1].mass * listspec[ipos].elemcontent[ipos1];
        }

        listspec[ipos].hasmass = 1;

        /* Get species phase */
        itemp = 44;
        if ((strncmp(&linein1[itemp], "l", 1) == 0) ||
            (strncmp(&linein1[itemp], "L", 1) == 0))
          listspec[ipos].phase = 1;
        else if ((strncmp(&linein1[itemp], "s", 1) == 0) ||
                 (strncmp(&linein1[itemp], "S", 1) == 0))
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
TCKMI_resetreacdatasurf(reactionSurf* currentreac, char* aunits, char* eunits)
{

  int i;

  currentreac[0].isdup = -2;
  currentreac[0].isstick = 0;
  currentreac[0].iscov = 0;
  currentreac[0].cov_count = 0;
  currentreac[0].isrev = 0;
  currentreac[0].isbal = 0;
  currentreac[0].iscomp = 0;
  currentreac[0].inreac = 0;
  currentreac[0].inprod = 0;

  currentreac[0].isrevset = 0;
  currentreac[0].ishvset = 0;

  for (i = 0; i < 2 * NSPECREACMAX; i++)
    currentreac[0].spec[i] = -1;
  for (i = 0; i < 2 * NSPECREACMAX; i++)
    currentreac[0].surf[i] = -1;
  for (i = 0; i < 2 * NSPECREACMAX; i++)
    currentreac[0].nuki[i] = 0;

  for (i = 0; i < 3; i++)
    currentreac[0].arhenfor[i] = 0.0;
  for (i = 0; i < 3; i++)
    currentreac[0].arhenrev[i] = 0.0;

  currentreac[0].hvpar = 0.0;

  strncpy(currentreac[0].aunits, aunits, 4);
  strncpy(currentreac[0].eunits, eunits, 4);

  return;
}
/* ------------------------------------------------------------------------- */
/**
 * \brief Interprets a character string containing reaction description
 *    - decides if the character string describes a reaction or the
 *      auxiliary information associated with one
 */
int
TCKMI_getreactionssurf(char* linein,
                       char* singleword,
                       speciesGas* listspec,
                       int* Nspec,
                       speciesSurf* listspecSurf,
                       int* NspecSurf,
                       reactionSurf* listreacSurf,
                       int* NreacSurf,
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
  for (i = 0; i < len1 - 4; i++)
    if (strncmp(&linein[i], "STICK", 5) == 0)
      iloc = i;
  //
  for (i = 0; i < len1 - 2; i++)
    if (strncmp(&linein[i], "COV", 3) == 0)
      iloc = i;

#ifdef DEBUGMSG
  printf("TCKMI_getreactions: %d \n", iloc);
#endif

  if ((iloc > -1) && (*NreacSurf == 0)) {
    /* Error, cannot have auxiliary data before reading at least one reaction */
    *ierror = 500;
    return 1;
  }

  if (iloc > -1) {
    /* Character string contains auxiliary info */
#ifdef DEBUGMSG
    printf("TCKMI_getreactions: entering TCKMI_getreacauxl\n");
#endif
    TCKMI_getreacauxlsurf(linein,
                          singleword,
                          listspec,
                          Nspec,
                          listspecSurf,
                          NspecSurf,
                          listreacSurf,
                          NreacSurf,
                          ierror);
#ifdef DEBUGMSG
    printf("TCKMI_getreactions: done with TCKMI_getreacauxl\n");
#endif
  } else {
    /* Character string contains reaction description */
    /* First, reset the info for the current reaction */
    TCKMI_resetreacdatasurf(&listreacSurf[*NreacSurf], aunits, eunits);
    /* Next, interpret the string, and increment the current reaction line */
    TCKMI_getreaclinesurf(linein,
                          singleword,
                          listspec,
                          Nspec,
                          listspecSurf,
                          NspecSurf,
                          listreacSurf,
                          NreacSurf,
                          ierror);
    (*NreacSurf)++;
  }

  return (0);
}

/* -------------------------------------------------------------------------  */
/**
 * \brief Interprets a character string containing reaction description
 *   (auxiliary information)
 */
int
TCKMI_getreacauxlsurf(char* linein,
                      char* singleword,
                      speciesGas* listspecGas,
                      int* NspecGas,
                      speciesSurf* listspecSurf,
                      int* NspecSurf,
                      reactionSurf* listreacSurf,
                      int* NreacSurf,
                      int* ierror)
{
  int len1;
  int ilenkey = 20, ilenval = 200;
  int inum, i, ireac; ///ipos, iswitch, indx; not used
  double dvalues[20];
  char* wordkey = (char*)malloc(ilenkey * sizeof(char));
  char* wordval = (char*)malloc(ilenval * sizeof(char));

  /* Return immediately if error flag is not zero */
  if (*ierror > 0)
    return 1;

  /* Store reaction location in the list */
  ireac = (*NreacSurf) - 1;

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

    if ((inum == 1) && ((strncmp(wordkey, "DUP", 3) != 0) &&
                        (strncmp(wordkey, "STICK", 5) != 0))) {
      *ierror = 711;
      return 1;
    }

    if ((inum == 2) && ((strncmp(wordkey, "DUP", 3) == 0) ||
                        (strncmp(wordkey, "STICK", 5) == 0))) {
      *ierror = 712;
      return 1;
    }

    if (strncmp(wordkey, "DUP", 3) == 0)
      listreacSurf[ireac].isdup = -1;

    else if (strncmp(wordkey, "STICK", 5) == 0)
      listreacSurf[ireac].isstick = 1;
    //
    else if (strncmp(wordkey, "COV", 3) == 0) {
      listreacSurf[ireac].iscov = 1;
      listreacSurf[ireac].cov_count +=1;

      std::vector<std::string> items;

      std::string line(wordval);
      std::string delimiter = " ";
      size_t pos = 0;
      std::string token;
      while ((pos = line.find(delimiter)) != std::string::npos) {
        items.push_back(line.substr(0, pos));
        line.erase(0, pos + delimiter.length());
      }
      items.push_back(line);

      double number(0);
      std::string species_name ;
      int count_cov(0);
      const int count_cov_reac = listreacSurf[ireac].cov_count - 1;
      for (size_t i = 0; i < items.size(); i++) {
        if(items[i].find_first_not_of(' ') != std::string::npos)
        {
        try
          {
            number = std::stod(items[i]);
            listreacSurf[ireac].cov_param[count_cov + 3*count_cov_reac ] = number;
            count_cov++;
#ifdef DEBUGMSG
            printf("var %s %e \n",items[i].c_str(), number);
#endif
          }
        catch(std::exception& e)
          {
            species_name = items[i];
          }

        }
      }

      int ipos = 0;
      char *specname = &species_name[0];

      // TCKMI_checkspecinlistsurf(specname, listspecSurf, NspecSurf, &ipos);
      // if (ipos == *NspecSurf) {
      //   printf("%s\n", specname);
      //   *ierror = 665;
      //   return 1;
      // }
      int isGas = 0;
      TCKMI_checkspecinlistsurf(specname, listspecSurf, NspecSurf, &ipos);
      if (ipos == *NspecSurf) {
        isGas = 1;
        TCKMI_checkspecinlistgas(specname, listspecGas, NspecGas, &ipos);
        if (ipos == *NspecGas) {
          printf("%s\n", specname);
          *ierror = 665;
          return 1;
        }
      }

      listreacSurf[ireac].cov_species_index[count_cov_reac] = ipos;
      listreacSurf[ireac].cov_isgas[count_cov_reac] = isGas ;
      listreacSurf[ireac].cov_reaction_index[count_cov_reac] = ireac;


#ifdef DEBUGMSG
    printf("COV %s species index %d, reaction index %d \n", species_name.c_str(), ipos, ireac );
#endif

    }


    else if (strncmp(wordkey, "REV", 3) == 0) {

      if (listreacSurf[ireac].isrevset > 0) {
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
        listreacSurf[ireac].arhenrev[i] = dvalues[i];
      listreacSurf[ireac].isrevset = 1;

    } /* End test for REV keyword */

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
          strncpy(listreacSurf[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "KCAL", 4) == 0)
          strncpy(listreacSurf[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "JOUL", 4) == 0)
          strncpy(listreacSurf[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "KJOU", 4) == 0)
          strncpy(listreacSurf[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "KELV", 4) == 0)
          strncpy(listreacSurf[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "EVOL", 4) == 0)
          strncpy(listreacSurf[ireac].eunits, wordkey, 4);
        else if (strncmp(wordkey, "MOLE", 4) == 0) {
          if (strncmp(&wordkey[4], "C", 1) == 0)
            strncpy(listreacSurf[ireac].aunits, "MOLC", 4);
          else
            strncpy(listreacSurf[ireac].aunits, wordkey, 4);
        } else {
          *ierror = 761;
          return 1;
        }
      }

    } /* End test for UNITS keyword */
    else {
      printf("TCKMI_getreacauxlsurf(): keyword %s not implemented!\n", wordkey);
      exit(1);
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
 *   (equation + forward Arrhenius parameters)
 */
int
TCKMI_getreaclinesurf(char* linein,
                      char* singleword,
                      speciesGas* listspec,
                      int* Nspec,
                      speciesSurf* listspecSurf,
                      int* NspecSurf,
                      reactionSurf* listreacSurf,
                      int* NreacSurf,
                      int* ierror)
{

  char specname[LENGTHOFSPECNAME], stoicoeff[LENGTHOFSPECNAME];
  char reac[lenstr02], prod[lenstr02];
  char *reacprod; /// , *reacfall, *prodfall; not used
  ///int ithrdb; /// not used
  int ipos = 0, ipos1 = 0; ///, ipos2 = 0; /// not used
  int i, len1;

  int *aplus, iplus;

  int j;
  ///int ireacthrdb = -2; //notused
  ///int iprodthrdb = -2; //notused

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
    listreacSurf[*NreacSurf].arhenfor[2 - i] = atof(singleword);
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
      listreacSurf[*NreacSurf].isrev = 1;
    } else if (strncmp(&linein[i], "=>", 2) == 0) {
      ipos = i;
      ipos1 = i + 1;
      listreacSurf[*NreacSurf].isrev = -1;
    } else if ((i > 0) && (strncmp(&linein[i], "=", 1) == 0) &&
               (strncmp(&linein[i - 1], "=", 1) != 0)) {
      ipos = i;
      ipos1 = i;
      listreacSurf[*NreacSurf].isrev = 1;
    }
    if (ipos > 0)
      break;

  } /* Done with the for checking for delimiters */

  if (ipos == 0) {
    *ierror = 630;
    return 1;
  }

#ifdef DEBUGMSG
  // printf("---> %d  %d %d\n", listreac[*Nreac].isrev, ipos, ipos1);
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
  printf("---> %d  |%s| |%s|\n", listreacSurf[*NreacSurf].isrev, reac, prod);
#endif
  /* Check for fall-of reactions */

  /* Check if reactant and product strings are still valid */
  if ((strlen(reac) == 0) || (strlen(prod) == 0)) {
    *ierror = 605;
    return 1;
  }

  /* Check if the reaction has real coeffs -> look for decimal points */

  /* Start looking for species in the strings of reactants and products */
  for (j = 0; j < 2; j++) {

    ///int icheckTB = 0;  // not used
    ///int icheckWL = 0;  // not used
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

      {
        int istcoeff = 1;
        ///double rstcoeff = 1.0; /// not used
        /* Check for numbers */
        len1 = strlen(specname);
        TCKMI_findnonnum(specname, &ipos);

        /* identify stoichiometric coefficients */
        if (ipos > 0) {
          memset(stoicoeff, 0, LENGTHOFSPECNAME);
          strncpy(stoicoeff, specname, ipos);
          istcoeff = atoi(stoicoeff);
        }

        int noRects = listreacSurf[*NreacSurf].inreac;
        int noProds = listreacSurf[*NreacSurf].inprod;
        if (j == 0) {

          /* reactants */
          if (noRects == NSPECREACMAX) {
            *ierror = 645;
            return 1;
          }
          listreacSurf[*NreacSurf].nuki[noRects] = -istcoeff;

        } else {
          /* products */
          if (noProds == NSPECREACMAX) {
            printf("%d\n", *NreacSurf);
            *ierror = 655;
            return 1;
          }
          listreacSurf[*NreacSurf].nuki[NSPECREACMAX + noProds] = istcoeff;
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

        int isSurf = 0;
        TCKMI_checkspecinlistgas(specname, listspec, Nspec, &ipos);
        if (ipos == *Nspec) {
          isSurf = 1;
          TCKMI_checkspecinlistsurf(specname, listspecSurf, NspecSurf, &ipos);
          if (ipos == *NspecSurf) {
            printf("%s\n", specname);
            *ierror = 665;
            return 1;
          }
        }

        if (j == 0) {

          int irep = -1;
          int kspec;
          for (kspec = 0; kspec < noRects; kspec++) {
            if (ipos == listreacSurf[*NreacSurf].spec[kspec] &&
                isSurf == listreacSurf[*NreacSurf].surf[kspec]) {
              irep = kspec;
              break;
            }
          }

          if (irep >= 0) {
            /* found repeat species, combine info */
            {
              listreacSurf[*NreacSurf].nuki[irep] +=
                listreacSurf[*NreacSurf].nuki[noRects];
              listreacSurf[*NreacSurf].nuki[noRects] = 0;
            }
          } else {
            /* found new species in the current reaction */
            listreacSurf[*NreacSurf].spec[noRects] = ipos;
            listreacSurf[*NreacSurf].surf[noRects] = isSurf;
            listreacSurf[*NreacSurf].inreac += 1;
          }

        } else {
          int irep = -1;
          int kspec;
          for (kspec = 0; kspec < noProds; kspec++) {
            if (ipos == listreacSurf[*NreacSurf].spec[NSPECREACMAX + kspec] &&
                isSurf == listreacSurf[*NreacSurf].surf[NSPECREACMAX + kspec]) {
              irep = kspec;
              break;
            }
          }

          if (irep >= 0) {
            /* found repeat species, combine info */
            {
              listreacSurf[*NreacSurf].nuki[NSPECREACMAX + irep] +=
                listreacSurf[*NreacSurf]
                  .nuki[NSPECREACMAX + listreacSurf[*NreacSurf].inprod];
              listreacSurf[*NreacSurf]
                .nuki[NSPECREACMAX + listreacSurf[*NreacSurf].inprod] = 0;
            }
          } else {
            listreacSurf[*NreacSurf].spec[NSPECREACMAX + noProds] = ipos;
            listreacSurf[*NreacSurf].surf[NSPECREACMAX + noProds] = isSurf;
            listreacSurf[*NreacSurf].inprod += 1;
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

/* ------------------------------------------------------------------------- */
/**
 *  \brief Returns position of a species in the list of species
 *   The index goes from 0 to (Nspec-1); if the species is not found
 *   the value of Nspec is returned
 */
void
TCKMI_checkspecinlistsurf(char* specname,
                          speciesSurf* listspecSurf,
                          int* NspecSurf,
                          int* ipos)
{

  int i;

  (*ipos) = (*NspecSurf);

  for (i = 0; i < (*NspecSurf); i++)
    if (strcmp(specname, listspecSurf[i].name) == 0) {
      (*ipos) = i;
      break;
    }

  return;
}

/* ------------------------------------------------------------------------- */
/**
 *  \brief Returns position of a species in the list of species
 *   The index goes from 0 to (Nspec-1); if the species is not found
 *   the value of Nspec is returned
 */
void
TCKMI_checkspecinlistgas(char* specname,
                         speciesGas* listspec,
                         int* Nspec,
                         int* ipos)
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
                  FILE* fileascii)
{
  int i, j; //, k, ipos; // not used

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
      fprintf(fileascii, "%3d ", listspec[i].elemcontent[j]);
    }
    fprintf(fileascii, "\n");
  }
  fprintf(fileascii, "\n");

  /* Output species */
  fprintf(
    fileascii, "Number of surface species in the mechanism : %d\n", *NspecSurf);
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

  for (i = 0; i < *NspecSurf; i++) {
    fprintf(fileascii,
            "  %3d  %18s     %9.3e   ",
            i + 1,
            listspecSurf[i].name,
            listspecSurf[i].mass);

    for (j = 0; j < *Nelem; j++) {
      fprintf(fileascii, "%3d ", listspecSurf[i].elemcontent[j]);
    }
    fprintf(fileascii, "\n");
  }
  fprintf(fileascii, "\n");

  /* Output reactions */
  fprintf(fileascii, "Number of reactions in the mechanism : %d\n", *NreacSurf);
  if (*NreacSurf == 0)
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

  for (i = 0; i < *NreacSurf; i++) {
    fprintf(fileascii, "%4d%s ", i + 1, "->");

    /* Output reactant string */
    for (j = 0; j < listreacSurf[i].inreac; j++) {
      {
        if (-listreacSurf[i].nuki[j] != 1)
          fprintf(fileascii, "%d", -listreacSurf[i].nuki[j]);
      }
      if (listreacSurf[i].surf[j] == 0)
        fprintf(fileascii, "%s", listspec[listreacSurf[i].spec[j]].name);
      else
        fprintf(fileascii, "%s", listspecSurf[listreacSurf[i].spec[j]].name);

      if (j != listreacSurf[i].inreac - 1)
        fprintf(fileascii, "%s", " + ");
    }

    /* Output delimiter */
    if (listreacSurf[i].isrev > 0)
      fprintf(fileascii, "%s", " <=> ");
    else
      fprintf(fileascii, "%s", " => ");

    /* Output product string */
    for (j = 0; j < listreacSurf[i].inprod; j++) {
      {
        if (listreacSurf[i].nuki[NSPECREACMAX + j] != 1)
          fprintf(fileascii, "%d", listreacSurf[i].nuki[NSPECREACMAX + j]);
      }
      if (listreacSurf[i].surf[NSPECREACMAX + j] == 0)
        fprintf(fileascii,
                "%s",
                listspec[listreacSurf[i].spec[NSPECREACMAX + j]].name);
      else
        fprintf(fileascii,
                "%s",
                listspecSurf[listreacSurf[i].spec[NSPECREACMAX + j]].name);

      if (j != listreacSurf[i].inprod - 1)
        fprintf(fileascii, "%s", " + ");
    }

    fprintf(fileascii, "\n");
    /* Duplicate */
    if (listreacSurf[i].isdup > -2) {
      fprintf(fileascii,
              "    --> This reaction is a duplicate %d",
              listreacSurf[i].isdup);
      fprintf(fileascii, "\n");
    }

    /* stick */
    if (listreacSurf[i].isstick == 1) {
      fprintf(fileascii, "    --> STICK\n");
    }

    /*surface coverage modification */
    if (listreacSurf[i].iscov == 1) {

      for (int k = 0; k < listreacSurf[i].cov_count; k++) {
        fprintf(fileascii, "    --> COV parameters : ");

        if (listreacSurf[i].cov_isgas[k]){
          fprintf(fileascii," %s ",listspec[listreacSurf[i].cov_species_index[k]].name);
        }else{
          fprintf(fileascii," %s ",listspecSurf[listreacSurf[i].cov_species_index[k]].name);
        }

        fprintf(fileascii,
                "%11.4e  %5.2f  %11.4e\n",
                listreacSurf[i].cov_param[3*k],
                listreacSurf[i].cov_param[3*k + 1],
                listreacSurf[i].cov_param[3*k + 2]);
      }
    }

    /* Arrhenius parameters forward */
    fprintf(fileascii, "    --> Arrhenius coefficients (forward) : ");
    fprintf(fileascii,
            "%11.4e  %5.2f  %11.4e\n",
            listreacSurf[i].arhenfor[0],
            listreacSurf[i].arhenfor[1],
            listreacSurf[i].arhenfor[2]);

    /* Arrhenius parameters reverse */
    if (listreacSurf[i].isrevset > 0) {
      fprintf(fileascii, "    --> Arrhenius coefficients (reverse) : ");
      fprintf(fileascii,
              "%11.4e  %5.2f  %11.4e\n",
              listreacSurf[i].arhenrev[0],
              listreacSurf[i].arhenrev[1],
              listreacSurf[i].arhenrev[2]);
    }

    if ((strncmp(listreacSurf[i].eunits, eunits, 4) != 0) ||
        (strncmp(listreacSurf[i].aunits, aunits, 4) != 0)) {
      fprintf(fileascii, "    --> Units : ");
      if (strncmp(listreacSurf[i].aunits, "MOLE", 4) == 0)
        fprintf(fileascii, "moles & ");
      else
        fprintf(fileascii, "molecules & ");

      if (strncmp(listreacSurf[i].eunits, "CAL/", 4) == 0)
        fprintf(fileascii, "cal/mole\n");
      else if (strncmp(listreacSurf[i].eunits, "KCAL", 4) == 0)
        fprintf(fileascii, "kcal/mole\n");
      else if (strncmp(listreacSurf[i].eunits, "JOUL", 4) == 0)
        fprintf(fileascii, "Joules/mole\n");
      else if (strncmp(listreacSurf[i].eunits, "KJOU", 4) == 0)
        fprintf(fileascii, "kJoules/mole\n");
      else if (strncmp(listreacSurf[i].eunits, "KELV", 4) == 0)
        fprintf(fileascii, "Kelvins\n");
      else if (strncmp(listreacSurf[i].eunits, "EVOL", 4) == 0)
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
TCKMI_outunformsurf(int* Nelem,
                    double siteden,
                    speciesSurf* listspecSurf,
                    int* NspecSurf,
                    reactionSurf* listreacSurf,
                    int* NreacSurf,
                    char* aunits,
                    char* eunits,
                    FILE* filelist,
                    int* ierror)
{
  int i, j, k;

  if (*ierror == 0)
    fprintf(filelist, "SUCCESS\n");
  else {
    fprintf(filelist, "ERROR  \n");
    return (1);
  }

  int maxTpRange =
    3; /* number of temperatures defining ranges for thermo props */
  int nNASAinter = maxTpRange - 1;
  int nCpCoef = 5; /* no. of coeffs for the specific heat polynomial */
  int nNASAfit = nCpCoef + 2; /* no. of coeff for thermo props */

  int maxSpecInReac = 0;
  for (i = 0; i < (*NreacSurf); i++) {
    /* ...maximum number of species in a reaction */
    maxSpecInReac = MAX(maxSpecInReac, 2 * (listreacSurf[i].inreac));
    maxSpecInReac = MAX(maxSpecInReac, 2 * (listreacSurf[i].inprod));
  }

  fprintf(filelist,
          "%12d\n",
          maxSpecInReac); /* Maximum number of species in a reaction */
  fprintf(filelist,
          "%12d\n",
          nNASAinter); /* Number of temperature regions for thermo fits */
  fprintf(filelist,
          "%12d\n",
          nCpCoef); /* Number of polynomial coefficients for thermo fits */
  fprintf(filelist, "3\n"); /* Number of Arrhenius parameters */

  fprintf(filelist, "%12d\n", *NspecSurf); /* Number of species */
  fprintf(filelist, "%12d\n", *NreacSurf); /* Number of reactions */

  int count_cov(0);
  for (i = 0; i < (*NreacSurf); i++) {
    if (listreacSurf[i].iscov == 1)
      //number of reaction with cov modification
      // reaction can has more than one cov modification
      count_cov += listreacSurf[i].cov_count;
  }

  fprintf(filelist, "%12d\n", count_cov); /* Number of reactions with cov modification*/

  /* Tolerance for reaction balance */
  fprintf(filelist, "%24.16e\n", REACBALANCE());

  /* surface site fraction */
  fprintf(filelist, "%24.16e\n", siteden);

  /* Species' name and weights */
  for (i = 0; i < (*NspecSurf); i++)
    fprintf(filelist, "%-32s\n", listspecSurf[i].name);
  for (i = 0; i < (*NspecSurf); i++)
    fprintf(filelist, "%24.16e\n", listspecSurf[i].mass);

  /* Species elemental compositions */
  for (i = 0; i < (*NspecSurf); i++) {
    for (j = 0; j < (*Nelem); j++) {
      fprintf(filelist, "%12d\n", listspecSurf[i].elemcontent[j]);
    } /* Done loop over elements */

  } /* Done loop over species */

  /* Species charges */
  for (i = 0; i < (*NspecSurf); i++)
    fprintf(filelist, "%12d\n", listspecSurf[i].charge);

  /* Number of temperature regions for thermo fits (2 for now) */
  for (i = 0; i < (*NspecSurf); i++)
    fprintf(filelist, "%12d\n", 2);

  /* Species phase (-1=solid,0=gas,+1=liquid) */
  for (i = 0; i < (*NspecSurf); i++)
    fprintf(filelist, "%12d\n", listspecSurf[i].phase);

  /* Temperature for thermo fits ranges */
  for (i = 0; i < (*NspecSurf); i++) {
    fprintf(
      filelist, "%24.16e\n", listspecSurf[i].nasapoltemp[0]); /* lower  bound */
    fprintf(
      filelist, "%24.16e\n", listspecSurf[i].nasapoltemp[2]); /* middle bound */
    fprintf(
      filelist, "%24.16e\n", listspecSurf[i].nasapoltemp[1]); /* upper  bound */
  }

  /* Polynomial coeffs for thermo fits */
  for (i = 0; i < (*NspecSurf); i++)
    for (j = 0; j < nNASAinter; j++)
      for (k = 0; k < nNASAfit; k++)
        fprintf(filelist,
                "%24.16e\n",
                listspecSurf[i].nasapolcoefs[j * nNASAfit + k]);

  /* Basic reaction info */
  if ((*NreacSurf) > 0) {
    /* No. of reactants+products (negative values for irreversible reactions) */
    for (i = 0; i < (*NreacSurf); i++) {
      if (listreacSurf[i].isrev < 0)
        fprintf(filelist,
                "%12d\n",
                -(listreacSurf[i].inreac + listreacSurf[i].inprod));
      else
        fprintf(
          filelist, "%12d\n", listreacSurf[i].inreac + listreacSurf[i].inprod);
    }

    /* No. of reactants only */
    for (i = 0; i < (*NreacSurf); i++)
      fprintf(filelist, "%12d\n", listreacSurf[i].inreac);

    /* Stoichiometric coefficients */
    for (i = 0; i < (*NreacSurf); i++) {
      int nusumk = 0;
      for (j = 0; j < maxSpecInReac; j++) {
        if (j < maxSpecInReac / 2) {
          fprintf(filelist, "%12d\n", listreacSurf[i].nuki[j]);
          fprintf(filelist, "%12d\n", listreacSurf[i].surf[j]);
          fprintf(filelist, "%12d\n", listreacSurf[i].spec[j] + 1);
          nusumk += listreacSurf[i].nuki[j];
        } else {
          int j1;
          j1 = j - maxSpecInReac / 2;
          fprintf(filelist, "%12d\n", listreacSurf[i].nuki[NSPECREACMAX + j1]);
          fprintf(filelist, "%12d\n", listreacSurf[i].surf[NSPECREACMAX + j1]);
          fprintf(
            filelist, "%12d\n", listreacSurf[i].spec[NSPECREACMAX + j1] + 1);
          nusumk += listreacSurf[i].nuki[NSPECREACMAX + j1];
        }
      }
      fprintf(filelist, "%12d\n", nusumk);

    } /* Done with stoichimetric coefficients */

    /* Arrhenius parameters */
    for (i = 0; i < (*NreacSurf); i++) {
      fprintf(filelist, "%24.16e\n", listreacSurf[i].arhenfor[0]);
      fprintf(filelist, "%24.16e\n", listreacSurf[i].arhenfor[1]);
      fprintf(filelist, "%24.16e\n", listreacSurf[i].arhenfor[2]);
    }

    /* If reactions are duplicates or not */
    for (i = 0; i < (*NreacSurf); i++) {
      if (listreacSurf[i].isdup > -2)
        fprintf(filelist, "1\n"); /* yes */
      else
        fprintf(filelist, "0\n"); /* no  */
    }

    /* If reactions are stick or not */
    for (i = 0; i < (*NreacSurf); i++) {
      if (listreacSurf[i].isstick == 1)
        fprintf(filelist, "1\n"); /* yes */
      else
        fprintf(filelist, "0\n"); /* no  */
    }

    /* If reactions needs a coverage modification or not */
    for (i = 0; i < (*NreacSurf); i++) {
      if (listreacSurf[i].iscov == 1){
        fprintf(filelist, "1\n"); /* yes it is cov*/
        const int NumofCOV =  listreacSurf[i].cov_count;
        fprintf(filelist,"%d \n", NumofCOV); /*Number of COV in reaction*/
        for (int k = 0; k < NumofCOV; k++) {
          fprintf(filelist,"%d \n",listreacSurf[i].cov_species_index[k]); // species index
          fprintf(filelist,"%d \n",listreacSurf[i].cov_isgas[k]); // is surface?
          fprintf(filelist,"%d \n",listreacSurf[i].cov_reaction_index[k]); // reaction index
          fprintf(filelist, "%24.16e \n",listreacSurf[i].cov_param[3*k]); //eta
          fprintf(filelist, "%24.16e \n",listreacSurf[i].cov_param[3*k + 1]); // mu
          fprintf(filelist, "%24.16e \n",listreacSurf[i].cov_param[3*k + 2]); //epsilon
        }

      } else
        fprintf(filelist, "0\n"); /* no, it is not cov  */
    }


  } /* Done if Nreac > 0 */

  return (0);

} /* Done with "TCKMI_outunform" */

/* ------------------------------------------------------------------------- */
/**
 * \brief Verifies corectness and completness for all reactions in the list
 */

int
TCKMI_rescalereacsurf(reactionSurf* listreac, int* Nreac)
{

  int i, j;
  double factor;

  for (i = 0; i < (*Nreac); i++) {

    /* Pre-exponential factor */
    factor = 1.0;
    if (strncmp(listreac[i].aunits, "MOLC", 4) == 0) {
      /* Found molecules for pre-exponential factor, need to rescale */
      double nureac = 0, nuprod = 0;
      {
        /* Sum the (real) stoichiometric coefficients */
        for (j = 0; j < listreac[i].inreac; j++)
          nureac += listreac[i].nuki[j];
        for (j = 0; j < listreac[i].inprod; j++)
          nuprod += listreac[i].nuki[NSPECREACMAX + j];
      }

      /* Scale only if not MOME or XSMI */
      {
        {
          factor = pow(NAVOG, nureac - 1);
          listreac[i].arhenfor[0] *= factor;
          if (listreac[i].isrev > 0) {
            factor = pow(NAVOG, nuprod - 1);
            listreac[i].arhenrev[0] *= factor;
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

    /* if reaction has cov modification*/
    if (listreac[i].iscov == 1){
      //epsilon
      for (int k = 0; k <listreac[i].cov_count; k++) {
        listreac[i].cov_param[3*k + 2] *= factor;
      }
    }

  }

  return 0;

} /* Done with "TCKMI_rescalereacsurf" */

int
TCKMI_saveSurfRectionEquations(speciesGas* listspec,
                     speciesSurf* listspecSurf,
                     reactionSurf* listreacSurf,
                     int* NreacSurf,
                     FILE* fileascii)
{

  for (int i = 0; i < *NreacSurf; i++) {
    /* Output reactant string */
    for (int j = 0; j < listreacSurf[i].inreac; j++) {
      {
        if (-listreacSurf[i].nuki[j] != 1)
          fprintf(fileascii, "%d", -listreacSurf[i].nuki[j]);
      }
      if (listreacSurf[i].surf[j] == 0)
        fprintf(fileascii, "%s", listspec[listreacSurf[i].spec[j]].name);
      else
        fprintf(fileascii, "%s", listspecSurf[listreacSurf[i].spec[j]].name);

      if (j != listreacSurf[i].inreac - 1)
        fprintf(fileascii, "%s", " + ");
    }

    /* Output delimiter */
    if (listreacSurf[i].isrev > 0)
      fprintf(fileascii, "%s", " <=> ");
    else
      fprintf(fileascii, "%s", " => ");

    /* Output product string */
    for (int j = 0; j < listreacSurf[i].inprod; j++) {
      {
        if (listreacSurf[i].nuki[NSPECREACMAX + j] != 1)
          fprintf(fileascii, "%d", listreacSurf[i].nuki[NSPECREACMAX + j]);
      }
      if (listreacSurf[i].surf[NSPECREACMAX + j] == 0)
        fprintf(fileascii,
                "%s",
                listspec[listreacSurf[i].spec[NSPECREACMAX + j]].name);
      else
        fprintf(fileascii,
                "%s",
                listspecSurf[listreacSurf[i].spec[NSPECREACMAX + j]].name);

      if (j != listreacSurf[i].inprod - 1)
        fprintf(fileascii, "%s", " + ");
    }

    fprintf(fileascii, ",\n");

  }


  return 0;
}
