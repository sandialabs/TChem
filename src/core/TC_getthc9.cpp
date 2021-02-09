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

/* ------------------------------------------------------------------------- */
/**
 * \brief Reads thermodynamic properties (NASA polynomials) from the
 *        mechanism input file or from a separate file
 */
int
TCKMI_readValidLine(FILE* filein, char* linein)
{
  int len1 = 0;
  while ((len1 == 0) && (feof(filein) == 0)) {
    fgets(linein, 200, filein);
    TCKMI_cleancharstring(linein, &len1);
  }
  return (0);
}
int
TCKMI_getthermo9(char* singleword,
                 FILE* thermoin,
                 element* listelem,
                 int* Nelem,
                 species* listspec,
                 int* Nspec,
                 int* ithermo,
                 int* iread,
                 int* ierror)
{

  char linein1[200], linein2[200], linein3[200];
  /// double dtemp; /// not used
  int itemp, Nth9rng, Nth9npow, pows[7];
  FILE* thermofile;
  int i, irng, ip, len1;

  /* Exit if the error flag is not zero */
  if (*ierror > 0)
    return (1);

  /* Exit if value of iread flag is not appropriate (4) */
  if (*iread != 4) {
    (*ierror) = 300;
    return (1);
  }

  /* Set up the input file */
  thermofile = thermoin;

  /* Check if file pointer is null or if file is empty */
  if (thermofile == NULL) {
    (*ierror) = 320;
    return (1);
  }
  if (feof(thermofile) > 0) {
    (*ierror) = 338;
    return (1);
  }

  len1 = 0;
  while (len1 == 0) {
    /* Reading the first line in the file */
    fgets(linein1, 200, thermofile);
    if (feof(thermofile) > 0) {
      (*ierror) = 340;
      return (1);
    }
    TCKMI_cleancharstring(linein1, &len1);
  }

  len1 = 0;
  while (len1 == 0) {
    /* Reading the second line in the file, hopefully the temperature range */
    fgets(linein1, 200, thermofile);
    if (feof(thermofile) > 0) {
      (*ierror) = 342;
      return (1);
    }
    TCKMI_cleancharstring(linein1, &len1);
  }

  /* Start reading properties */
  while (feof(thermofile) == 0) {

    int ipos, ipos1;

    /* Clear memory locations */
    memset(linein1, 0, 200);
    memset(linein2, 0, 200);
    memset(linein3, 0, 200);

    TCKMI_readValidLine(thermofile, linein1);
    TCKMI_cleancharstring(linein1, &len1);

    /* If end of file exit */
    if (feof(thermofile) > 0) {
      if (len1 == 0) {
        return (0);
      }
      return (1);
    }

    /* Test for keyword */
    strncpy(singleword, linein1, 4);
    TCKMI_wordtoupper(singleword, singleword, 4);

    /* If "END" is encountered */
    if (strncmp(singleword, "END", 3) == 0) {
      return (0);
    }

    TCKMI_readValidLine(thermofile, linein2);
    if (feof(thermofile) > 0) {
      (*ierror) = 360;
      return (1);
    }

    /* Get species name and convert it to uppercase */
    len1 = strcspn(linein1, " ");
    strncpy(singleword, linein1, len1);
    singleword[len1] = 0;
    TCKMI_cleancharstring(singleword, &len1);
    TCKMI_wordtoupper(singleword, singleword, len1);

    /* Check if species name is not null */
    if (len1 == 0) {
      (*ierror) = 370;
      return 1;
    }

    /* Get number of temperature ranges */
    len1 = 2;
    strncpy(singleword, linein2, len1);
    singleword[len1] = 0;
    TCKMI_checkstrnum(singleword, &len1, ierror);
    if (*ierror > 0)
      return (1);
    Nth9rng = (int)atof(singleword);

    /* Check if species is of interest */
    TCKMI_checkspecinlist(singleword, listspec, Nspec, &ipos);

    if (ipos < (*Nspec)) {
      /* Found species of interest */
      listspec[ipos].hasthermo = 2;

      /* Get number of temperature ranges */
      listspec[ipos].Nth9rng = Nth9rng;

#ifdef DEBUGMSG
      printf("found species %18s Trange: ", listspec[ipos].name);
#endif

      /* Done with temperature range, check element content */
      for (i = 0; i < NUMBEROFELEMINSPEC; i++) {
        listspec[ipos].elemindx[i] = -1;
        listspec[ipos].elemcontent[i] = -1;
      }
      for (i = 0; i < 5; i++) {
        itemp = 10 + i * 8;
        /* Extract element name */
        strncpy(singleword, &linein2[itemp], 2);
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
          (*ierror) = 380;
          return (1);
        }

        listspec[ipos].elemindx[i] = ipos1;

        /* Good element, find numbers */
        len1 = 6;
        strncpy(singleword, &linein2[itemp + 2], len1);
        singleword[len1] = 0;
        TCKMI_checkstrnum(singleword, &len1, ierror);
        if (*ierror > 0)
          return 1;
        itemp = (int)atof(singleword);

        listspec[ipos].elemcontent[i] = itemp;

        /* See if charge needs to be updated */
        if (strncmp(listelem[ipos1].name, "E", 1) == 0)
          listspec[ipos].charge += listspec[ipos].elemindx[i] * (-1);

      } /* done checking for element content */

      listspec[ipos].numofelem = i;

      /* Update mass */
      len1 = 13;
      strncpy(singleword, &linein2[52], len1);
      singleword[len1] = 0;
      TCKMI_checkstrnum(singleword, &len1, ierror);
      if (*ierror > 0)
        return (1);
      listspec[ipos].mass = atof(singleword);
      listspec[ipos].hasmass = 1;

      if (listspec[ipos].Nth9rng == 0) {
        /* No temperature range */
        TCKMI_readValidLine(thermofile, linein1);
      } else {

        for (irng = 0; irng < listspec[ipos].Nth9rng; irng++) {

          TCKMI_readValidLine(thermofile, linein1);
          TCKMI_readValidLine(thermofile, linein2);
          TCKMI_readValidLine(thermofile, linein3);

          /* Extract temperature range */
          len1 = 11;
          strncpy(singleword, &linein1[0], len1);
          singleword[len1] = 0;
          TCKMI_checkstrnum(singleword, &len1, ierror);
          if (*ierror > 0)
            return (1);
          listspec[ipos].Nth9Temp[2 * irng] = atof(singleword);

          strncpy(singleword, &linein1[len1], len1);
          singleword[len1] = 0;
          TCKMI_checkstrnum(singleword, &len1, ierror);
          if (*ierror > 0)
            return (1);
          listspec[ipos].Nth9Temp[2 * irng + 1] = atof(singleword);

          /* Extract number of polynomial coefficients */
          len1 = 1;
          strncpy(singleword, &linein1[22], len1);
          singleword[len1] = 0;
          TCKMI_checkstrnum(singleword, &len1, ierror);
          if (*ierror > 0)
            return (1);
          Nth9npow = (int)atof(singleword);

          /* get powers of polynomial terms */
          for (ip = 0; ip < Nth9npow; ip++) {
            len1 = 5;
            itemp = 23 + len1 * i;
            strncpy(singleword, &linein1[itemp], len1);
            singleword[len1] = 0;
            TCKMI_checkstrnum(singleword, &len1, ierror);
            pows[ip] = (int)atof(singleword);
          }

          /* get polynomial coefficients */
          /* a1-a5 */
          for (ip = 0; ip < MIN(Nth9npow, 5); ip++) {
            len1 = 16;
            itemp = 0 + len1 * i;
            strncpy(singleword, &linein2[itemp], len1);
            singleword[len1] = 0;
            TCKMI_checkstrnum(singleword, &len1, ierror);
            listspec[ipos].Nth9coefs[irng * 9 + pows[ip] + 2] =
              atof(singleword);
          }
          /* a6-a7 */
          for (ip = 6; ip < MIN(Nth9npow, 7); ip++) {
            len1 = 16;
            itemp = 0 + len1 * i;
            strncpy(singleword, &linein3[itemp], len1);
            singleword[len1] = 0;
            TCKMI_checkstrnum(singleword, &len1, ierror);
            listspec[ipos].Nth9coefs[irng * 9 + pows[ip] + 2] =
              atof(singleword);
          }
          /* b1-b2 */
          for (ip = 8; ip < 9; ip++) {
            len1 = 16;
            itemp = 48 + len1 * i;
            strncpy(singleword, &linein3[itemp], len1);
            singleword[len1] = 0;
            TCKMI_checkstrnum(singleword, &len1, ierror);
            listspec[ipos].Nth9coefs[irng * 9 + ip] = atof(singleword);
          }

        } /* done loop over all temp ranges */

      } /* done reading polynomial coefficients */

    } /* Done with if the species is of interest */
    else {
      if (listspec[ipos].Nth9rng == 0) {
        /* No temperature range */
        TCKMI_readValidLine(thermofile, linein1);
      } else {

        for (irng = 0; irng < listspec[ipos].Nth9rng; irng++) {

          TCKMI_readValidLine(thermofile, linein1);
          TCKMI_readValidLine(thermofile, linein2);
          TCKMI_readValidLine(thermofile, linein3);
        }
      }
    } /* done skipping over a species */

  } /* done reading the file */

  return 0;
}
