def getMassFractionH2(phi):
    ne = 2 
    EW = [1.00797, 15.99940] # H O 
    
    # define species compositions and names
    iFU  = [2, 0]
    iO2  = [0, 1]
    iH2O = [2, 1]

    ispec = {'Fuel':iFU, 'O2' :iO2 , \
             'H2O':iH2O}
    
    # compute molecular weights
    for spec in ispec:
        sumW = 0.0
        i    = 0
        while i < ne:
            sumW += ispec[spec][i]*EW[i]
            i += 1
        ispec[spec].append(sumW)
    Wfuel = ispec['Fuel'][ne]
    Wo2   = ispec['O2'  ][ne]
    Wh2o  = ispec['H2O' ][ne]
    
    small  = 1.e-15
    
    # compute REACTANTS
    alpha  =  1. / phi 
    denom   = Wfuel+alpha*Wo2
    Yr_fuel = Wfuel/denom
    Yr_o2   = alpha *Wo2  /denom

    return Yr_fuel, Yr_o2

def getMassFractionCO(phi):
    ne = 5
    EW = [1.00797, 15.99940, 12.01115, 14.00670, 39.94800] # H O C N AR
    
    # define species compositions and names
    iFU  = [0, 0, 1, 0, 0]
    iO2  = [0, 2, 0 , 0, 0]
    iH2O = [2, 1, 0 , 0, 0]
    iCO2 = [0, 2, 1 , 0, 0]
    iN2  = [0, 0, 0 , 2, 0]
    iAR  = [0, 0, 0 , 0, 1]

    ispec = {'Fuel':iFU, 'O2' :iO2 , \
             'CO2':iCO2, 'H2O':iH2O, \
             'N2' :iN2 , 'AR' :iAR}
    
    # define N2/O2 mole ratio in air
    #n2o2air = 0.79/0.21
    n2o2air = 0.7808/0.2095
    aro2air = 0.0097/0.2095
    
    # compute molecular weights
    for spec in ispec:
        sumW = 0.0
        i    = 0
        while i < ne:
            sumW += ispec[spec][i]*EW[i]
            i += 1
        ispec[spec].append(sumW)
    Wfuel = ispec['Fuel'][ne]
    Wo2   = ispec['O2'  ][ne]
    Wn2   = ispec['N2'  ][ne]
    Wco2  = ispec['CO2' ][ne]
    Wh2o  = ispec['H2O' ][ne]
    War   = ispec['AR'  ][ne]
    
    small  = 1.e-15
    
    # compute REACTANTS
    alpha  =  1. / phi 
    betaN2 = alpha * n2o2air
    betaAR = alpha * aro2air
    
    denom   = Wfuel+alpha*Wo2+betaN2*Wn2+betaAR*War
    Yr_fuel = Wfuel/denom
    Yr_o2   = alpha *Wo2  /denom
    Yr_n2   = betaN2*Wn2  /denom
    Yr_ar   = betaAR*War  /denom

    return Yr_fuel, Yr_o2, Yr_n2, Yr_ar

def getPhifromYo2(nC,Yr_o2):
    ne = 5
    EW = [1.00797, 15.99940, 12.01115, 14.00670, 39.94800] # H O C N AR
    
    # define species compositions and names
    iFU  = [2*nC+2, 0, nC, 0, 0]
    iO2  = [0     , 2, 0 , 0, 0]
    iH2O = [2     , 1, 0 , 0, 0]
    iCO2 = [0     , 2, 1 , 0, 0]
    iN2  = [0     , 0, 0 , 2, 0]
    iAR  = [0     , 0, 0 , 0, 1]
    
    ispec = {'Fuel':iFU, 'O2' :iO2 , \
             'CO2':iCO2, 'H2O':iH2O, \
             'N2' :iN2 , 'AR' :iAR}
    
    # define N2/O2 mole ratio in air
    #n2o2air = 0.79/0.21
    n2o2air = 0.7808/0.2095
    aro2air = 0.0097/0.2095
    # compute molecular weights
    for spec in ispec:
        sumW = 0.0
        i    = 0
        while i < ne:
            sumW += ispec[spec][i]*EW[i]
            i += 1
        ispec[spec].append(sumW)
    Wfuel = ispec['Fuel'][ne]
    Wo2   = ispec['O2'  ][ne]
    Wn2   = ispec['N2'  ][ne]
    Wco2  = ispec['CO2' ][ne]
    Wh2o  = ispec['H2O' ][ne]
    War   = ispec['AR'  ][ne]
    
    alpha = Wfuel/(Wo2/Yr_o2 - (Wo2 + n2o2air*Wn2 + aro2air*War) )
    # compute REACTANTS
    return (3*nC+1) / (2*alpha)

def getMassFraction(nC, phi, print_output=False):
    
    outtype = 0
    # define element mass (taken from ckinterp)
    ne = 5
    EW = [1.00797, 15.99940, 12.01115, 14.00670, 39.94800] # H O C N AR
    
    # define species compositions and names
    iFU  = [2*nC+2, 0, nC, 0, 0]
    iO2  = [0     , 2, 0 , 0, 0]
    iH2O = [2     , 1, 0 , 0, 0]
    iCO2 = [0     , 2, 1 , 0, 0]
    iN2  = [0     , 0, 0 , 2, 0]
    iAR  = [0     , 0, 0 , 0, 1]
    
    ispec = {'Fuel':iFU, 'O2' :iO2 , \
             'CO2':iCO2, 'H2O':iH2O, \
             'N2' :iN2 , 'AR' :iAR}
    
    # define N2/O2 mole ratio in air
    #n2o2air = 0.79/0.21
    n2o2air = 0.7808/0.2095
    aro2air = 0.0097/0.2095
    
    # compute molecular weights
    for spec in ispec:
        sumW = 0.0
        i    = 0
        while i < ne:
            sumW += ispec[spec][i]*EW[i]
            i += 1
        ispec[spec].append(sumW)
    Wfuel = ispec['Fuel'][ne]
    Wo2   = ispec['O2'  ][ne]
    Wn2   = ispec['N2'  ][ne]
    Wco2  = ispec['CO2' ][ne]
    Wh2o  = ispec['H2O' ][ne]
    War   = ispec['AR'  ][ne]
    
    small  = 1.e-15
    
    # compute REACTANTS
    alpha  =  (3*nC+1) / (2*phi) 
    betaN2 = alpha * n2o2air
    betaAR = alpha * aro2air
    
    denom   = Wfuel+alpha*Wo2+betaN2*Wn2+betaAR*War
    Yr_fuel =        Wfuel/denom
    Yr_o2   = alpha *Wo2  /denom
    Yr_n2   = betaN2*Wn2  /denom
    Yr_ar   = betaAR*War  /denom
    denom   = 1.0+alpha+betaN2+betaAR
    Xr_fuel = 1.0   /denom
    Xr_o2   = alpha /denom
    Xr_n2   = betaN2/denom
    Xr_ar   = betaAR/denom
   
    
    if  ( outtype == 1 ) :
        if ( nC == 8 ) :
            print ('spec IC8H18  ',Xr_fuel)
        elif ( nC == 7 ) :
            print ('spec NC7H16  ',Xr_fuel)
        else :
            if ( nC == 0 ):
                print ('spec H{0s:d}     '.format(int(2*nC+2)), Xr_fuel)
            elif nC == 1 :
                print ('spec CH{0:d}   '.format(4), Xr_fuel)
            else :# nC > 1
                print ('spec C{0:d}H{1:d}   s'.format(int(nC),int(2*nC+2)), Xr_fuel)
                #print 'spec C{0:d}H{1:d}   {other}'.format(int(nC),int(2*nC+2),other=Xr_fuel)
        print ('spec O2      ',Xr_o2)
        print ('spec N2      ',Xr_n2)
        print ('spec AR      ',Xr_ar)
        sys.exit()
    if print_output:  
       print ( 'Reactants for equivalence ratio ',phi)
       print ( 'MASS')
       if ( Yr_fuel > small ) :
           print ( 'REAC  FUEL ',Yr_fuel)
       if ( Yr_o2  > small ) :
           print ( 'REAC  O2   ',Yr_o2)
       if ( Yr_n2  > small ) :
           print ( 'REAC  N2   ',Yr_n2)
       if ( Yr_ar  > small ) :
           print ( 'REAC  AR   ',Yr_ar)
       print ( 'MOLE')
       if ( Xr_fuel > small ) :
           print ( 'REAC  FUEL ',Xr_fuel)
       if ( Xr_o2  > small ) :
           print ( 'REAC  O2   ',Xr_o2)
       if ( Xr_n2  > small ) :
           print ( 'REAC  N2   ',Xr_n2)
       if ( Xr_ar  > small ) :
           print ( 'REAC  AR   ',Xr_ar)
       
       # compute PRODUCTS
       denom = Wco2*min(alpha/2,1.0)+Wh2o*min(alpha,2)+Wn2*betaN2+War*betaAR\
              +Wfuel*max(1-alpha/2,0)+Wo2*max(alpha-2,0.0)
       Yp_fuel= Wfuel*max(1-alpha/2,0)/denom
       Yp_o2  = Wo2  *max(alpha-2,0.0)/denom
       Yp_co2 = Wco2 *min(alpha/2,1.0)/denom
       Yp_h2o = Wh2o *min(alpha,2)    /denom
       Yp_n2  = Wn2  *betaN2          /denom
       Yp_ar  = War  *betaAR          /denom
       denom  = 1.0+alpha+betaN2+betaAR
       Xp_fuel= max(1-alpha/2,0)/denom
       Xp_o2  = max(alpha-2,0.0)/denom
       Xp_co2 = min(alpha/2,1.0)/denom
       Xp_h2o = min(alpha,2)    /denom
       Xp_n2  = betaN2          /denom
       Xp_ar  = betaAR          /denom
      
       print ( 'Products for equivalence ratio ',phi)
       print ( 'MASS')
       if ( Yp_fuel > small ) :
           print ( 'PROD  FUEL ',Yp_fuel)
       if ( Yp_o2  > small ) :
           print ( 'PROD  O2   ',Yp_o2)
       if ( Yp_co2  > small ) :
           print ( 'PROD  CO2  ',Yp_co2)
       if ( Yp_h2o  > small ) :
           print ( 'PROD  H2O  ',Yp_h2o)
       if ( Yp_n2  > small ) :
           print ( 'PROD  N2   ',Yp_n2)
       if ( Yp_ar  > small ) :
           print ( 'PROD  AR   ',Yp_ar)
       print ( 'MOLE' )
       if ( Xp_fuel > small ) :
           print ( 'PROD  FUEL ',Xp_fuel)
       if ( Xp_o2  > small ) :
           print ( 'PROD  O2   ',Xp_o2)
       if ( Xp_co2  > small ) :
           print ( 'PROD  CO2  ',Xp_co2)
       if ( Xp_h2o  > small ) :
           print ( 'PROD  H2O  ',Xp_h2o)
       if ( Xp_n2  > small ) :
           print ( 'PROD  N2   ',Xp_n2)
       if ( Xp_ar  > small ) :
           print ( 'PROD  AR   ',Xp_ar)
    return  Yr_fuel, Yr_o2, Yr_n2, Yr_ar
