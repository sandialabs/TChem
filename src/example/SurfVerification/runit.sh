
TCHEM_INSTALL=/Users/odiazib/csp_clang_bld_develop/TChem++/install
exec=$TCHEM_INSTALL/example/TChem_NetProductionRateSurfacePerMass.x
inputs=data/reaction-rates-surfaces/PT/

chemfile=$inputs'chem.inp'
thermfile=$inputs'therm.dat'
chemSurfFile=$inputs'chemSurf.inp'
thermSurffile=$inputs'thermSurf.dat'
#operating condition files
inputfile=$inputs'inputGas.dat'
inputfileSurf=$inputs'inputSurfGas.dat'
#output files
outputfile='omega.dat'
outputfileGasSurf='omegaGasSurf.dat'
outputfileSurf='omegaSurf.dat'

$exec --chemfile=$chemfile --chemfile=$chemfile --thermfile=$thermfile --chemSurffile=$chemSurfFile --thermSurffile=$thermSurffile --inputfile=$inputfile --inputfileSurf=$inputfileSurf --outputfile=$outputfile --outputfileGasSurf=$outputfileGasSurf --outputfileSurf=$outputfileSurf
