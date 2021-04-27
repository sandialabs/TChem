
exec=$TCHEM_INSTALL_PATH/example/TChem_ThermalProperties.x
inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
N=1
N=100
N=400
N=3600
N=10000
N=1000000
samplefile="sample"$N".dat"

OMP_NUM_THREADS=8 OMP_PLACES=threads  OMP_PROC_BIND=spread $exec --chemfile=$chemfile --thermfile=$thermfile --inputfile=$samplefile
