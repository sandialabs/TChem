
exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDJacobians.x
inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
use_sample_format=true
outputfile=".dat"
#N=1
#N=100
#N=400
#N=3600
#N=10000
#N=1000000  

for N in 1 100 400 3600 10000
do
 samplefile="sample"$N".dat"
 output_file_times="wall_times"$N".dat"
 OMP_NUM_THREADS=8 OMP_PLACES=threads  OMP_PROC_BIND=spread $exec --outputfile=$outputfile --verbose=true --output-file-times=$output_file_times --use-sample-format=$use_sample_format --chemfile=$chemfile --thermfile=$thermfile --inputfile=$samplefile
done
