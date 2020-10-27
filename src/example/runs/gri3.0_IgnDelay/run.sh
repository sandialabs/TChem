
exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDSA.x
inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/
use_prefixPath=false
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
samplefile=inputs/"sample.dat"
save=-1
dtmin=1e-8
dtmax=1e-1
tend=200
max_time_iterations=500
max_newton_iterations=20
atol_newton=1e-12
rtol_newton=1e-6
tol_time=1e-6
OnlyComputeIgnDelayTime=true

$exec --use_prefixPath=$use_prefixPath --chemfile=$chemfile --thermfile=$thermfile --samplefile=$samplefile --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations
