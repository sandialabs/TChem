
exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDSA.x
inputs=inputs/
use_prefixPath=false
chemfile=$inputs"chem_ndodecane.inp"
thermfile=$inputs"therm_ndodecane.dat"
samplefile=$inputs"sample.dat"
save=1
dtmin=1e-20
dtmax=1e-1
tend=0.5
max_time_iterations=200
max_newton_iterations=20
atol_newton=1e-18
rtol_newton=1e-8
tol_time=1e-7
OnlyComputeIgnDelayTime=false

$exec --use_prefixPath=$use_prefixPath --chemfile=$chemfile --thermfile=$thermfile --samplefile=$samplefile --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations
