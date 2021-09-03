
exec=$TCHEM_INSTALL_PATH/example/TChem_AtmosphericChemistry.x
inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/
chemfile="config_troe/config_troe.yaml"
dtmin=1e-20
dtmax=1e-1
tend=100
max_time_iterations=100
max_newton_iterations=20
atol_newton=1e-18
rtol_newton=1e-8
tol_time=1e-8
time_iterations_per_interval=50
outputfile="Troe.dat"

$exec --chemfile=$chemfile  --outputfile=$outputfile --unit-test="troe" --time-iterations-per-interval=$time_iterations_per_interval --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations
