TCHEM_INSTALL_PATH=${HOME}/Documents/CODE/Getz/install/tchem
exec=$TCHEM_INSTALL_PATH/example/TChem_AtmosphericChemistry.x
chemfile="config_chapman.yaml"
dtmin=1e-20
dtmax=1e2
tend=1000000
max_time_iterations=1000
max_newton_iterations=20
atol_newton=1e-18
rtol_newton=1e-8
tol_time=1e-6
outputfile="Chapman.dat"

$exec --chemfile=$chemfile --outputfile=$outputfile --time-iterations-per-interval=10 --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations
