
exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDSA.x
inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/
dtmin=1e-20
dtmax=1e-3
tend=2
max_time_iterations=260
max_newton_iterations=20
atol_newton=1e-18
rtol_newton=1e-8
tol_time=1e-6
time_iterations_per_interval=10

$exec --inputsPath=$inputs --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --time-iterations-per-interval=$time_iterations_per_interval --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations
