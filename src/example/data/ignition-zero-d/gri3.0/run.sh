
exec=../../../TChem_IgnitionZeroDSA.x
inputs=../gri3.0/
save=20
dtmin=1e-8
dtmax=1e-3
tend=2
max_time_iterations=520
max_newton_iterations=100
atol_newton=1e-12
rtol_newton=1e-6
tol_time=1e-4

$exec --inputsPath=$inputs --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations
