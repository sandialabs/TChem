
exec=../../../TChem_IgnitionZeroDSA.x
inputs=../CO/
save=500
dtmin=1e-8
dtmax=1e-3
tend=10
max_time_iterations=1000
T_threshold=1500
max_newton_iterations=500
atol_newton=1e-10
rtol_newton=1e-6
tol_time=1e-4

export OMP_NUM_THREADS=48

$exec --inputsPath=$inputs --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --max-newton-iterations=$max_newton_iterations --T_threshold=$T_threshold --output_frequency=$save --dtmin=$dtmin --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations 

