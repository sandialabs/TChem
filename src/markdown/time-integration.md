
We present the setup for canonical examples that are available through TChem. All models presented in this section are setup to be run in parallel, possibly exploiting several layers of parallelism available on the platform of choice. We start with a description of a 2-nd order backward differentiation formula (BDF2) time stepping algorithm in [Section](#timeintegration). BDF2 was implemented via Kokkos and takes advantage of parallel threads available through the Kokkos interface. We then present results for homogenous batch reactors in [Section](#0dignition), and the plug-flow reactor, in [Section](#plugflowreactorpfrproblemwithgasandsurfacesreactions).

# Time Integration

For solving a stiff time ODEs, a time step size is limited by a stability condition rather than a truncation error. To obtain a reliable solution, we use a stable time integration method i.e., 2nd order Trapezoidal Backward Difference Formula (TrBDF2). The TrBDF2 scheme is a composite single step method. The method is 2nd order accurate and $L$-stable.

* R. E. Bank, W. M. Coughran, W. Fichtner, E. H. Grosse, D. J. Rose & R. K. Smith Transient simulation of silicon devices and circuits. IEEE Trans. Comput. Aided Des. CAD-4, 436-451, 1985.

## TrBDF2

For example, we consider a following system of time Ordinary Differential Equations (ODEs).
$$
\frac{du_{i}}{dt} = f_{i}(u,t)
$$
As its name states, the method advances the solution from $t_{n}$ to an intermediate time $t_{n+\gamma} = t_{n} + \gamma \Delta t$ by applying the Trapezoidal rule.
$$
u_{n+\gamma} - \gamma \frac{\Delta t}{2} f_{n+\gamma} = u_{n} + \gamma \frac{\Delta t}{2} f_{n}
$$
Next, it uses BDF2 to march the solution from $t_{n+\gamma}$ to $t_{n+1} = t_{n} + \Delta t$ as follows.
$$
u_{n+1} - \frac{1-\gamma}{2-\gamma} \Delta t f_{n+1} = \frac{1}{\gamma(2-\gamma)}u_{n+\gamma} - \frac{(1-\gamma)^2}{\gamma(2-\gamma)} u_{n}
$$
We solve the above non-linear equations iteratively using the Newton method. The Newton equation of the first Trapezoidal step is described:
$$
\left[] I - \gamma \frac{\Delta}{2} \left(\frac{\partial f}{\partial u}\right)^{(k)}\right]\delta u^{(k)} = -(u_{n+\gamma}^{(k)} - u_{n}) + \gamma \frac{\Delta t}{2}(f_{n+\gamma}^{(k)}+f_{n})
$$  
Then, the Newton equation of the BDF2 is described as follows.
$$
\left[I-\frac{1-\gamma}{2-\gamma} \Delta t \left(\frac{\partial f}{\partial u}\right)^{(k)}\right]\delta u^{(k)} =
-\left(u_{n+1}^{(k)} - \frac{1}{\gamma(2-\gamma)} u_{n+\gamma}+\frac{(1-\gamma)^2}{\gamma(2-\gamma)}u_{n}\right) + \frac{1-\gamma}{2-\gamma}\Delta t f_{n+1}^{(k)}
$$
Here, we denote a Jacobian as $J_{prob} = \frac{\partial f}{\partial u}$. The modified Jacobian's used for solving the Newton equations of the above Trapezoidal rule and the BDF2 are given as follows
$$
A_{tr} = I - \gamma \frac{\Delta t}{2} J_{prob} \qquad
A_{bdf} = I - \frac{1-\gamma}{2-\gamma}\Delta t J_{prob}
$$
while their right hand sides are defined as
$$
b_{tr} = -(u_{n+\gamma}^{(k)} - u_{n}) + \gamma \frac{\Delta t}{2}(f_{n+\gamma}^{(k)}+f_{n}) \\
b_{bdf} = -\left(u_{n+1}^{(k)} - \frac{1}{\gamma(2-\gamma)} u_{n+\gamma}+\frac{(1-\gamma)^2}{\gamma(2-\gamma)}u_{n}\right) + \frac{1-\gamma}{2-\gamma}\Delta t f_{n+1}^{(k)}
$$
In this way, a Newton solver can iteratively solves a problem $A(u) \delta u = b(u)$ with updating $u += \delta u$.

The timestep size $\Delta t$ can be adapted within a range $(\Delta t_{min}, \Delta t_{max})$ using a local error estimator.
$$
\text{error} \approx 2 k_{\gamma} \Delta t \left( \frac{1}{\gamma} f_{n} = \frac{1}{\gamma(1-\gamma)}f_{n+\gamma} + \frac{1}{1-\gamma} f_{n+1}\right) \quad \text{where} \quad  
k_{\gamma} = \frac{-3 \gamma^2 + 4 \gamma - 2}{12(2-\gamma)}
$$
This error is minimized when using a $\gamma = 2- \sqrt{2}$.


## Timestep Adaptivity

TChem uses weighted root-mean-square (WRMS) norms evaluating the estimated error. This approach is used in [Sundial package](https://computing.llnl.gov/sites/default/files/public/ida_guide.pdf). A weighting factor is computed as
$$
w_i = 1/\left( \text{rtol}_i | u_i | + \text{atol}_i \right)
$$
and the normalized error norm is computed as follows.
$$
\text{norm} = \left( \sum_i^m \left( \text{err}_i*w_i \right)^2 \right)/m
$$
This error norm close to 1 is considered as *small* and we increase the time step size and if the error norm is bigger than 10, the time step size decreases by half.

## Interface to Time Integrator

Our time integrator advance times for each sample independently in a parallel for. A namespace ``Impl`` is used to define a code interface for an individual sample.
```
TChem::Impl::TimeIntegrator::team_invoke_detail(
  /// kokkos team thread communicator
  const MemberType& member,
  /// abstract problem generator computing J_{prob} and f
  const ProblemType& problem,
  /// control parameters
  const ordinal_type& max_num_newton_iterations,
  const ordinal_type& max_num_time_iterations,
  /// absolute and relative tolerence size 2 array
  const RealType1DViewType& tol_newton,
  /// a vector of absolute and relative tolerence size Nspec x 2
  const RealType2DViewType& tol_time,
  /// \Delta t input, min, max
  const real_type& dt_in,
  const real_type& dt_min,
  const real_type& dt_max,
  /// time begin and end
  const real_type& t_beg,
  const real_type& t_end,
  /// input state vector at time begin
  const RealType1DViewType& vals,
  /// output for a restarting purpose: time, delta t, state vector
  const RealType0DViewType& t_out,
  const RealType0DViewType& dt_out,
  const RealType1DViewType& vals_out,
  const WorkViewType& work) {
  /// A pseudo code is illustrated here to describe the workflow

  /// This object is used to estimate the local errors
  TrBDF2<problem_type> trbdf2(problem);
  /// A_{tr} and b_{tr} are computed using the problem provided J_{prob} and f
  TrBDF2_Part1<problem_type> trbdf2_part1(problem);
  /// A_{bdf} and b_{bdf} are computed using the problem provided J_{prob} and f
  TrBDF2_Part2<problem_type> trbdf2_part2(problem);

  for (ordinal_type iter=0;iter<max_num_time_iterations && dt != zero;++iter) {
    /// evaluate function f_n
    problem.computeFunction(member, u_n, f_n);

    /// trbdf_part1 provides A_{tr} and b_{tr} solving A_{tr} du = b_{tr}
    /// and update u_gamma += du iteratively until it converges
    TChem::Impl::NewtonSolver(member, trbdf_part1, u_gamma, du);

    /// evaluate function f_gamma
    problem.computeFunction(member, u_gamma, f_gamma);

    /// trbdf_part2 provides A_{bdf} and b_{bdf} solving A_{bdf} du = b_{bdf}
    /// and update u_np += du iteratively until it converges
    TChem::Impl::NewtonSolver(member, trbdf_part2, u_np, du);

    /// evaluate function f_np
    problem.computeFunction(member, u_np, f_np);

    /// adjust time step
    trbdf2.computeTimeStepSize(member,
      dt_min, dt_max, tol_time, f_n, f_gamma, f_np, /// input for error evaluation
      dt); /// output

    /// account for the time end
    dt = ((t + dt) > t_end) ? t_end - t : dt;      
  }

  /// store current time step and state vectors for a restarting purpose
```  
This ``TimeIntegrator`` code requires for a user to provide a problem object. A problem class should include the following interface in order to be used with the time integrator.
```
template<typename KineticModelConstDataType>
struct MyProblem {
  ordinal_type getNumberOfTimeODEs();
  ordinal_type getNumberOfConstraints();
  /// the number of equations should be sum of number of time ODEs and number of constraints
  ordinal_type getNumberOfEquations();

  /// temporal workspace necessary for this problem class
  ordinal_type getWorkSpaceSize();

  /// x is initialized in the first Newton iteration
  void computeInitValues(const MemberType& member,
                         const RealType1DViewType& x) const;

  /// compute f(x)
  void computeFunction(const MemberType& member,
                       const RealType1DViewType& x,
                       const RealType1DViewType& f) const;

  /// compute J_{prob} at x                       
  void computeJacobian(const MemberType& member,
                       const RealType1DViewType& x,
                       const RealType2DViewType& J) const;
};
```
