
# Time Integration

For stiff ODEs/DAEs, the numerical time step size is limited by a stability condition rather than a truncation error. To obtain a reliable solution, we use a stable time integration method i.e., 2nd order Trapezoidal Backward Difference Formula (TrBDF2). The TrBDF2 scheme is a composite single step method. The method is 2-nd order accurate and $L$-stable. This ODE/DAE solver was implemented via Kokkos, it takes advantage of parallel threads available through the Kokkos interface, and it is part of the [Tines](https://github.com/sandialabs/Tines) library.

* R. E. Bank, W. M. Coughran, W. Fichtner, E. H. Grosse, D. J. Rose, and R. K. Smith, Transient simulation of silicon devices and circuits, IEEE Trans. Comput. Aided Des. CAD-4, 436-451, 1985.
