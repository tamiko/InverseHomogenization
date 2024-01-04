InverseHomogenization
=====================

The source file for the shape optimization cell problem. The source code is
organized in a similar manner than typical DOPeLib example steps [1]: The
PDE and cost function are described in
 - `localfunctional.h`
 - `localpde.h`
Various helper macros, functions and classes are found in `geometry.h`,
`helper.h`, `periodicity_constraints.h`, `preconditioner.h`,
`problem_description.h`.

Example configuration files are given in `cell_problem.prm.template`,
`cell_problem_table1.prm.template`, `dope.prm.template`.


How to compile
--------------

 - Obtain, compile and install deal.II [2] (deal.II is packaged and readily
   available in binary form for various Linux distributions).
 - Obtain, compile and install DOPeLib [1]. Export the library location via
   the environment variable `DOPE_DIR`.
 - Now, compile and install the `InverseHomogenization` program via
```
cmake .
make
```

[1] https://winnifried.github.io/dopelib/download.html

[2] https://www.dealii.org/
