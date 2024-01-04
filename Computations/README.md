Computations
============

Parameter files for the numerical computations summarized in Section 4.1,
4.2 and 4.3.


How to run
----------

 - First, compile the `InverseHomogenization` executable, see instructions
   in `../InverseHomogenization/README.md`

 - Change to a subdirectory and run the executable with something like
```
./InverseHomogenization | tee output.log
```
   The file `output.log` will contain the command line logs and the
   subdirectory `Results0` will contain prm files with the solutions for
   state and adjoint equation (configurable via `dope.prm`).
