/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2012-2023 by the DOpElib authors
 * Copyright (C) 2021-2023 by Manaswinee Bezbaruah, Matthias Maier, Winifried Wollner
 **/

#include <iostream>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/precondition_block.h>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <basic/mol_spacetimehandler.h>
#include <container/integratordatacontainer.h>
#include <container/optproblemcontainer.h>
#include <include/parameterreader.h>
#include <interfaces/functionalinterface.h>
#include <opt_algorithms/reducednewtonalgorithm.h>
#include <opt_algorithms/reducedtrustregionnewton.h>
#include <problemdata/noconstraints.h>
#include <problemdata/simpledirichletdata.h>
#include <reducedproblems/statreducedproblem.h>
#include <templates/cglinearsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/gmreslinearsolver.h>
#include <templates/integrator.h>
#include <templates/newtonsolver.h>
#include <wrapper/preconditioner_wrapper.h>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

#include "../Algorithms/reducedbfgsalgorithm.h"
#include "../Algorithms/reduceddampedbfgsalgorithm.h"
#include "../Algorithms/reducedgradientdescentalgorithm.h"
#include "geometry.h"
#include "localfunctional.h"
#include "localpde.h"
#include "periodicity_constraints.h"
#include "preconditioner.h"
#include "problem_description.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// TODO: Change Dimension here
static constexpr int DIM = 2;

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

#define CDC ElementDataContainer
#define FDC FaceDataContainer
#if DEAL_II_VERSION_GTE(9,3,0)
#define DOFHANDLER false
#else
#define DOFHANDLER DoFHandler
#endif

typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> COSTFUNCTIONAL;
typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> FUNCTIONALINTERFACE;

typedef OptProblemContainer<FUNCTIONALINTERFACE,
                            COSTFUNCTIONAL,
                            LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
                            SimpleDirichletData<VECTOR, DIM>,
                            NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
                            SPARSITYPATTERN,
                            VECTOR,
                            DIM,
                            DIM>
    OP;

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR, DIM> IDC;

typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;

// Linear Systems with UMFPACK
// typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
// GMRES with block LU decomposition
using PRECONDITIONER = Preconditioner<MATRIX>;
using LINEARSOLVER = GMRESLinearSolverWithMatrix<PRECONDITIONER, SPARSITYPATTERN, MATRIX, VECTOR>;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;

// Inverse BFGS Method (only first derivatives needed)
typedef ReducedBFGSAlgorithm<OP, VECTOR> RNA_BFGS;
// Inverse DampedBFGS Method (only first derivatives needed)
typedef ReducedDampedBFGSAlgorithm<OP, VECTOR> RNA_dBFGS;
// Gradient Descent (only first derivatives needed)
typedef ReducedGradientDescentAlgorithm<OP, VECTOR> RNA_GD;

// This class represents the reduced problem and steers the solution process.
typedef StatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, DIM, DIM> RP;

// The spacetimehandler manages all the things related to the degrees of
// freedom in space and time (for the optimization as well as the state variable!)
typedef MethodOfLines_SpaceTimeHandler<FESystem, DOFHANDLER, SPARSITYPATTERN, VECTOR, DIM, DIM> STH;

int main(int argc, char **argv)
{
  constexpr static unsigned int control_size = DIM;
  constexpr static unsigned int state_size = 2 * DIM;

  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv);

  /* Problem parameters: */

  ParameterAcceptor parameter_acceptor("algorithm setup");

  double radius = 0.3;
  unsigned int refinements = 5;
  parameter_acceptor.add_parameter("geometry radius", radius, "The radius of the interface");
  parameter_acceptor.add_parameter("geometry refinements", refinements, "Number of global refinements");

  std::string strategy = "GD";
  unsigned int niter = 2;
  std::string cases = "forward";
  parameter_acceptor.add_parameter("solver type", strategy, "Valid options are: 'GD', 'BFGS', 'dBFGS'");
  parameter_acceptor.add_parameter("solver niter", niter, "Number of iterations");
  parameter_acceptor.add_parameter("solver cases", cases, "Valid options are: 'forward', 'check', 'solve'");

  std::vector<double> inverse_scale_{0.1};
  std::vector<unsigned int> iterations_{100};
  parameter_acceptor.add_parameter("inverse scale", inverse_scale_, "Inverse scale for BFGS/dBFGS");
  parameter_acceptor.add_parameter("iterations", iterations_, "Number of iterations per step for GD/BFGS/dBFGS");

  /* DoPE parameters: */

  string paramfile = "dope.prm";

  if (argc == 2) {
    paramfile = argv[1];
  } else if (argc > 2) {
    std::cout << "Usage: " << argv[0] << " [ paramfile ] " << std::endl;
    return -1;
  }

  ParameterReader pr;
  RP::declare_params(pr);
  RNA_GD::declare_params(pr);
  RNA_BFGS::declare_params(pr);
  RNA_dBFGS::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<DIM> triangulation;

  // Set up Finite Elements
  FESystem<DIM> control_fe(FE_Q<DIM>(1), control_size);
  FESystem<DIM> state_fe(FE_Q<DIM>(1), state_size);

  const unsigned q_order = 3; // FIXME
  QUADRATURE quadrature_formula(q_order);
  FACEQUADRATURE face_quadrature_formula(q_order);
  IDC idc(quadrature_formula, face_quadrature_formula);

  ProblemDescription<DIM> PD;
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(PD);
  COSTFUNCTIONAL LFunc(PD);

  STH DOFH(triangulation, control_fe, state_fe, DOpEtypes::stationary, false);

  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> Constraints;

  OP P(LFunc, LPDE, Constraints, DOFH);

  /*
   * Now, read in all parameters:
   */

  ParameterAcceptor::initialize("cell_problem.prm");

  create_circle(triangulation, radius);
  triangulation.refine_global(refinements);

  /*
   * We need periodic boundary conditions for the state:
   */

  PeriodicityConstraints<DOFHANDLER, DIM> periodicity_constraints;
  DOFH.SetUserDefinedDoFConstraints(periodicity_constraints);

  /*
   * And Dirichlet boundary conditions for the control:
   */

  std::vector<bool> ccomp_mask(control_size, true);
  DOpEWrapper::ZeroFunction<2> czf(control_size);
  P.SetControlDirichletBoundaryColors(0, ccomp_mask, &czf);
  P.SetControlDirichletBoundaryColors(1, ccomp_mask, &czf);
  P.SetControlDirichletBoundaryColors(2, ccomp_mask, &czf);
  P.SetControlDirichletBoundaryColors(3, ccomp_mask, &czf);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  /* Output handling: */

  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);

  RNA_GD AlgGD(&P, &solver, pr, &ex, &out);
  RNA_BFGS AlgBFGS(&P, &solver, pr, &ex, &out);
  RNA_dBFGS AlgdBFGS(&P, &solver, pr, &ex, &out);
  AlgGD.ReInit();
  AlgBFGS.ReInit();
  AlgdBFGS.ReInit();

  out.ReInit();
  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem, pr);
  q = 0.;

  AssertThrow(iterations_.size() == niter, ExcMessage("oops, parameters do not match"));
  AssertThrow(inverse_scale_.size() == niter, ExcMessage("oops, parameters do not match"));

  for (unsigned int i = 0; i < niter; i++) {
    stringstream outp;
    outp << "*************************************************\n";
    outp << "*             Starting " << cases << "               \n";
    outp << "*   CDoFs   : ";
    q.PrintInfos(outp);
    outp << "*   SDoFs   : ";
    solver.StateSizeInfo(outp);
    outp << "*************************************************";
    // We print this header with priority 1, i.e., always, and 1 empty line in front and after.
    out.Write(outp, 1, 1, 1);

    //
    // Make sure we use the correct stabilization parameters for the ith
    // iteration:
    //

    AlgGD.get_nonlinear_maxiter() = iterations_[i];
    AlgBFGS.get_nonlinear_maxiter() = iterations_[i];
    AlgBFGS.get_init_inverse_scale_() = inverse_scale_[i];
    AlgdBFGS.get_nonlinear_maxiter() = iterations_[i];
    AlgdBFGS.get_init_inverse_scale_() = inverse_scale_[i];

    //
    // define a set_control lambda that is used for the "forward" problem
    // case when we debug computing the state equation.
    //
    auto set_control = [&](auto &q) {
      const auto &dof_handler =
      static_cast<const dealii::DoFHandler<DIM> &>(DOFH.GetControlDoFHandler());
      auto &control = q.GetSpacialVector();

      const auto callable = [](const dealii::Point<DIM> &x_hat) {
        dealii::Point<DIM> q_hat;
        q_hat[0] = 1.0 * x_hat[1] * (1. - x_hat[1]) * x_hat[0] * (1. - x_hat[0]);
        q_hat[1] = 2.0 * q_hat[0];
        return q_hat;
      };

      VectorTools::interpolate(dof_handler, to_function<DIM>(callable), control);
    };

    try {
      if (cases == "forward") {
        set_control(q);
        /* FIXME: remove this constraint... */
        AssertThrow(strategy == "GD", ExcMessage("case == forward requires strategy == GD"));
        AlgGD.SolveForward(q);

      } else if (cases == "check") {
        set_control(q);
        ControlVector<VECTOR> dq(q);
        set_control(dq);
        /* FIXME: remove this constraint... */
        AssertThrow(strategy == "GD", ExcMessage("case == forward requires strategy == GD"));
        AlgGD.CheckGrads(0., q, dq, 4, 0.01);

      } else if (cases == "solve") {
        try {
          if (strategy == "GD")
            AlgGD.Solve(q);
          else if (strategy == "BFGS")
            AlgBFGS.Solve(q);
          else if (strategy == "dBFGS")
            AlgdBFGS.Solve(q);
          else {
            std::cerr << "Wrong solver type: " << strategy << std::endl;
            abort();
          }
        } catch (...) {
          std::cout << "Warning: algorithm aborted with exception." << std::endl;
        }
      } else {
        std::cerr << "Wrong case: " << cases << std::endl;
        abort();
      }
    } catch (DOpEException &e) {
      std::cout << "Warning: During execution of `" + e.GetThrowingInstance() +
                       "` the following Problem occurred!"
                << std::endl;
      std::cout << e.GetErrorMessage() << std::endl;
    }

    if (i != niter - 1) {
      // use next set of stabilization parameters:
      LFunc.signal_next_iteration_by_incrementing_current_iteration_variable();

      out.ReInit();
      AlgGD.ReInit();
      AlgBFGS.ReInit();
      AlgdBFGS.ReInit();
    }
  }

  return 0;
}

#undef FDC
#undef CDC
