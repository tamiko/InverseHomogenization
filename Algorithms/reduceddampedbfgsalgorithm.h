/**
*
* Copyright (C) 2012-2023 by the DOpElib authors
*
* This file is part of DOpElib
*
* DOpElib is free software: you can redistribute it
* and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either
* version 3 of the License, or (at your option) any later
* version.
*
* DOpElib is distributed in the hope that it will be
* useful, but WITHOUT ANY WARRANTY; without even the implied
* warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE.  See the GNU General Public License for more
* details.
*
* Please refer to the file LICENSE.TXT included in this distribution
* for further information on this license.
*
**/

#ifndef REDUCEDDAMPEDBFGS__ALGORITHM_H_
#define REDUCEDDAMPEDBFGS__ALGORITHM_H_

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <opt_algorithms/reducedalgorithm.h>
#include <include/parameterreader.h>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

#include <iostream>
#include <assert.h>
#include <iomanip>
namespace DOpE
{
  /**
   * @class ReducedDampedBFGSAlgorithm
   *
   * This class provides a solver for equality constrained optimization
   * problems in reduced form, i.e., the dependent variable is
   * assumed to be eliminated by solving the equation. I.e.,
   * we solve the problem min j(q)
   *
   * The solution is done with a linesearch algorithm, see, e.g.,
   * Nocedal & Wright.
   *
   * @tparam <PROBLEM>    The problem container. See, e.g., OptProblemContainer
   * @tparam <VECTOR>     The vector type of the solution.
   */
  template <typename PROBLEM, typename VECTOR>
  class ReducedDampedBFGSAlgorithm : public ReducedAlgorithm<PROBLEM, VECTOR>
  {
  public:
    /**
     * The constructor for the algorithm
     *
     * @param OP              A pointer to the problem container
     * @param S               The reduced problem. This object handles the equality
     *                        constraint. For the interface see ReducedProblemInterface.
     * @param param_reader    A parameter reader to access user given runtime parameters.
     * @param Except          The DOpEExceptionHandler. This is used to handle the output
     *                        by all exception.
     * @param Output          The DOpEOutputHandler. This takes care of all output
     *                        generated by the problem.
     * @param base_priority   An offset for the priority of the output generated by the algorithm.
     */
    ReducedDampedBFGSAlgorithm(PROBLEM *OP,
                           ReducedProblemInterface<PROBLEM, VECTOR> *S,
                           ParameterReader &param_reader,
                           DOpEExceptionHandler<VECTOR> *Except=NULL,
                           DOpEOutputHandler<VECTOR> *Output=NULL,
                           int base_priority=0);
    virtual ~ReducedDampedBFGSAlgorithm();

    /**
     * Used to declare run time parameters. This is needed to declare all
     * parameters a startup without the need for an object to be already
     * declared.
     */
    static void declare_params(ParameterReader &param_reader);

    /**
     * This solves an Optimizationproblem in only the control variable
     * by a newtons method.
     *
     * @param q           The initial point.
     * @param global_tol  An optional parameter specifying the required  tolerance.
     *                    The actual tolerance is the maximum of this and the one specified in the param
     *                    file. Its default value is negative, so that it has no influence if not specified.
     */
    virtual int Solve(ControlVector<VECTOR> &q,double global_tol=-1.);
    /**
     * This returns the natural norm of the newton residual. This means the norm of the gradient of the
     * reduced cost functional.
     *
     * @param q           The initial point.
     */
    double BFGSResidual(const ControlVector<VECTOR> &q);

    virtual void ReInit()
    {
      ReducedAlgorithm<PROBLEM, VECTOR>::ReInit();
      for(unsigned int i = 0; i < memory_; i++)
      {
	if ( d_vals_[i] != NULL )
	{
	  delete d_vals_[i];
	  d_vals_[i] = NULL;
	}
	if ( By_vals_[i] != NULL )
	{
	  delete By_vals_[i];
	  By_vals_[i] = NULL;
	}
      }
    }

    unsigned int &get_nonlinear_maxiter() { return nonlinear_maxiter_; }

    double &get_init_inverse_scale_() { return init_inverse_scale_; }

  protected:
    /**
     * Performs an Powell-Wolfe-type linesearch to find a point of sufficient descent
     * for the functional j along the direction dq.
     *
     *
     * @param dq                    The search direction.
     * @param gradient              The l^2 gradient of the costfunctional at q,
     *                              i.e., the gradient_i = \delta_{q_i} j(q)
     *                              where q_i denotes the i-th DoF for the control.
     * @param gradient_new          The l^2 Gradient at the new point
     * @param gradient_new_transposed   The Control-space gradient at the new point
     * @param q                     The control. Needs to be the last evaluation point of
     *                              j in the begining and is at the end the updated
     *                              control q+\alpha dq.
     */
    virtual int ReducedBFGSLineSearch(ControlVector<VECTOR> &dq,
				      const ControlVector<VECTOR> &gradient,
				      const ControlVector<VECTOR> &gradient_transposed,
				      double &cost,
				      ControlVector<VECTOR> &q);
    /**
     * Calculates the action of the inverse BFGS-Matrix B applied to 'gradient'. The result is 
     * stored in dq.
     *
     * @param dq                   The vector storing B y
     * @param gradient             the l^2 representation of y, i.e., a dual vector
     * @param gradient_transposed  the control space representation of y
     * @param iter                 the current iteration index
     * @param iterstart            the index of the iteration to which the BFGS-Matrix belongs
     *                             when calling this method iter should be iterstart. It then 
     *                             recursively calls previous iterations of the method.
     */
    void ApplyBFGSMatrix(ControlVector<VECTOR> &dq,
			 const ControlVector<VECTOR> &gradient,
			 const ControlVector<VECTOR> &gradient_transposed,
                         unsigned int iter, unsigned int iterstart);
    /**
     * Stores the data for the next iteration.
     *
     * @param dq                   The current step d_k
     * @param By                   The Vector By (in control space representation)
     * @param y                    Dual representation of the vector y
     * @param iter                 the current iteration index
     */
    void Store(const ControlVector<VECTOR> &dq,
	       const ControlVector<VECTOR> &By,
	       const ControlVector<VECTOR> &y,
	       unsigned int iter);
    /**
     * Evaluates the squared residual, i.e., the scalar product gradient*gradient_transposed
     */
    virtual double Residual(const ControlVector<VECTOR> &gradient,
                            const ControlVector<VECTOR> &gradient_transposed)
    {
      return  gradient*gradient_transposed;
    }

    /**
     * Shows Diagnostic information on the iteration
     */
    void PrintDiagnostics(const ControlVector<VECTOR> &dq,
			  const ControlVector<VECTOR> &gradient,
			  const ControlVector<VECTOR> &gradient_transposed,
			  unsigned int iter,
			  double t_min)
    {
      //Diagnostics
      double angle = Residual(dq,gradient);
      double initial_steplength =  sqrt(Residual(dq,dq));
      angle /= sqrt(Residual(gradient,gradient_transposed));
      angle /= initial_steplength;
      std::stringstream out;
      out<<"\t\t BFGS linesearch: stopped after " <<iter<<" iterations\t step-length: "<<t_min<<"\t Angle: "<<angle<<"\t l2-length of dq: "<<initial_steplength<<"\n";
      this->GetOutputHandler()->Write(out,4+this->GetBasePriority());
    }

  private:
    unsigned int nonlinear_maxiter_, line_maxiter_, memory_;
    double       nonlinear_tol_, nonlinear_global_tol_, linesearch_gamma_, linesearch_beta_,
       init_inverse_scale_;
    bool         compute_functionals_in_every_step_;
    std::string postindex_;
    std::vector<double> first_factors_;
    std::vector<double> second_factors_;
    std::vector<ControlVector<VECTOR>* > d_vals_; 
    std::vector<ControlVector<VECTOR>* > By_vals_; 
  };

  /***************************************************************************************/
  /****************************************IMPLEMENTATION*********************************/
  /***************************************************************************************/
  using namespace dealii;

  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  void ReducedDampedBFGSAlgorithm<PROBLEM, VECTOR>::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("reducedDampedBFGSalgorithm parameters");
    param_reader.declare_entry("nonlinear_maxiter", "10",Patterns::Integer(0));
    param_reader.declare_entry("nonlinear_tol", "1.e-7",Patterns::Double(0));
    param_reader.declare_entry("nonlinear_global_tol", "1.e-11",Patterns::Double(0));

    param_reader.declare_entry("line_maxiter", "10",Patterns::Integer(0));
    param_reader.declare_entry("linesearch_gamma", "0.3",Patterns::Double(0,0.5));
    param_reader.declare_entry("linesearch_beta", "0.1",Patterns::Double(0));

    param_reader.declare_entry("memory", "10",Patterns::Integer(0));
    param_reader.declare_entry("init_inverse_scale", "1.",Patterns::Double(0));
    
    param_reader.declare_entry("compute_functionals_in_every_step", "false",Patterns::Bool());

    ReducedAlgorithm<PROBLEM, VECTOR>::declare_params(param_reader);
  }
  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  ReducedDampedBFGSAlgorithm<PROBLEM, VECTOR>::ReducedDampedBFGSAlgorithm(PROBLEM *OP,
      ReducedProblemInterface<PROBLEM, VECTOR> *S,
      ParameterReader &param_reader,
      DOpEExceptionHandler<VECTOR> *Except,
      DOpEOutputHandler<VECTOR> *Output,
      int base_priority)
    : ReducedAlgorithm<PROBLEM, VECTOR>(OP,S,param_reader,Except,Output,base_priority)
  {

    param_reader.SetSubsection("reducedDampedBFGSalgorithm parameters");
    nonlinear_maxiter_    = param_reader.get_integer ("nonlinear_maxiter");
    nonlinear_tol_        = param_reader.get_double ("nonlinear_tol");
    nonlinear_global_tol_ = param_reader.get_double ("nonlinear_global_tol");
    
    line_maxiter_         = param_reader.get_integer ("line_maxiter");
    linesearch_gamma_       = param_reader.get_double ("linesearch_gamma");
    linesearch_beta_       = param_reader.get_double ("linesearch_beta");

    assert(linesearch_gamma_ < 0.5);
    memory_    = param_reader.get_integer ("memory");
    init_inverse_scale_ = param_reader.get_double ("init_inverse_scale");
    
    compute_functionals_in_every_step_  = param_reader.get_bool ("compute_functionals_in_every_step");

    postindex_ = "_"+this->GetProblem()->GetName();
    
    first_factors_.resize(memory_);
    second_factors_.resize(memory_);
    d_vals_.resize(memory_,NULL);
    By_vals_.resize(memory_,NULL);
  }

  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  ReducedDampedBFGSAlgorithm<PROBLEM, VECTOR>::~ReducedDampedBFGSAlgorithm()
  {
    assert(d_vals_.size() == memory_);
    assert(By_vals_.size() == memory_);
    for(unsigned int i = 0; i < memory_; i++)
    {
      if ( d_vals_[i] != NULL )
	delete d_vals_[i];
      if ( By_vals_[i] != NULL )
	delete By_vals_[i];
    }
  }

  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  double ReducedDampedBFGSAlgorithm<PROBLEM, VECTOR>::BFGSResidual(const ControlVector<VECTOR> &q)
  {
    //Solve j'(q) = 0
    ControlVector<VECTOR> gradient(q), gradient_transposed(q);

    try
      {
        this->GetReducedProblem()->ComputeReducedCostFunctional(q);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"ReducedDampedBFGSAlgorithm::BFGSResidual");
      }

    try
      {
        this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"ReducedDampedBFGSAlgorithm::BFGSResidual");
      }

    return sqrt(Residual(gradient,gradient_transposed));
  }

  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  int ReducedDampedBFGSAlgorithm<PROBLEM, VECTOR>::Solve(ControlVector<VECTOR> &q,double global_tol)
  {

    q.ReInit();
    //Solve j'(q) = 0
    ControlVector<VECTOR> dq(q), B_y(q), gradient(q), gradient_transposed(q), gradient_new(q), gradient_new_transposed(q);
    ControlVector<VECTOR> y(q), y_t(q);

    unsigned int iter=0;
    double cost=0.;
    std::stringstream out;
    this->GetOutputHandler()->InitNewtonOut(out);

    out << "**************************************************\n";
    out << "*        Starting Reduced DampedBFGS Algorithm       *\n";
    out << "*   Solving : "<<this->GetProblem()->GetName()<<"\t*\n";
    out << "*  CDoFs : ";
    q.PrintInfos(out);
    out << "*  SDoFs : ";
    this->GetReducedProblem()->StateSizeInfo(out);
    out << "**************************************************";
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);

    this->GetOutputHandler()->SetIterationNumber(iter,"OptBFGS"+postindex_);

    this->GetOutputHandler()->Write(q,"Control"+postindex_,"control");

    try
      {
        cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"ReducedDampedBFGSAlgorithm::Solve");
      }

    out<< "CostFunctional: " << cost;
    this->GetOutputHandler()->Write(out,2+this->GetBasePriority());

    if (compute_functionals_in_every_step_ == true)
      {
        try
          {
            this->GetReducedProblem()->ComputeReducedFunctionals(q);
          }
        catch (DOpEException &e)
          {
            this->GetExceptionHandler()->HandleCriticalException(e);
          }
      }

    try
      {
        this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"ReducedDampedBFGSAlgorithm::Solve");
      }

    double res = Residual(gradient,gradient_transposed);//gradient*gradient_transposed;
    double firstres = res;

    assert(res >= 0);

    this->GetOutputHandler()->Write(gradient,"BFGSResidual"+postindex_,"control");
    out<< "\t BFGS step: " <<iter<<"\t Residual (abs.): "<<sqrt(res)<<"\n";
    out<< "\t BFGS step: " <<iter<<"\t Residual (rel.): "<<std::scientific<<sqrt(res)/sqrt(res)<<"\n";
    this->GetOutputHandler()->Write(out,3+this->GetBasePriority());
    int lineiter =0;
    unsigned int miniter = 0;
    if (global_tol > 0.)
      miniter = 1;

    global_tol =  std::max(nonlinear_global_tol_,global_tol);
    while (( (res >= global_tol*global_tol) && (res >= nonlinear_tol_*nonlinear_tol_*firstres) ) ||  iter < miniter )
      {
        iter++;
        this->GetOutputHandler()->SetIterationNumber(iter,"OptBFGS"+postindex_);

        if (iter > nonlinear_maxiter_)
          {
            throw DOpEIterationException("Iteration count exceeded bounds!","ReducedDampedBFGSAlgorithm::Solve");
          }

        //Compute a search direction
	//Use the correct gradient! (here called _transposed) The usual l^2 representation
	//is typically very bad!

	ApplyBFGSMatrix(dq,gradient,gradient_transposed, iter-1, iter-1);
	dq *= -1.;
        //Linesearch
        try
          {
	    //Check if dq is a descent direction
	    double reduction = gradient*dq;
	    if(reduction > 0)
	    {
	      this->GetOutputHandler()->WriteError("Waring: computed direction doesn't seem to be a descend direction! Trying negative gradient instead.");
	      dq = gradient_transposed;
	      dq *= -1.;
	    }
            lineiter = ReducedBFGSLineSearch(dq,gradient,gradient_transposed,cost,q);
          }
        catch (DOpEIterationException &e)
          {
            //Seems uncritical too many line search steps, it'll probably work
            //So only write a warning, and continue.
            this->GetExceptionHandler()->HandleException(e,"ReducedDampedBFGSAlgorithm::Solve");
            lineiter = -1;
          }
//        catch(DOpEException& e)
//        {
//          this->GetExceptionHandler()->HandleCriticalException(e);
//        }

        out<< "CostFunctional: " << cost;
        this->GetOutputHandler()->Write(out,3+this->GetBasePriority());

        if (compute_functionals_in_every_step_ == true)
	{
	  try
	  {
	    this->GetReducedProblem()->ComputeReducedFunctionals(q);
	  }
	  catch (DOpEException &e)
	  {
	    this->GetExceptionHandler()->HandleCriticalException(e);
	  }
	}
	try
	{
	  this->GetReducedProblem()->ComputeReducedGradient(q,gradient_new,gradient_new_transposed);
	}
	catch (DOpEException &e)
	{
	  this->GetExceptionHandler()->HandleCriticalException(e,"ReducedDampedBFGSAlgorithm::BFGSResidual");
	}

	res = Residual(gradient_new,gradient_new_transposed);
        //Prepare the next Iteration
	if( (res >= global_tol*global_tol) && (res >= nonlinear_tol_*nonlinear_tol_*firstres) )
	{
	  //#################################################################
	  y = gradient_new;
	  y.add(-1.,gradient);
	  y_t = gradient_new_transposed;
	  y_t.add(-1.,gradient_transposed);
	  ApplyBFGSMatrix(B_y,y,y_t, iter-1, iter-1);
	  
	  //#################################################################
	  // +++++ Check if d_modified is needed +++++
	  // Check if <d,y> > 0, where y is gradient_transposed_new - gradient_transposed and d is dq, calculated in the linesearch
	  // gradient_new and gradient_transposed_new are calculated in the linesearch, as well as it is not necessary to solve it
	  // for the Armijo condition

	  double y_By = y*B_y;
          if (y_By < 0.)
          {
              this->GetOutputHandler()->WriteError("Warning: Resetting y_By!!!  ");
              y_By = 0.;
          }
	  
	  double gradient_reduction = y*dq;
	  if(gradient_reduction <= 0.2*y_By){ 
	    double theta = 0.8*y_By/(y_By-gradient_reduction);
	    
	    std::cout << "theta = " << theta << std::endl;
	    //dq = theta * dq + (1-theta)*B_y;
	    dq *= theta;
	    dq.add((1-theta),B_y);
	    
	    std::cout <<"gradient_reduction with dq " << gradient_reduction << std::endl;
	    gradient_reduction = (y*dq);
	    std::cout <<"gradient_reduction with dq_mod " << gradient_reduction << std::endl;
	    assert(gradient_reduction > 0.);
	  }

	 //#################################################################
	 	  
	  Store(dq,B_y,y,iter);
	}
	gradient = gradient_new;
	gradient_transposed = gradient_new_transposed;
	this->GetOutputHandler()->Write(q,"Control"+postindex_,"control");
	this->GetOutputHandler()->Write(gradient,"BFGSResidual"+postindex_,"control");

        out<<"\t BFGS step: " <<iter<<"\t Residual (rel.): "<<this->GetOutputHandler()->ZeroTolerance(sqrt(res)/sqrt(firstres),1.0)<< "\t LineSearch {"<<lineiter<<"} ";
        this->GetOutputHandler()->Write(out,3+this->GetBasePriority());
      }



    //We are done write total evaluation
    out<< "CostFunctional: " << cost;
    this->GetOutputHandler()->Write(out,2+this->GetBasePriority());
    try
      {
        this->GetReducedProblem()->ComputeReducedFunctionals(q);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"ReducedDampedBFGSAlgorithm::Solve");
      }

    out << "**************************************************\n";
    out << "*        Stopping Reduced BFGS Algorithm       *\n";
    out << "*             after "<<std::setw(6)<<iter<<"  Iterations           *\n";
    out.precision(4);
    out << "*             with rel. Residual "<<std::scientific << std::setw(11) << this->GetOutputHandler()->ZeroTolerance(sqrt(res)/sqrt(firstres),1.0)<<"          *\n";
    out.precision(10);
    out << "**************************************************";
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);
    return iter;
  }


  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  int ReducedDampedBFGSAlgorithm<PROBLEM, VECTOR>::ReducedBFGSLineSearch(ControlVector<VECTOR> &dq,
									 const ControlVector<VECTOR>  &gradient,
									 const ControlVector<VECTOR>  &gradient_transposed,
									 double &cost,
									 ControlVector<VECTOR> &q)
  {
    double gamma = linesearch_gamma_;
    double beta = linesearch_beta_;

    double costnew = 0.;
    bool force_linesearch=false;
    unsigned int iter =0;
    
    double reduction = gradient*dq;
    if (reduction > 0)
    {
      this->GetOutputHandler()->WriteError("Waring: computed direction doesn't seem to be a descend direction!");
      reduction = 0;
      throw DOpEException("Direction does not seem to be a descent direction!","ReducedDampedBFGSAlgorithm::ReducedBFGSLineSearch");
    }

    if (fabs(reduction) < 1.e-10*cost)
    {
      this->GetOutputHandler()->WriteError("Warning: Predicted reduction is extremely small! ");
      double gradient_reduction = -1.*(gradient*gradient_transposed);
      if(gradient_reduction > 1.e-10*cost)
      {
	this->GetOutputHandler()->WriteError("Warning: Trying negative gradient instead.");
	dq = gradient_transposed;
	dq *=-1.;
	reduction = gradient_reduction;
      }
    }

    //Search for Armijo Steplength
    q+=dq;
    try
      {
        costnew = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
      }
    catch (DOpEException &e)
      {
//    this->GetExceptionHandler()->HandleException(e);
        force_linesearch = true;
        this->GetOutputHandler()->Write("Computing Cost Failed",4+this->GetBasePriority());
      }

    double alpha=1;

    if (line_maxiter_ > 0)
      {
        if (fabs(reduction) < 1.e-10*cost)
          reduction = 0.;
        if (std::isinf(costnew) || std::isnan(costnew) || (costnew >= cost + gamma*alpha*reduction) || force_linesearch)
          {
            this->GetOutputHandler()->Write("\t linesearch ",4+this->GetBasePriority());
            while (std::isinf(costnew) || std::isnan(costnew) || (costnew >= cost + gamma*alpha*reduction) || force_linesearch)
              {
                iter++;
                if (iter > line_maxiter_)
                  {
                    if (force_linesearch)
                      {
                        throw DOpEException("Iteration count exceeded bounds while unable to compute the CostFunctional!","ReducedNewtonAlgorithm::ReducedNewtonLineSearch");
                      }
                    else
                      {
                        cost = costnew;
                        throw DOpEIterationException("Iteration count exceeded bounds!","ReducedNewtonAlgorithm::ReducedNewtonLineSearch");
                      }
                  }
                force_linesearch = false;
                q.add(alpha*(beta-1.),dq);
                alpha *= beta;

                try
                  {
                    costnew = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
                  }
                catch (DOpEException &e)
                  {
                    //this->GetExceptionHandler()->HandleException(e);
                    force_linesearch = true;
                    this->GetOutputHandler()->Write("Computing Cost Failed",4+this->GetBasePriority());
                  }
              }
          }
        cost = costnew;
      }
    dq*=alpha;
    PrintDiagnostics(dq,gradient,gradient_transposed,iter,alpha);
    return iter;

  }

  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  void ReducedDampedBFGSAlgorithm<PROBLEM, VECTOR>::ApplyBFGSMatrix(ControlVector<VECTOR> &dq,
							      const ControlVector<VECTOR> &gradient,
							      const ControlVector<VECTOR> &gradient_transposed,
							      unsigned int iter,
							      unsigned int iterstart)
  {
    assert(iterstart >= iter);
    if( iter == 0 || iterstart == iter + memory_)
    {
      dq = gradient_transposed;
      dq *= 1./init_inverse_scale_;
    }
    else
    {
      ApplyBFGSMatrix(dq,gradient,gradient_transposed,iter-1,iterstart);
      unsigned int index = iter%memory_;
      double dy= *(d_vals_[index])*gradient;
      double Byy = *(By_vals_[index])*gradient;
      double d_factor = (2.*dy-Byy)*first_factors_[index]+dy*second_factors_[index];
      dq.add(d_factor,*(d_vals_[index]));
      double By_factor = -dy*first_factors_[index];
      dq.add(By_factor,*(By_vals_[index]));
    }
  }
  
  template <typename PROBLEM, typename VECTOR>
  void ReducedDampedBFGSAlgorithm<PROBLEM, VECTOR>::Store(const ControlVector<VECTOR> &dq,
						    const ControlVector<VECTOR> &By,
						    const ControlVector<VECTOR> &y,
						    unsigned int iter)
  {
    unsigned int index = iter%memory_;
    //d is not d_k!
    double d = dq*y;
    if(std::isinf(1./d) || std::isnan(1./d))
    {
      this->GetOutputHandler()->WriteError("Warning: BFGS Quotient '1/(d,y)_Q' is undefined!");
      throw DOpEException("Can't store BFGS-Update as quotients vanish!","ReducedBFGSAlgorithm::Store");
    }
    if(std::isinf(1./(d*d)) || std::isnan(1./(d*d)))
    {
      this->GetOutputHandler()->WriteError("Warning: BFGS Quotient '1/(d,y)_Q^2' is undefined!");
      throw DOpEException("Can't store BFGS-Update as quotients vanish!","ReducedBFGSAlgorithm::Store");
    }
    if( d < 0 )
    {
      this->GetOutputHandler()->WriteError("Warning: Wrong sign in <y,d> in BFGS Update!");
      throw DOpEException("Wrong sign in <y,d> in BFGS Update; Aborting!","ReducedBFGSAlgorithm::Store");
    }

    first_factors_[index] = 1./d;
    second_factors_[index] = -1./(d*d) *(d-By*y);
    if(d_vals_[index] == NULL)
    {
      d_vals_[index] = new ControlVector<VECTOR>(dq);
    }
    d_vals_[index]->equ(1.,dq);
    if(By_vals_[index] == NULL)
    {
      By_vals_[index] = new ControlVector<VECTOR>(By);
    }
    By_vals_[index]->equ(1.,By);
  }
  
}
#endif