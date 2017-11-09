// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef ROL_INTERIORPOINTSTEP_H
#define ROL_INTERIORPOINTSTEP_H

#include "ROL_CompositeStep.hpp"
#include "ROL_ConstraintStatusTest.hpp"
#include "ROL_InteriorPoint.hpp"
#include "ROL_ObjectiveFromBoundConstraint.hpp"
#include "ROL_Types.hpp"
#include "ROL_Constraint_Partitioned.hpp"


namespace ROL {

template <class Real>
class InteriorPointStep : public Step<Real> {

typedef InteriorPoint::PenalizedObjective<Real> IPOBJ;
typedef Constraint_Partitioned<Real>            IPCON;

private:

  std::shared_ptr<StatusTest<Real> >       status_;
  std::shared_ptr<Step<Real> >             step_;
  std::shared_ptr<IPOBJ>                   ipobj_;
  std::shared_ptr<IPCON>                   ipcon_;
  std::shared_ptr<Algorithm<Real> >        algo_;
  std::shared_ptr<Teuchos::ParameterList>  parlist_;
  std::shared_ptr<BoundConstraint<Real> >  bnd_;

  // Storage
  std::shared_ptr<Vector<Real> > x_;
  std::shared_ptr<Vector<Real> > g_;
  std::shared_ptr<Vector<Real> > l_;
  std::shared_ptr<Vector<Real> > c_;

  Real mu_;      // Barrier parameter
  Real mumin_;   // Minimal value of barrier parameter
  Real mumax_;   // Maximal value of barrier parameter
  Real rho_;     // Barrier parameter reduction factor
  int  maxit_;   // Maximum number of interior point subproblem solves

  // For the subproblem
  Real gtol_;           // Status test gradient tolerance
  Real ctol_;           // Status test constraint tolerance
  Real stol_;           // Status test step tolerance
  int subproblemIter_;  // Status test maximum number of iterations

  int verbosity_;       // Adjust level of detail in printing step information

  bool hasEquality_;

public:

  using Step<Real>::initialize;
  using Step<Real>::compute;
  using Step<Real>::update;

  ~InteriorPointStep() {}

  InteriorPointStep(Teuchos::ParameterList &parlist) :
    Step<Real>(),
    status_(nullptr),
    step_(nullptr),
    ipobj_(nullptr),
    ipcon_(nullptr),
    algo_(nullptr),
    x_(nullptr),
    g_(nullptr),
    l_(nullptr),
    c_(nullptr),
    hasEquality_(false) {

    using Teuchos::ParameterList;

    verbosity_ = parlist.sublist("General").get("Print Verbosity",0);

    // List of general Interior Point parameters
    ParameterList& iplist  = parlist.sublist("Step").sublist("Interior Point");

    mu_             = iplist.get("Initial Barrier Penalty",1.0);
    mumin_          = iplist.get("Minimum Barrier Penalty",1.e-4);
    mumax_          = iplist.get("Maximum Barrier Penalty",1e8);
    rho_            = iplist.get("Barrier Penalty Reduction Factor",0.5);
    subproblemIter_ = iplist.get("Subproblem Iteration Limit",10);


    // List of Status Test parameters
    ParameterList& stlist  = parlist.sublist("Status Test");

    gtol_  = stlist.get("Gradient Tolerance", 1.e-8);
    ctol_  = stlist.get("Constraint Tolerance", 1.e-8);
    stol_  = stlist.get("Step Tolerance", 1.e-8);
    maxit_ = stlist.get("Iteration Limit", 100);

    parlist_.reset(&parlist, [](Teuchos::ParameterList*){});
  }

  /** \brief Initialize step with equality constraint
   */
  void initialize( Vector<Real> &x, const Vector<Real> &g,
                   Vector<Real> &l, const Vector<Real> &c,
                   Objective<Real> &obj, Constraint<Real> &con,
                   AlgorithmState<Real> &algo_state ) {
    hasEquality_ = true;

    std::shared_ptr<StepState<Real> > state = Step<Real>::getState();
    state->descentVec    = x.clone();
    state->gradientVec   = g.clone();
    state->constraintVec = c.clone();

    // Initialize storage
    x_ = x.clone();
    g_ = g.clone();
    l_ = l.clone();
    c_ = c.clone();

    x_->set(x);

    ipobj_.reset(&dynamic_cast<IPOBJ&>(obj), [](IPOBJ*){});
    ipcon_.reset(&dynamic_cast<IPCON&>(con), [](IPCON*){});

    // Set initial penalty
    ipobj_->updatePenalty(mu_);

    algo_state.nfval = 0;
    algo_state.ncval = 0;
    algo_state.ngrad = 0;

    Real zerotol = 0.0;
    obj.update(x,true,algo_state.iter);
    algo_state.value = obj.value(x,zerotol);

    obj.gradient(*g_,x,zerotol);
    algo_state.gnorm = g_->norm();

    con.value(*c_,x,zerotol);
    algo_state.cnorm = c_->norm();

    algo_state.nfval += ipobj_->getNumberFunctionEvaluations();
    algo_state.ngrad += ipobj_->getNumberGradientEvaluations();
    algo_state.ncval += ipcon_->getNumberConstraintEvaluations();

  }



  void initialize( Vector<Real> &x, const Vector<Real> &g, Vector<Real> &l, const Vector<Real> &c,
                   Objective<Real> &obj, Constraint<Real> &con, BoundConstraint<Real> &bnd,
                   AlgorithmState<Real> &algo_state ) {
    bnd.projectInterior(x);
    initialize(x,g,l,c,obj,con,algo_state);
  }


  /** \brief Initialize step with no equality constraint
   */
  void initialize( Vector<Real> &x, const Vector<Real> &g,
                   Objective<Real> &obj, BoundConstraint<Real> &bnd,
                   AlgorithmState<Real> &algo_state ) {
    bnd.projectInterior(x);

    std::shared_ptr<StepState<Real> > state = Step<Real>::getState();
    state->descentVec    = x.clone();
    state->gradientVec   = g.clone();

    // Initialize storage
    x_ = x.clone(); x_->set(x);
    g_ = g.clone();

    // Set initial penalty
    ipobj_.reset(&dynamic_cast<IPOBJ&>(obj), [](IPOBJ*){});
    ipobj_->updatePenalty(mu_);

    algo_state.nfval = 0;
    algo_state.ncval = 0;
    algo_state.ngrad = 0;

    Real zerotol = std::sqrt(ROL_EPSILON<Real>());
    obj.update(x,true,algo_state.iter);
    algo_state.value = obj.value(x,zerotol);

    obj.gradient(*g_,x,zerotol);
    algo_state.gnorm = g_->norm();

    algo_state.cnorm = static_cast<Real>(0);

    algo_state.nfval += ipobj_->getNumberFunctionEvaluations();
    algo_state.ngrad += ipobj_->getNumberGradientEvaluations();

    bnd_ = std::make_shared<BoundConstraint<Real>>();
    bnd_->deactivate();
  }



  /** \brief Compute step (equality constraints).
  */
  void compute( Vector<Real>         &s,
                const Vector<Real>   &x,
                const Vector<Real>   &l,
                Objective<Real>      &obj,
                Constraint<Real>     &con,
                AlgorithmState<Real> &algo_state ) {
    // Grab interior point objective and constraint
    ipobj_.reset(&dynamic_cast<IPOBJ&>(obj), [](IPOBJ*){});
    ipcon_.reset(&dynamic_cast<IPCON&>(con), [](IPCON*){});

    // Create the algorithm
    algo_ = std::make_shared<Algorithm<Real>>("Composite Step",*parlist_,false);

    //  Run the algorithm
    x_->set(x);
    algo_->run(*x_,*g_,*l_,*c_,*ipobj_,*ipcon_,false);
    s.set(*x_); s.axpy(-1.0,x);

    // Get number of iterations from the subproblem solve
    subproblemIter_ = (algo_->getState())->iter;
  }

  void compute( Vector<Real>          &s,
                const Vector<Real>    &x,
                const Vector<Real>    &l,
                Objective<Real>       &obj,
                Constraint<Real>      &con,
                BoundConstraint<Real> &bnd,
                AlgorithmState<Real>  &algo_state ) {
    compute(s,x,l,obj,con,algo_state);
  }

  // Bound constrained
  void compute( Vector<Real>          &s,
                const Vector<Real>    &x,
                Objective<Real>       &obj,
                BoundConstraint<Real> &bnd,
                AlgorithmState<Real>  &algo_state ) {
    // Grab interior point objective and constraint
    ipobj_.reset(&dynamic_cast<IPOBJ&>(obj), [](IPOBJ*){});

    // Create the algorithm
    algo_ = std::make_shared<Algorithm<Real>>("Trust Region",*parlist_,false);

    //  Run the algorithm
    x_->set(x);
    algo_->run(*x_,*g_,*ipobj_,*bnd_,false);
    s.set(*x_); s.axpy(-1.0,x);

    // Get number of iterations from the subproblem solve
    subproblemIter_ = (algo_->getState())->iter;
  }



  /** \brief Update step, if successful (equality constraints).
  */
  void update( Vector<Real>         &x,
               Vector<Real>         &l,
               const Vector<Real>   &s,
               Objective<Real>      &obj,
               Constraint<Real>     &con,
               AlgorithmState<Real> &algo_state ) {
    // Grab interior point objective and constraint
    ipobj_.reset(&dynamic_cast<IPOBJ&>(obj), [](IPOBJ*){});
    ipcon_.reset(&dynamic_cast<IPCON&>(con), [](IPCON*){});

    // If we can change the barrier parameter, do so
    if( (rho_< 1.0 && mu_ > mumin_) || (rho_ > 1.0 && mu_ < mumax_) ) {
      mu_ *= rho_;
      ipobj_->updatePenalty(mu_);
    }

    std::shared_ptr<StepState<Real> > state = Step<Real>::getState();

    // Update optimization vector
    x.plus(s);

    algo_state.iterateVec->set(x);
    state->descentVec->set(s);
    algo_state.snorm = s.norm();
    algo_state.iter++;

    Real zerotol = 0.0;

    algo_state.value = ipobj_->value(x,zerotol);
    algo_state.value = ipobj_->getObjectiveValue();

    ipcon_->value(*c_,x,zerotol);
    state->constraintVec->set(*c_);

    ipobj_->gradient(*g_,x,zerotol);
    state->gradientVec->set(*g_);

    ipcon_->applyAdjointJacobian(*g_,*l_,x,zerotol);
    state->gradientVec->plus(*g_);

    algo_state.gnorm = g_->norm();
    algo_state.cnorm = state->constraintVec->norm();
    algo_state.snorm = s.norm();

    algo_state.nfval += ipobj_->getNumberFunctionEvaluations();
    algo_state.ngrad += ipobj_->getNumberGradientEvaluations();
    algo_state.ncval += ipcon_->getNumberConstraintEvaluations();

  }

  void update( Vector<Real>          &x,
               Vector<Real>          &l,
               const Vector<Real>    &s,
               Objective<Real>       &obj,
               Constraint<Real>      &con,
               BoundConstraint<Real> &bnd,
               AlgorithmState<Real>  &algo_state ) {
    update(x,l,s,obj,con,algo_state);

    std::shared_ptr<StepState<Real> > state = Step<Real>::getState();
    x_->set(x);
    x_->axpy(static_cast<Real>(-1),state->gradientVec->dual());
    bnd.project(*x_);
    x_->axpy(static_cast<Real>(-1),x);
    algo_state.gnorm = x_->norm();
  }

  void update( Vector<Real>          &x,
               const Vector<Real>    &s,
               Objective<Real>       &obj,
               BoundConstraint<Real> &bnd,
               AlgorithmState<Real>  &algo_state ) {
    // Grab interior point objective
    ipobj_.reset(&dynamic_cast<IPOBJ&>(obj), [](IPOBJ*){});

    // If we can change the barrier parameter, do so
    if( (rho_< 1.0 && mu_ > mumin_) || (rho_ > 1.0 && mu_ < mumax_) ) {
      mu_ *= rho_;
      ipobj_->updatePenalty(mu_);
    }

    std::shared_ptr<StepState<Real> > state = Step<Real>::getState();

    // Update optimization vector
    x.plus(s);

    algo_state.iterateVec->set(x);
    state->descentVec->set(s);
    algo_state.snorm = s.norm();
    algo_state.iter++;

    Real zerotol = std::sqrt(ROL_EPSILON<Real>());

    algo_state.value = ipobj_->value(x,zerotol);
    algo_state.value = ipobj_->getObjectiveValue();

    ipobj_->gradient(*g_,x,zerotol);
    state->gradientVec->set(*g_);

    x_->set(x);
    x_->axpy(static_cast<Real>(-1),state->gradientVec->dual());
    bnd.project(*x_);
    x_->axpy(static_cast<Real>(-1),x);

    algo_state.gnorm = x_->norm();
    algo_state.snorm = s.norm();

    algo_state.nfval += ipobj_->getNumberFunctionEvaluations();
    algo_state.ngrad += ipobj_->getNumberGradientEvaluations();
  }

  /** \brief Print iterate header.
  */
  std::string printHeader( void ) const {
    std::stringstream hist;

    if( verbosity_ > 0 ) {

      hist << std::string(116,'-') << "\n";
      hist << "Interior Point status output definitions\n\n";

      hist << "  IPiter  - Number of interior point steps taken\n";
      hist << "  SPiter  - Number of subproblem solver iterations\n";
      hist << "  penalty - Penalty parameter multiplying the barrier objective\n";
      hist << "  fval    - Number of objective evaluations\n";
      if (hasEquality_) {
        hist << "  cnorm   - Norm of the composite constraint\n";
        hist << "  gLnorm  - Norm of the Lagrangian's gradient\n";
      }
      else {
        hist << "  gnorm   - Norm of the projected norm of the objective gradient\n";
      }
      hist << "  snorm   - Norm of step (update to optimzation and slack vector)\n";
      hist << "  #fval   - Number of objective function evaluations\n";
      hist << "  #grad   - Number of gradient evaluations\n";
      if (hasEquality_) {
        hist << "  #cval   - Number of composite constraint evaluations\n";
      }
      hist << std::string(116,'-') << "\n";
    }

    hist << "  ";
    hist << std::setw(9)  << std::left  << "IPiter";
    hist << std::setw(9)  << std::left  << "SPiter";
    hist << std::setw(15) << std::left  << "penalty";
    hist << std::setw(15) << std::left  << "fval";
    if (hasEquality_) {
      hist << std::setw(15) << std::left  << "cnorm";
      hist << std::setw(15) << std::left  << "gLnorm";
    }
    else {
      hist << std::setw(15) << std::left  << "gnorm";
    }
    hist << std::setw(15) << std::left  << "snorm";
    hist << std::setw(8)  << std::left  << "#fval";
    hist << std::setw(8)  << std::left  << "#grad";
    if (hasEquality_) {
      hist << std::setw(8)  << std::left  << "#cval";
    }

    hist << "\n";
    return hist.str();
  }

  /** \brief Print step name.
  */
  std::string printName( void ) const {
    std::stringstream hist;
    hist << "\n" << "Primal Interior Point Solver\n";
    return hist.str();
  }

  /** \brief Print iterate status.
  */
  std::string print( AlgorithmState<Real> &algo_state, bool pHeader = false ) const {
    std::stringstream hist;
    hist << std::scientific << std::setprecision(6);
    if ( algo_state.iter == 0 ) {
      hist << printName();
    }
    if ( pHeader ) {
      hist << printHeader();
    }
    if ( algo_state.iter == 0 ) {
      hist << "  ";
      hist << std::setw(9)  << std::left << algo_state.iter;
      hist << std::setw(9)  << std::left << subproblemIter_;
      hist << std::setw(15) << std::left << mu_;
      hist << std::setw(15) << std::left << algo_state.value;
      if (hasEquality_) {
        hist << std::setw(15) << std::left << algo_state.cnorm;
      }
      hist << std::setw(15) << std::left << algo_state.gnorm;
      hist << "\n";
    }
    else {
      hist << "  ";
      hist << std::setw(9)  << std::left << algo_state.iter;
      hist << std::setw(9)  << std::left << subproblemIter_;
      hist << std::setw(15) << std::left << mu_;
      hist << std::setw(15) << std::left << algo_state.value;
      if (hasEquality_) {
        hist << std::setw(15) << std::left << algo_state.cnorm;
      }
      hist << std::setw(15) << std::left << algo_state.gnorm;
      hist << std::setw(15) << std::left << algo_state.snorm;
//      hist << std::scientific << std::setprecision(6);
      hist << std::setw(8) << std::left << algo_state.nfval;
      hist << std::setw(8) << std::left << algo_state.ngrad;
      if (hasEquality_) {
        hist << std::setw(8) << std::left << algo_state.ncval;
      }
      hist << "\n";
    }
    return hist.str();
  }

}; // class InteriorPointStep

} // namespace ROL

#endif // ROL_INTERIORPOINTSTEP_H
