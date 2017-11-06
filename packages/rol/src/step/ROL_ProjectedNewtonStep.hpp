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

#ifndef ROL_PROJECTEDNEWTONSTEP_H
#define ROL_PROJECTEDNEWTONSTEP_H

#include "ROL_Types.hpp"
#include "ROL_Step.hpp"

/** @ingroup step_group
    \class ROL::ProjectedNewtonStep
    \brief Provides the interface to compute optimization steps
           with projected Newton's method using line search.
*/

namespace ROL {

template <class Real>
class ProjectedNewtonStep : public Step<Real> {
private:

  std::shared_ptr<Vector<Real> > gp_; ///< Additional vector storage
  std::shared_ptr<Vector<Real> > d_;  ///< Additional vector storage
  int verbosity_;                  ///< Verbosity level
  const bool computeObj_;
  bool useProjectedGrad_;          ///< Whether or not to use to the projected gradient criticality measure

public:

  using Step<Real>::initialize;
  using Step<Real>::compute;
  using Step<Real>::update;

  /** \brief Constructor.

      Standard constructor to build a ProjectedNewtonStep object.  Algorithmic 
      specifications are passed in through a Teuchos::ParameterList.

      @param[in]     parlist    is a parameter list containing algorithmic specifications
  */
  ProjectedNewtonStep( Teuchos::ParameterList &parlist, const bool computeObj = true )
    : Step<Real>(), gp_(nullptr), d_(nullptr),
      verbosity_(0), computeObj_(computeObj), useProjectedGrad_(false) {
    // Parse ParameterList
    Teuchos::ParameterList& Glist = parlist.sublist("General");
    useProjectedGrad_ = Glist.get("Projected Gradient Criticality Measure", false);
    verbosity_ = parlist.sublist("General").get("Print Verbosity",0);
  }

  void initialize( Vector<Real> &x, const Vector<Real> &s, const Vector<Real> &g, 
                   Objective<Real> &obj, BoundConstraint<Real> &bnd, 
                   AlgorithmState<Real> &algo_state ) {
    Step<Real>::initialize(x,s,g,obj,bnd,algo_state);
    gp_ = g.clone();
    d_  = s.clone();
  }

  void compute( Vector<Real> &s, const Vector<Real> &x,
                Objective<Real> &obj, BoundConstraint<Real> &bnd,
                AlgorithmState<Real> &algo_state ) {
    Real tol = std::sqrt(ROL_EPSILON<Real>()), one(1);
    std::shared_ptr<StepState<Real> > step_state = Step<Real>::getState();

    // Compute projected Newton step
    // ---> Apply inactive-inactive block of inverse hessian to gradient
    gp_->set(*(step_state->gradientVec));
    bnd.pruneActive(*gp_,*(step_state->gradientVec),x,algo_state.gnorm);
    obj.invHessVec(s,*gp_,x,tol);
    bnd.pruneActive(s,*(step_state->gradientVec),x,algo_state.gnorm);
    // ---> Add in active gradient components
    gp_->set(*(step_state->gradientVec));
    bnd.pruneInactive(*gp_,*(step_state->gradientVec),x,algo_state.gnorm);
    s.plus(gp_->dual());
    s.scale(-one);
  }

  void update( Vector<Real> &x, const Vector<Real> &s,
               Objective<Real> &obj, BoundConstraint<Real> &bnd,
               AlgorithmState<Real> &algo_state ) {
    Real tol = std::sqrt(ROL_EPSILON<Real>()), one(1);
    std::shared_ptr<StepState<Real> > step_state = Step<Real>::getState();

    // Update iterate and store previous step
    algo_state.iter++;
    d_->set(x);
    x.plus(s);
    bnd.project(x);
    (step_state->descentVec)->set(x);
    (step_state->descentVec)->axpy(-one,*d_);
    algo_state.snorm = s.norm();

    // Compute new gradient
    obj.update(x,true,algo_state.iter);
    if ( computeObj_ ) {
      algo_state.value = obj.value(x,tol);
      algo_state.nfval++;
    }
    obj.gradient(*(step_state->gradientVec),x,tol);
    algo_state.ngrad++;

    // Update algorithm state
    (algo_state.iterateVec)->set(x);
    if ( useProjectedGrad_ ) {
      gp_->set(*(step_state->gradientVec));
      bnd.computeProjectedGradient( *gp_, x );
      algo_state.gnorm = gp_->norm();
    }
    else {
      d_->set(x);
      d_->axpy(-one,(step_state->gradientVec)->dual());
      bnd.project(*d_);
      d_->axpy(-one,x);
      algo_state.gnorm = d_->norm();
    }
  }

  std::string printHeader( void ) const {
    std::stringstream hist;

    if( verbosity_>0 ) {
      hist << std::string(109,'-') <<  "\n";
      hist << EDescentToString(DESCENT_NEWTON);
      hist << " status output definitions\n\n";
      hist << "  iter     - Number of iterates (steps taken) \n";
      hist << "  value    - Objective function value \n";
      hist << "  gnorm    - Norm of the gradient\n";
      hist << "  snorm    - Norm of the step (update to optimization vector)\n";
      hist << "  #fval    - Cumulative number of times the objective function was evaluated\n";
      hist << "  #grad    - Number of times the gradient was computed\n";
      hist << std::string(109,'-') << "\n";
    }

    hist << "  ";
    hist << std::setw(6)  << std::left << "iter";
    hist << std::setw(15) << std::left << "value";
    hist << std::setw(15) << std::left << "gnorm";
    hist << std::setw(15) << std::left << "snorm";
    hist << std::setw(10) << std::left << "#fval";
    hist << std::setw(10) << std::left << "#grad";
    hist << "\n";
    return hist.str();
  }
  std::string printName( void ) const {
    std::stringstream hist;
    hist << "\n" << EDescentToString(DESCENT_NEWTON) << "\n";
    return hist.str();
  }
  std::string print( AlgorithmState<Real> &algo_state, bool print_header = false ) const {
    std::stringstream hist;
    hist << std::scientific << std::setprecision(6);
    if ( algo_state.iter == 0 ) {
      hist << printName();
    }
    if ( print_header ) {
      hist << printHeader();
    }
    if ( algo_state.iter == 0 ) {
      hist << "  ";
      hist << std::setw(6) << std::left << algo_state.iter;
      hist << std::setw(15) << std::left << algo_state.value;
      hist << std::setw(15) << std::left << algo_state.gnorm;
      hist << "\n";
    }
    else {
      hist << "  ";
      hist << std::setw(6)  << std::left << algo_state.iter;
      hist << std::setw(15) << std::left << algo_state.value;
      hist << std::setw(15) << std::left << algo_state.gnorm;
      hist << std::setw(15) << std::left << algo_state.snorm;
      hist << std::setw(10) << std::left << algo_state.nfval;
      hist << std::setw(10) << std::left << algo_state.ngrad;
      hist << "\n";
    }
    return hist.str();
  }
}; // class ProjectedNewtonStep

} // namespace ROL

#endif
