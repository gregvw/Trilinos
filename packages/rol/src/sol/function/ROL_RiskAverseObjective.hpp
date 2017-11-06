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

#ifndef ROL_RISKAVERSEOBJECTIVE_HPP
#define ROL_RISKAVERSEOBJECTIVE_HPP

#include <memory>
#include "ROL_Vector.hpp"
#include "ROL_Objective.hpp"
#include "ROL_SampleGenerator.hpp"
#include "ROL_RiskMeasureFactory.hpp"
#include "ROL_ConvexCombinationRiskMeasure.hpp"

namespace ROL {

template<class Real>
class RiskAverseObjective : public Objective<Real> {
private:
  // Objective function definition
  std::shared_ptr<Objective<Real> >             ParametrizedObjective_; // Parametrized objective function
  std::shared_ptr<RiskMeasure<Real> >           RiskMeasure_;           // Risk measure

  // Sampler generators
  std::shared_ptr<SampleGenerator<Real> >       ValueSampler_;          // Sampler for objective value
  std::shared_ptr<SampleGenerator<Real> >       GradientSampler_;       // Sampler for objective gradient
  std::shared_ptr<SampleGenerator<Real> >       HessianSampler_;        // Sampler for objective Hessian-times-a-vector

  // Additional storage
  bool firstUpdate_;
  bool storage_;
  std::map<std::vector<Real>,Real>                         value_storage_;
  std::map<std::vector<Real>,std::shared_ptr<Vector<Real> > > gradient_storage_;
  std::shared_ptr<Vector<Real> > x_;
  std::shared_ptr<Vector<Real> > v_;
  std::shared_ptr<Vector<Real> > g_;
  std::shared_ptr<Vector<Real> > hv_;

  // Evaluate objective function at current parameter
  void getValue(Real &val, const Vector<Real> &x,
          const std::vector<Real> &param, Real &tol) {
    if ( storage_ && value_storage_.count(param) ) {
      val = value_storage_[param];
    }
    else {
      ParametrizedObjective_->setParameter(param);
      val = ParametrizedObjective_->value(x,tol);
      if ( storage_ ) {
        value_storage_.insert(std::pair<std::vector<Real>,Real>(param,val));
      }
    }
//std::cout << "BATCH ID: " << ValueSampler_->batchID() << "  "
//          << "POINT: (" << param[0] << ", " << param[1] << ", " << param[2] << ", " << param[3] << ")  "
//          << "VALUE: " << val << "\n";
  }
 
  // Evaluate gradient of objective function at current parameter
  void getGradient(Vector<Real> &g, const Vector<Real> &x,
             const std::vector<Real> &param, Real &tol) {
    if ( storage_ && gradient_storage_.count(param) ) {
      g.set(*(gradient_storage_[param]));
    }
    else {
      ParametrizedObjective_->setParameter(param);
      ParametrizedObjective_->gradient(g,x,tol);
      if ( storage_ ) {
        std::shared_ptr<Vector<Real> > tmp = g.clone();
        gradient_storage_.insert(std::pair<std::vector<Real>,std::shared_ptr<Vector<Real> > >(param,tmp));
        gradient_storage_[param]->set(g);
      }
    }
//std::cout << "BATCH ID: " << GradientSampler_->batchID() << "  "
//          << "POINT: (" << param[0] << ", " << param[1] << ", " << param[2] << ", " << param[3] << ")  "
//          << "GNORM: " << g.norm() << "\n";
  }

  // Evaluate Hessian-times-a-vector at current parameter
  void getHessVec(Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &x,
            const std::vector<Real> &param, Real &tol) {
    ParametrizedObjective_->setParameter(param);
    ParametrizedObjective_->hessVec(hv,v,x,tol);
  }

public:
  virtual ~RiskAverseObjective() {}

  RiskAverseObjective( const std::shared_ptr<Objective<Real> >       &pObj,
                       const std::shared_ptr<RiskMeasure<Real> >     &rm,
                       const std::shared_ptr<SampleGenerator<Real> > &vsampler,
                       const std::shared_ptr<SampleGenerator<Real> > &gsampler,
                       const std::shared_ptr<SampleGenerator<Real> > &hsampler,
                       const bool storage = true,
                       const int comp = 0, const int index = 0 )
    : ParametrizedObjective_(pObj), RiskMeasure_(rm),
      ValueSampler_(vsampler), GradientSampler_(gsampler), HessianSampler_(hsampler),
      firstUpdate_(true), storage_(storage) {
    value_storage_.clear();
    gradient_storage_.clear();
    RiskMeasure_->setRiskVectorInfo(comp,index);
  }

  RiskAverseObjective( const std::shared_ptr<Objective<Real> >       &pObj,
                       const std::shared_ptr<RiskMeasure<Real> >     &rm,
                       const std::shared_ptr<SampleGenerator<Real> > &vsampler,
                       const std::shared_ptr<SampleGenerator<Real> > &gsampler,
                       const bool storage = true,
                       const int comp = 0, const int index = 0 )
    : RiskAverseObjective(pObj,rm,vsampler,gsampler,gsampler,storage,comp,index) {}

  RiskAverseObjective( const std::shared_ptr<Objective<Real> >       &pObj,
                       const std::shared_ptr<RiskMeasure<Real> >     &rm,
                       const std::shared_ptr<SampleGenerator<Real> > &sampler,
                       const bool storage = true,
                       const int comp = 0, const int index = 0 )
    : RiskAverseObjective(pObj,rm,sampler,sampler,sampler,storage,comp,index) {}

  RiskAverseObjective( const std::shared_ptr<Objective<Real> >       &pObj,
                             Teuchos::ParameterList               &parlist,
                       const std::shared_ptr<SampleGenerator<Real> > &vsampler,
                       const std::shared_ptr<SampleGenerator<Real> > &gsampler,
                       const std::shared_ptr<SampleGenerator<Real> > &hsampler,
                       const int comp = 0, const int index = 0 )
    : ParametrizedObjective_(pObj),
      ValueSampler_(vsampler), GradientSampler_(gsampler), HessianSampler_(hsampler),
      firstUpdate_(true) {
    std::string name = parlist.sublist("SOL").sublist("Risk Measure").get("Name","CVaR");
    if (name != "Convex Combination Risk Measure") {
      RiskMeasure_ = RiskMeasureFactory<Real>(parlist);
    }
    else {
      RiskMeasure_ = std::make_shared<ConvexCombinationRiskMeasure<Real>>(parlist);
    }
    storage_ = parlist.sublist("SOL").get("Store Sampled Value and Gradient",true);
    value_storage_.clear();
    gradient_storage_.clear();
    RiskMeasure_->setRiskVectorInfo(comp,index);
  }

  RiskAverseObjective( const std::shared_ptr<Objective<Real> >       &pObj,
                             Teuchos::ParameterList               &parlist,
                       const std::shared_ptr<SampleGenerator<Real> > &vsampler,
                       const std::shared_ptr<SampleGenerator<Real> > &gsampler,
                       const int comp = 0, const int index = 0 )
    : RiskAverseObjective(pObj,parlist,vsampler,gsampler,gsampler,comp,index) {}

  RiskAverseObjective( const std::shared_ptr<Objective<Real> >       &pObj,
                             Teuchos::ParameterList               &parlist,
                       const std::shared_ptr<SampleGenerator<Real> > &sampler, 
                       const int comp = 0, const int index = 0 )
    : RiskAverseObjective(pObj,parlist,sampler,sampler,sampler,comp,index) {}

  virtual void update( const Vector<Real> &x, bool flag = true, int iter = -1 ) {
    RiskMeasure_->reset(x_,x);
    if ( firstUpdate_ ) {
      g_  = (x_->dual()).clone();
      hv_ = (x_->dual()).clone();
      firstUpdate_ = false;
    }
    ParametrizedObjective_->update(*x_,flag,iter);
    ValueSampler_->update(*x_);
    if ( storage_ ) {
      value_storage_.clear();
    }
    if ( flag ) {
      GradientSampler_->update(*x_);
      HessianSampler_->update(*x_);
      if ( storage_ ) {
        gradient_storage_.clear();
      }
    }
  }

  virtual Real value( const Vector<Real> &x, Real &tol ) {
    Real val = 0.0;
    RiskMeasure_->reset(x_,x);
    for ( int i = 0; i < ValueSampler_->numMySamples(); i++ ) {
      getValue(val,*x_,ValueSampler_->getMyPoint(i),tol);
      RiskMeasure_->update(val,ValueSampler_->getMyWeight(i));
    }
    return RiskMeasure_->getValue(*ValueSampler_);
  }

  virtual void gradient( Vector<Real> &g, const Vector<Real> &x, Real &tol ) {
    Real val = 0.0;
    g.zero();
    RiskMeasure_->reset(x_,x);
    for ( int i = 0; i < GradientSampler_->numMySamples(); i++ ) {
      getValue(val,*x_,GradientSampler_->getMyPoint(i),tol);
      getGradient(*g_,*x_,GradientSampler_->getMyPoint(i),tol);
      RiskMeasure_->update(val,*g_,GradientSampler_->getMyWeight(i));
    }
    RiskMeasure_->getGradient(g,*GradientSampler_);
  }

  virtual void hessVec( Vector<Real> &hv, const Vector<Real> &v,
                  const Vector<Real> &x, Real &tol ) {
    Real val = 0.0, gv = 0.0;
    hv.zero();
    RiskMeasure_->reset(x_,x,v_,v);
    for ( int i = 0; i < HessianSampler_->numMySamples(); i++ ) {
      getValue(val,*x_,HessianSampler_->getMyPoint(i),tol);
      getGradient(*g_,*x_,HessianSampler_->getMyPoint(i),tol);
      getHessVec(*hv_,*v_,*x_,HessianSampler_->getMyPoint(i),tol);
      gv = g_->dot(v_->dual());
      RiskMeasure_->update(val,*g_,gv,*hv_,HessianSampler_->getMyWeight(i));
    }
    RiskMeasure_->getHessVec(hv,*HessianSampler_);
  }

  virtual void precond( Vector<Real> &Pv, const Vector<Real> &v,
                  const Vector<Real> &x, Real &tol ) {
    Pv.set(v.dual());
  }
};

}

#endif
