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

#ifndef ROL_SROMGENERATOR_HPP
#define ROL_SROMGENERATOR_HPP

#include "ROL_OptimizationSolver.hpp"
#include "ROL_ScalarLinearConstraint.hpp"
#include "ROL_SampleGenerator.hpp"
#include "ROL_MomentObjective.hpp"
#include "ROL_CDFObjective.hpp"
#include "ROL_LinearCombinationObjective.hpp"
#include "ROL_SROMVector.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_SingletonVector.hpp"
#include "ROL_Bounds.hpp"

namespace ROL {

template<class Real>
class SROMGenerator : public SampleGenerator<Real> {
private:
  // Parameterlist for optimization
  Teuchos::ParameterList parlist_;
  // Vector of distributions (size = dimension of space)
  std::vector<std::shared_ptr<Distribution<Real> > > dist_;

  const int dimension_;
  int numSamples_;
  int numMySamples_;
  int numNewSamples_;
  bool adaptive_;
  bool print_;

  Real ptol_;
  Real atol_;

  void pruneSamples(const ProbabilityVector<Real> &prob,
                    const AtomVector<Real>        &atom) {
    // Remove points with zero weight
    std::vector<std::vector<Real> > pts;
    std::vector<Real> wts;
    for (int i = 0; i < numMySamples_; i++) {
      if ( prob.getProbability(i) > ptol_ ) {
        pts.push_back(*(atom.getAtom(i)));
        wts.push_back(prob.getProbability(i));
      }
    }
    numMySamples_ = wts.size();
    // Remove atoms that are within atol of each other
    Real err = 0.0;
    std::vector<Real> pt;
    std::vector<int> ind;
    for (int i = 0; i < numMySamples_; i++) {
      pt = pts[i]; ind.clear();
      for (int j = i+1; j < numMySamples_; j++) {
        err = 0.0;
        for (int d = 0; d < dimension_; d++) {
          err += std::pow(pt[d] - pts[j][d],2);
        }
        err = std::sqrt(err);
        if ( err < atol_ ) {
          ind.push_back(j);
          for (int d = 0; d < dimension_; d++) {
            pts[i][d] += pts[j][d];
            wts[i]    += wts[j];
          }
        }
      }
      if ( ind.size() > 0 ) {
        for (int d = 0; d < dimension_; d++) {
          pts[i][d] /= (Real)(ind.size()+1);
        }
        for (int k = ind.size()-1; k >= 0; k--) {
          pts.erase(pts.begin()+ind[k]);
          wts.erase(wts.begin()+ind[k]);
        }
      }
      numMySamples_ = wts.size();
    }
    // Renormalize weights
    Real psum = 0.0, sum = 0.0;
    for (int i = 0; i < numMySamples_; i++) {
      psum += wts[i];
    }
    SampleGenerator<Real>::sumAll(&psum,&sum,1);
    for (int i = 0; i < numMySamples_; i++) {
      wts[i] /= sum;
    }
    // Set points and weights
    SampleGenerator<Real>::setPoints(pts);
    SampleGenerator<Real>::setWeights(wts);
  }

public:

  SROMGenerator(Teuchos::ParameterList                          &parlist,
          const std::shared_ptr<BatchManager<Real> >               &bman,
          const std::vector<std::shared_ptr<Distribution<Real> > > &dist)
    : SampleGenerator<Real>(bman), parlist_(parlist), dist_(dist),
      dimension_(dist.size()) {
    // Get SROM sublist
    Teuchos::ParameterList list = parlist.sublist("SOL").sublist("Sample Generator").sublist("SROM");
    numSamples_    = list.get("Number of Samples",50);
    adaptive_      = list.get("Adaptive Sampling",false);
    numNewSamples_ = list.get("Number of New Samples Per Adaptation",0);
    print_         = list.get("Output to Screen",false);
    ptol_          = list.get("Probability Tolerance",1.e2*std::sqrt(ROL_EPSILON<Real>()));
    atol_          = list.get("Atom Tolerance",1.e2*std::sqrt(ROL_EPSILON<Real>()));
    print_        *= !SampleGenerator<Real>::batchID();
    // Compute batch local number of samples
    int rank    = (int)SampleGenerator<Real>::batchID();
    int nProc   = (int)SampleGenerator<Real>::numBatches();
    int frac    = numSamples_ / nProc;
    int rem     = numSamples_ % nProc;
    numMySamples_  = frac + ((rank < rem) ? 1 : 0);
    // Initialize vectors
    std::shared_ptr<ProbabilityVector<Real> > prob, prob_lo, prob_hi, prob_eq;
    std::shared_ptr<AtomVector<Real> > atom, atom_lo, atom_hi, atom_eq;
    std::shared_ptr<Vector<Real> > x, x_lo, x_hi, x_eq;
    initialize_vectors(prob,prob_lo,prob_hi,prob_eq,atom,atom_lo,atom_hi,atom_eq,x,x_lo,x_hi,x_eq,bman);
    std::shared_ptr<Vector<Real> > l
      = std::make_shared<SingletonVector<Real>>(0.0);
    bool optProb = false, optAtom = true;
    for ( int i = 0; i < 2; i++ ) {
      if ( i == 0 ) { optProb = false; optAtom = true;  }
      if ( i == 1 ) { optProb = true;  optAtom = true;  }
      // Initialize objective function
      std::vector<std::shared_ptr<Objective<Real> > > obj_vec;
      std::shared_ptr<Objective<Real> > obj;
      initialize_objective(obj_vec,obj,dist,bman,optProb,optAtom,list);
      // Initialize constraints
      std::shared_ptr<BoundConstraint<Real> > bnd
        = std::make_shared<Bounds<Real>>(x_lo,x_hi);
      std::shared_ptr<Constraint<Real> > con
        = std::make_shared<ScalarLinearConstraint<Real>>(x_eq,1.0);
      // Test objective and constraints
      if ( print_ ) { std::cout << "\nCheck derivatives of CDFObjective\n"; }
      check_objective(*x,obj_vec[0],bman,optProb,optAtom);
      if ( print_ ) { std::cout << "\nCheck derivatives of MomentObjective\n"; }
      check_objective(*x,obj_vec[1],bman,optProb,optAtom);
      if ( print_ ) { std::cout << "\nCheck derivatives of LinearCombinationObjective\n"; }
      check_objective(*x,obj,bman,optProb,optAtom);
      if ( print_ && optProb ) { std::cout << "\nCheck ScalarLinearConstraint\n"; }
      check_constraint(*x,con,bman,optProb);
      // Solve optimization problems to sample
      std::shared_ptr<Algorithm<Real> > algo;
      initialize_optimizer(algo,list,optProb);
      if ( optProb ) {
        OptimizationProblem<Real> optProblem(obj,x,bnd,con,l);
        OptimizationSolver<Real>  optSolver(optProblem, list);
        optSolver.solve(std::cout);
      }
      else {
        algo->run(*x,*obj,*bnd,print_);
      }
    }
    // Prune samples with zero weight and set samples/weights
    pruneSamples(*prob,*atom);
  }

  void refine(void) {}

private:

  void get_scaling_vectors(std::vector<Real> &typw, std::vector<Real> &typx) const {
    typw.clear(); typx.clear();
    typw.resize(numMySamples_,(Real)(numSamples_*numSamples_));
    typx.resize(numMySamples_*dimension_,0);
    Real mean = 1, var = 1, one(1);
    for (int j = 0; j < dimension_; j++) {
      mean = std::abs(dist_[j]->moment(1));
      var  = dist_[j]->moment(2) - mean*mean;
      mean = ((mean > ROL_EPSILON<Real>()) ? mean : std::sqrt(var));
      mean = ((mean > ROL_EPSILON<Real>()) ? mean : one);
      for (int i = 0; i < numMySamples_; i++) {
        typx[i*dimension_ + j] = one/(mean*mean);
      }
    }
  }

  void initialize_vectors(std::shared_ptr<ProbabilityVector<Real> >  &prob,
                          std::shared_ptr<ProbabilityVector<Real> >  &prob_lo,
                          std::shared_ptr<ProbabilityVector<Real> >  &prob_hi,
                          std::shared_ptr<ProbabilityVector<Real> >  &prob_eq,
                          std::shared_ptr<AtomVector<Real> >         &atom,
                          std::shared_ptr<AtomVector<Real> >         &atom_lo,
                          std::shared_ptr<AtomVector<Real> >         &atom_hi,
                          std::shared_ptr<AtomVector<Real> >         &atom_eq,
                          std::shared_ptr<Vector<Real> >             &vec,
                          std::shared_ptr<Vector<Real> >             &vec_lo,
                          std::shared_ptr<Vector<Real> >             &vec_hi,
                          std::shared_ptr<Vector<Real> >             &vec_eq,
                          const std::shared_ptr<BatchManager<Real> > &bman) const {
    // Compute scaling for probability and atom vectors
    std::vector<Real> typx, typw;
    get_scaling_vectors(typw,typx);
    // Compute initial guess and bounds for probability and atom vectors
    std::vector<Real> pt(dimension_*numMySamples_,0.), wt(numMySamples_,1./(Real)numSamples_);
    std::vector<Real> pt_lo(dimension_*numMySamples_,0.), pt_hi(dimension_*numMySamples_,0.);
    std::vector<Real> wt_lo(numMySamples_,0.), wt_hi(numMySamples_,1.);
    std::vector<Real> pt_eq(dimension_*numMySamples_,0.), wt_eq(numMySamples_,1.);
    Real lo = 0., hi = 0.;
    srand(12345*SampleGenerator<Real>::batchID());
    for ( int j = 0; j < dimension_; j++) {
      lo = dist_[j]->lowerBound();
      hi = dist_[j]->upperBound();
      for (int i = 0; i < numMySamples_; i++) {
        pt[i*dimension_ + j] = dist_[j]->invertCDF((Real)rand()/(Real)RAND_MAX);
        //pt[i*dimension_ + j] = dist_[j]->invertCDF(0);
        pt_lo[i*dimension_ + j] = lo;
        pt_hi[i*dimension_ + j] = hi;
      }
    }
    // Build probability, atom, and SROM vectors
    prob = std::make_shared<PrimalProbabilityVector<Real>>(
           std::make_shared<std::vector<Real>>(wt),bman,
           std::make_shared<std::vector<Real>>(typw));
    atom = std::make_shared<PrimalAtomVector<Real>>(
           std::make_shared<std::vector<Real>>(pt),bman,numMySamples_,dimension_,
           std::make_shared<std::vector<Real>>(typx));
    vec  = std::make_shared<SROMVector<Real>>(prob,atom);
    // Lower and upper bounds on Probability Vector
    prob_lo = std::make_shared<PrimalProbabilityVector<Real>>(
              std::make_shared<std::vector<Real>>(wt_lo),bman,
              std::make_shared<std::vector<Real>>(typw));
    prob_hi = std::make_shared<PrimalProbabilityVector<Real>>(
              std::make_shared<std::vector<Real>>(wt_hi),bman,
              std::make_shared<std::vector<Real>>(typw));
    // Lower and upper bounds on Atom Vector
    atom_lo = std::make_shared<PrimalAtomVector<Real>>(
              std::make_shared<std::vector<Real>>(pt_lo),bman,numMySamples_,dimension_,
              std::make_shared<std::vector<Real>>(typx));
    atom_hi = std::make_shared<PrimalAtomVector<Real>>(
              std::make_shared<std::vector<Real>>(pt_hi),bman,numMySamples_,dimension_,
              std::make_shared<std::vector<Real>>(typx));
    // Lower and upper bounds on SROM Vector
    vec_lo = std::make_shared<SROMVector<Real>>(prob_lo,atom_lo);
    vec_hi = std::make_shared<SROMVector<Real>>(prob_hi,atom_hi);
    // Constraint vectors
    prob_eq = std::make_shared<DualProbabilityVector<Real>>(
              std::make_shared<std::vector<Real>>(wt_eq),bman,
              std::make_shared<std::vector<Real>>(typw));
    atom_eq = std::make_shared<DualAtomVector<Real>>(
              std::make_shared<std::vector<Real>>(pt_eq),bman,numMySamples_,dimension_,
              std::make_shared<std::vector<Real>>(typx));
    vec_eq  = std::make_shared<SROMVector<Real>>(prob_eq,atom_eq);
  }

  void initialize_objective(std::vector<std::shared_ptr<Objective<Real> > > &obj_vec,
                            std::shared_ptr<Objective<Real> >               &obj,
                            const std::vector<std::shared_ptr<Distribution<Real> > > &dist,
                            const std::shared_ptr<BatchManager<Real> >      &bman,
                            const bool optProb, const bool optAtom,
                            Teuchos::ParameterList                       &list) const {
    // Build CDF objective function
    Real scale = list.get("CDF Smoothing Parameter",1.e-2);
    obj_vec.push_back(std::make_shared<CDFObjective<Real>>(dist,bman,scale,optProb,optAtom));
    // Build moment matching objective function
    Teuchos::Array<int> tmp_order
      = Teuchos::getArrayFromStringParameter<int>(list,"Moments");
    std::vector<int> order(tmp_order.size(),0);
    for ( int i = 0; i < tmp_order.size(); i++) {
      order[i] = static_cast<int>(tmp_order[i]);
    }
    obj_vec.push_back(std::make_shared<MomentObjective<Real>>(dist,order,bman,optProb,optAtom));
    // Build linear combination objective function
    Teuchos::Array<Real> tmp_coeff
      = Teuchos::getArrayFromStringParameter<Real>(list,"Coefficients");
    std::vector<Real> coeff(2,0.);
    coeff[0] = tmp_coeff[0]; coeff[1] = tmp_coeff[1];
    obj = std::make_shared<LinearCombinationObjective<Real>>(coeff,obj_vec);
  }

  void initialize_optimizer(std::shared_ptr<Algorithm<Real> > &algo,
                            Teuchos::ParameterList         &parlist,
                            const bool optProb) const {
    std::string type = parlist.sublist("Step").get("Type","Trust Region");
    if ( optProb ) {
      if ( type == "Moreau-Yosida Penalty" ) {
        algo = std::make_shared<Algorithm<Real>>("Moreau-Yosida Penalty",parlist,false);
      }
      else if ( type == "Augmented Lagrangian" ) {
        algo = std::make_shared<Algorithm<Real>>("Augmented Lagrangian",parlist,false);
      }
      else {
        algo = std::make_shared<Algorithm<Real>>("Interior Point",parlist,false);
      }
    }
    else {
      algo = std::make_shared<Algorithm<Real>>("Trust Region",parlist,false);
    }
  }

  void check_objective(const Vector<Real>                      &x,
                       const std::shared_ptr<Objective<Real> >    &obj,
                       const std::shared_ptr<BatchManager<Real> > &bman,
                       const bool optProb, const bool optAtom) {
    // Get scaling for probability and atom vectors
    std::vector<Real> typx, typw;
    get_scaling_vectors(typw,typx);
    // Set random direction
    std::vector<Real> pt(dimension_*numMySamples_,0.), wt(numMySamples_,0.);
    for (int i = 0; i < numMySamples_; i++) {
      wt[i] = (optProb ? (Real)rand()/(Real)RAND_MAX : 0);
      for ( int j = 0; j < dimension_; j++) {
        pt[i*dimension_ + j] = (optAtom ? dist_[j]->invertCDF((Real)rand()/(Real)RAND_MAX) : 0);
      }
    }
    std::shared_ptr<ProbabilityVector<Real> > dprob
      = std::make_shared<PrimalProbabilityVector<Real>>(
        std::make_shared<std::vector<Real>>(wt),bman,
        std::make_shared<std::vector<Real>>(typw));
    std::shared_ptr<AtomVector<Real> > datom
      = std::make_shared<PrimalAtomVector<Real>>(
        std::make_shared<std::vector<Real>>(pt),bman,numMySamples_,dimension_,
        std::make_shared<std::vector<Real>>(typx));
    SROMVector<Real> d = SROMVector<Real>(dprob,datom);
    // Check derivatives
    obj->checkGradient(x,d,print_);
    obj->checkHessVec(x,d,print_);
  }

  void check_constraint(const Vector<Real>                            &x,
                        const std::shared_ptr<Constraint<Real> >         &con,
                        const std::shared_ptr<BatchManager<Real> >       &bman,
                        const bool optProb) {
    if ( optProb ) {
      SingletonVector<Real> c(1.0);
      // Get scaling for probability and atom vectors
      std::vector<Real> typx, typw;
      get_scaling_vectors(typw,typx);
      // Set random direction
      std::vector<Real> pt(dimension_*numMySamples_,0.), wt(numMySamples_,0.);
      for (int i = 0; i < numMySamples_; i++) {
        wt[i] = (Real)rand()/(Real)RAND_MAX;
        for ( int j = 0; j < dimension_; j++) {
          pt[i*dimension_ + j] = dist_[j]->invertCDF((Real)rand()/(Real)RAND_MAX);
        }
      }
      std::shared_ptr<ProbabilityVector<Real> > dprob
        = std::make_shared<PrimalProbabilityVector<Real>>(
          std::make_shared<std::vector<Real>>(wt),bman,
          std::make_shared<std::vector<Real>>(typw));
      std::shared_ptr<AtomVector<Real> > datom
        = std::make_shared<PrimalAtomVector<Real>>(
          std::make_shared<std::vector<Real>>(pt),bman,numMySamples_,dimension_,
          std::make_shared<std::vector<Real>>(typx));
      SROMVector<Real> d = SROMVector<Real>(dprob,datom);
      // Check derivatives
      con->checkApplyJacobian(x,d,c,print_);
      con->checkAdjointConsistencyJacobian(c,d,x,print_);
    }
  }
};

}

#endif
