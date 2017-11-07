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

/*! \file  example_01.cpp
    \brief Shows how to minimize Rosenbrock's function using Newton-Krylov.
*/

#define USE_HESSVEC 1

#include "ROL_Rosenbrock.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_TrustRegionStep.hpp"
#include "ROL_StatusTest.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include <iostream>

typedef double RealT;


/*** Declare two vector spaces. ***/

// Forward declarations:

template <class Real, class Element=Real>
class OptStdVector;  // Optimization space.

template <class Real, class Element=Real>
class OptDualStdVector;  // Dual optimization space.


// Vector space definitions:

// Optimization space.
template <class Real, class Element>
class OptStdVector : public ROL::Vector<Real> {

  typedef std::vector<Element> vector;
  typedef ROL::Vector<Real>    V;

  typedef typename vector::size_type uint;

private:
std::shared_ptr<std::vector<Element> >  std_vec_;
mutable std::shared_ptr<OptDualStdVector<Real> >  dual_vec_;

public:

OptStdVector(const std::shared_ptr<std::vector<Element> > & std_vec) : std_vec_(std_vec), dual_vec_(nullptr) {}

void plus( const ROL::Vector<Real> &x ) {
     
  std::shared_ptr<const vector> xvalptr = dynamic_cast<const OptStdVector&>(x).getVector(); 
  uint dimension  = std_vec_->size();
  for (uint i=0; i<dimension; i++) {
    (*std_vec_)[i] += (*xvalptr)[i];
  }
}

void scale( const Real alpha ) {
  uint dimension = std_vec_->size();
  for (uint i=0; i<dimension; i++) {
    (*std_vec_)[i] *= alpha;
  }
}

Real dot( const ROL::Vector<Real> &x ) const {
  Real val = 0;
    
  std::shared_ptr<const vector> xvalptr = dynamic_cast<const OptStdVector&>(x).getVector(); 
  uint dimension  = std_vec_->size();
  for (uint i=0; i<dimension; i++) {
    val += (*std_vec_)[i]*(*xvalptr)[i];
  }
  return val;
}

Real norm() const {
  Real val = 0;
  val = std::sqrt( dot(*this) );
  return val;
}

std::shared_ptr<ROL::Vector<Real> > clone() const {
  return std::make_shared<OptStdVector( Teuchos::std::make_shared<std::vector<Element>(std_vec_->size>>()) );
}

std::shared_ptr<const std::vector<Element> > getVector() const {
  return std_vec_;
}

std::shared_ptr<std::vector<Element> > getVector() {
  return std_vec_;
}

std::shared_ptr<ROL::Vector<Real> > basis( const int i ) const {
    
  std::shared_ptr<vector> e_rcp = std::make_shared<vector(std_vec_->size>(),0.0);
  (*e_rcp)[i] = 1.0;
  
  std::shared_ptr<V> e = std::make_shared<OptStdVector>( e_rcp );
  
  return e;
}

int dimension() const {return static_cast<int>(std_vec_->size());}

const ROL::Vector<Real> & dual() const {
  dual_vec_ = std::make_shared<OptDualStdVector<Real>( Teuchos::std::make_shared<std::vector<Element>>>(*std_vec_) );
  return *dual_vec_;
}

}; // class OptStdVector


// Dual optimization space.
template <class Real, class Element>
class OptDualStdVector : public ROL::Vector<Real> {

  typedef std::vector<Element> vector;
  typedef ROL::Vector<Real>    V;

  typedef typename vector::size_type uint;

private:
std::shared_ptr<std::vector<Element> >  std_vec_;
mutable std::shared_ptr<OptStdVector<Real> >  dual_vec_;

public:

OptDualStdVector(const std::shared_ptr<std::vector<Element> > & std_vec) : std_vec_(std_vec), dual_vec_(nullptr) {}

void plus( const ROL::Vector<Real> &x ) {
      
  std::shared_ptr<const vector> xvalptr = dynamic_cast<const OptDualStdVector&>(x).getVector(); 

  uint dimension  = std_vec_->size();
  for (uint i=0; i<dimension; i++) {
    (*std_vec_)[i] += (*xvalptr)[i];
  }
}

void scale( const Real alpha ) {
  uint dimension = std_vec_->size();
  for (uint i=0; i<dimension; i++) {
    (*std_vec_)[i] *= alpha;
  }
}

Real dot( const ROL::Vector<Real> &x ) const {
  Real val = 0;
     
  std::shared_ptr<const vector> xvalptr = dynamic_cast<const OptDualStdVector&>(x).getVector(); 
  uint dimension  = std_vec_->size();
  for (uint i=0; i<dimension; i++) {
    val += (*std_vec_)[i]*(*xvalptr)[i];
  }
  return val;
}

Real norm() const {
  Real val = 0;
  val = std::sqrt( dot(*this) );
  return val;
}

std::shared_ptr<ROL::Vector<Real> > clone() const {
  return std::make_shared<OptDualStdVector( Teuchos::std::make_shared<std::vector<Element>(std_vec_->size>>()) );
}

std::shared_ptr<const std::vector<Element> > getVector() const {
  return std_vec_;
}

std::shared_ptr<std::vector<Element> > getVector() {
  return std_vec_;
}

std::shared_ptr<ROL::Vector<Real> > basis( const int i ) const {
    
  std::shared_ptr<vector> e_rcp = std::make_shared<vector(std_vec_->size>(),0.0);
  (*e_rcp)[i] = 1.0;
  
  std::shared_ptr<V> e = std::make_shared<OptDualStdVector>( e_rcp );
  return e;
}

int dimension() const {return std_vec_->size();}

const ROL::Vector<Real> & dual() const {
  dual_vec_ = std::make_shared<OptStdVector<Real>( Teuchos::std::make_shared<std::vector<Element>>>(*std_vec_) );
  return *dual_vec_;
}

}; // class OptDualStdVector


/*** End of declaration of two vector spaces. ***/






int main(int argc, char *argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = &std::cout, false;
  else
    outStream = &bhs, false;

  int errorFlag  = 0;

  // *** Example body.

  try {

    ROL::ZOO::Objective_Rosenbrock<RealT, OptStdVector<RealT>, OptDualStdVector<RealT> > obj;
    int dim = 100; // Set problem dimension. Must be even.

    // Define algorithm.
    Teuchos::ParameterList parlist;
    std::string stepname = "Trust Region";
    parlist.sublist("Step").sublist(stepname).set("Subproblem Solver", "Truncated CG");
    parlist.sublist("General").sublist("Krylov").set("Iteration Limit",10);
    parlist.sublist("General").sublist("Krylov").set("Relative Tolerance",1e-2);
    parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance",1e-4);
    parlist.sublist("General").sublist("Secant").set("Use as Hessian",true);
    parlist.sublist("Status Test").set("Gradient Tolerance",1.e-12);
    parlist.sublist("Status Test").set("Step Tolerance",1.e-14);
    parlist.sublist("Status Test").set("Iteration Limit",100);
    ROL::Algorithm<RealT> algo(stepname,parlist);

    // Iteration Vector
    std::shared_ptr<std::vector<RealT> > x_rcp = std::make_shared<std::vector<RealT>>(dim, 0.0);
    std::shared_ptr<std::vector<RealT> > g_rcp = std::make_shared<std::vector<RealT>>(dim, 0.0);
    // Set Initial Guess
    for (int i=0; i<dim/2; i++) {
      (*x_rcp)[2*i]   = -1.2;
      (*x_rcp)[2*i+1] =  1.0;
      (*g_rcp)[2*i]   = 0;
      (*g_rcp)[2*i+1] = 0;
    }

    OptStdVector<RealT> x(x_rcp); // Iteration Vector
    OptDualStdVector<RealT> g(g_rcp); // zeroed gradient vector in dual space

    std::shared_ptr<std::vector<RealT> > aa_rcp = std::make_shared<std::vector<RealT>>(1, 1.0);
    OptDualStdVector<RealT> av(aa_rcp);
    std::shared_ptr<std::vector<RealT> > bb_rcp = std::make_shared<std::vector<RealT>>(1, 2.0);
    OptDualStdVector<RealT> bv(bb_rcp);
    std::shared_ptr<std::vector<RealT> > cc_rcp = std::make_shared<std::vector<RealT>>(1, 3.0);
    OptDualStdVector<RealT> cv(cc_rcp);
    std::vector<RealT> std_vec_err = av.checkVector(bv,cv,true,*outStream);

    // Run Algorithm
    std::vector<std::string> output = algo.run(x,g, obj, true, *outStream);

    // Get True Solution
    std::shared_ptr<std::vector<RealT> > xtrue_rcp = std::make_shared<std::vector<RealT>>(dim, 1.0);
    OptStdVector<RealT> xtrue(xtrue_rcp); 
   
    // Compute Errors
    x.axpy(-1.0, xtrue);
    RealT abserr = x.norm();
    RealT relerr = abserr/xtrue.norm();
    *outStream << std::scientific << "\n   Absolute solution error: " << abserr;
    *outStream << std::scientific << "\n   Relative solution error: " << relerr;
    if ( relerr > sqrt(ROL::ROL_EPSILON<RealT>()) ) {
      errorFlag += 1;
    }
    std::shared_ptr<std::vector<RealT> > vec_err_rcp = std::make_shared<std::vector<RealT>>(std_vec_err);
    ROL::StdVector<RealT> vec_err(vec_err_rcp);
    *outStream << std::scientific << "\n   Linear algebra error: " << vec_err.norm() << std::endl;
    if ( vec_err.norm() > 1e2*ROL::ROL_EPSILON<RealT>() ) {
      errorFlag += 1;
    }
  }
  catch (std::logic_error err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

