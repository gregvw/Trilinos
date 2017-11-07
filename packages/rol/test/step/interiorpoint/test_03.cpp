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

#define OPTIMIZATION_PROBLEM_REFACTOR

#include "ROL_RandomVector.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_NonlinearProgram.hpp"
#include "ROL_OptimizationProblem.hpp"
#include "ROL_InteriorPointPenalty.hpp"
#include "ROL_PrimalDualInteriorPointResidual.hpp"
#include "ROL_LinearOperatorFromConstraint.hpp"
#include "ROL_KrylovFactory.hpp"

#include "HS_ProblemFactory.hpp"

#include <iomanip>

/*! \file test_03.cpp 
    \brief Verify that the symmetrized version of a primal dual
           system is indeed symmetric and that the solution to 
           the unsymmetrized version satisfies the symmetrized version.
 
           Note: CG will almost certainly fail with exit flag 2 (negative
           eigenvalues)
*/


template<class Real> 
void printVector( const ROL::Vector<Real> &x, std::ostream &outStream ) {

  try {
    std::shared_ptr<const std::vector<Real> > xp = 
      dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();

    outStream << "Standard Vector" << std::endl;
    for( size_t i=0; i<xp->size(); ++i ) {
      outStream << (*xp)[i] << std::endl;
    }
  }
  catch( const std::bad_cast& e ) {
    outStream << "Partitioned Vector" << std::endl;
    
    typedef ROL::PartitionedVector<Real>    PV;
    typedef typename PV::size_type          size_type;

    const PV &xpv = dynamic_cast<const PV&>(x);

    for( size_type i=0; i<xpv.numVectors(); ++i ) {
      outStream << "--------------------" << std::endl;
      printVector( *(xpv.get(i)), outStream );
    }
    outStream << "--------------------" << std::endl;
  }
}

template<class Real> 
void printMatrix( const std::vector<std::shared_ptr<ROL::Vector<Real> > > &A,
                  const std::vector<std::shared_ptr<ROL::Vector<Real> > > &I,
                  std::ostream &outStream ) {
  typedef typename std::vector<Real>::size_type uint;
  uint dim = A.size();
   
  for( uint i=0; i<dim; ++i ) {
    for( uint j=0; j<dim; ++j ) {
      outStream << std::setw(6) << A[j]->dot(*(I[i])); 
    }
    outStream << std::endl;
  }
}


template<class Real> 
class IdentityOperator : public ROL::LinearOperator<Real> {
public:
  void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const {
    Hv.set(v);
  }
}; // IdentityOperator


typedef double RealT;

int main(int argc, char *argv[]) {
 
//  typedef std::vector<RealT>                             vector;

  typedef Teuchos::ParameterList                           PL;

  typedef ROL::Vector<RealT>                               V;
  typedef ROL::PartitionedVector<RealT>                    PV;
  typedef ROL::Objective<RealT>                            OBJ;
  typedef ROL::Constraint<RealT>                           CON;
  typedef ROL::BoundConstraint<RealT>                      BND;
  typedef ROL::OptimizationProblem<RealT>                  OPT;
  typedef ROL::NonlinearProgram<RealT>                     NLP;
  typedef ROL::LinearOperator<RealT>                       LOP;  
  typedef ROL::LinearOperatorFromConstraint<RealT>         LOPEC;
  typedef ROL::Krylov<RealT>                               KRYLOV;


  typedef ROL::InteriorPointPenalty<RealT>                 PENALTY;
  typedef ROL::PrimalDualInteriorPointResidual<RealT>      RESIDUAL;

   

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  int iprint = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs;
  if( iprint > 0 ) 
    outStream = &std::cout,false;
  else
    outStream = &bhs,false;

  int errorFlag = 0;
   
  try {

    RealT mu = 0.1;

    RealT tol = std::sqrt(ROL::ROL_EPSILON<RealT>());

    PL parlist;

    PL &iplist = parlist.sublist("Step").sublist("Primal Dual Interior Point");
    PL &lblist = iplist.sublist("Barrier Objective");

    lblist.set("Use Linear Damping", true);         // Not used in this test
    lblist.set("Linear Damping Coefficient",1.e-4); // Not used in this test 
    lblist.set("Initial Barrier Parameter", mu);

    PL &krylist = parlist.sublist("General").sublist("Krylov");
   
    krylist.set("Absolute Tolerance", 1.e-6);
    krylist.set("Relative Tolerance", 1.e-6);
    krylist.set("Iteration Limit", 50);

    // Create a Conjugate Gradients solver 
    krylist.set("Type","Conjugate Gradients"); 
    std::shared_ptr<KRYLOV> cg = ROL::KrylovFactory<RealT>(parlist);
    HS::ProblemFactory<RealT> problemFactory;

    // Choose an example problem with inequality constraints and
    // a mixture of finite and infinite bounds
    std::shared_ptr<NLP> nlp = problemFactory.getProblem(16);
    std::shared_ptr<OPT> opt = nlp->getOptimizationProblem();
 
    std::shared_ptr<V>   x   = opt->getSolutionVector();
    std::shared_ptr<V>   l   = opt->getMultiplierVector();
    std::shared_ptr<V>   zl  = x->clone(); zl->zero();
    std::shared_ptr<V>   zu  = x->clone(); zu->zero();

    std::shared_ptr<V>   scratch = x->clone();

    std::shared_ptr<PV>  x_pv = std::dynamic_pointer_cast<PV>(x);
    // New slack variable initialization does not guarantee strict feasibility.
    // This ensures that the slack variables are the same as the previous
    // implementation.
    (*std::dynamic_pointer_cast<ROL::StdVector<RealT> >(x_pv->get(1))->getVector())[0] = 1.0;

    std::shared_ptr<V>   sol = CreatePartitionedVector(x,l,zl,zu);   

    std::vector<std::shared_ptr<V> > I;
    std::vector<std::shared_ptr<V> > J;

    for( int k=0; k<sol->dimension(); ++k ) {
      I.push_back(sol->basis(k));
      J.push_back(sol->clone());
    }

    std::shared_ptr<V>   u = sol->clone();
    std::shared_ptr<V>   v = sol->clone();

    std::shared_ptr<V>   rhs = sol->clone();
    std::shared_ptr<V>   symrhs = sol->clone();

    std::shared_ptr<V>   gmres_sol = sol->clone();   gmres_sol->set(*sol);
    std::shared_ptr<V>   cg_sol = sol->clone();      cg_sol->set(*sol);
 
    IdentityOperator<RealT> identity;

    RandomizeVector(*u,-1.0,1.0);
    RandomizeVector(*v,-1.0,1.0);

    std::shared_ptr<OBJ> obj = opt->getObjective();
    std::shared_ptr<CON> con = opt->getConstraint();
    std::shared_ptr<BND> bnd = opt->getBoundConstraint();

    PENALTY penalty(obj,bnd,parlist);
 
    std::shared_ptr<const V> maskL = penalty.getLowerMask();
    std::shared_ptr<const V> maskU = penalty.getUpperMask();

    zl->set(*maskL);
    zu->set(*maskU);

    /********************************************************************************/
    /* Nonsymmetric representation test                                             */
    /********************************************************************************/

    int gmres_iter = 0;
    int gmres_flag = 0;

    // Form the residual's Jacobian operator
    std::shared_ptr<CON> res = std::make_shared<RESIDUAL>(obj,con,bnd,*sol,maskL,maskU,scratch,mu,false);
    std::shared_ptr<LOP> lop = std::make_shared<LOPEC>( sol, res );

    // Evaluate the right-hand-side
    res->value(*rhs,*sol,tol);

    // Create a GMRES solver
    krylist.set("Type","GMRES");
    std::shared_ptr<KRYLOV> gmres = ROL::KrylovFactory<RealT>(parlist);

     for( int k=0; k<sol->dimension(); ++k ) {
      res->applyJacobian(*(J[k]),*(I[k]),*sol,tol);
    }

    *outStream << "Nonsymmetric Jacobian" << std::endl;
    printMatrix(J,I,*outStream);

   // Solve the system 
    gmres->run( *gmres_sol, *lop, *rhs, identity, gmres_iter, gmres_flag );

    errorFlag += gmres_flag;

    *outStream << "GMRES terminated after " << gmres_iter << " iterations "
               << "with exit flag " << gmres_flag << std::endl;


    /********************************************************************************/
    /* Symmetric representation test                                                */
    /********************************************************************************/

    int cg_iter = 0;
    int cg_flag = 0;

    std::shared_ptr<V> jv = v->clone();
    std::shared_ptr<V> ju = u->clone();

    iplist.set("Symmetrize Primal Dual System",true);
    std::shared_ptr<CON> symres = std::make_shared<RESIDUAL>(obj,con,bnd,*sol,maskL,maskU,scratch,mu,true);
    std::shared_ptr<LOP> symlop = std::make_shared<LOPEC>( sol, res );
    symres->value(*symrhs,*sol,tol);

    symres->applyJacobian(*jv,*v,*sol,tol);
    symres->applyJacobian(*ju,*u,*sol,tol);
    *outStream << "Symmetry check |u.dot(jv)-v.dot(ju)| = "
               << std::abs(u->dot(*jv)-v->dot(*ju)) << std::endl;
 
    cg->run( *cg_sol, *symlop, *symrhs, identity, cg_iter, cg_flag );

    *outStream << "CG terminated after " << cg_iter << " iterations "
               << "with exit flag " << cg_flag << std::endl;

    *outStream << "Check that GMRES solution also is a solution to the symmetrized system"
               << std::endl;

    symres->applyJacobian(*ju,*gmres_sol,*sol,tol);
    ju->axpy(-1.0,*symrhs);
    RealT mismatch = ju->norm();
    if(mismatch>tol) {
      errorFlag++;
    }
    *outStream << "||J_sym*sol_nonsym-rhs_sym|| = " << mismatch << std::endl;

  }
  catch (std::logic_error err) {
    *outStream << err.what() << std::endl;
    errorFlag = -1000;
  }

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED" << std::endl;
  else
    std::cout << "End Result: TEST PASSED" << std::endl;

  return 0;
}

