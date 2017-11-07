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

#ifndef ROL_PRIMALDUALSYSTEMSTEP_H
#define ROL_PRIMALDUALSYSTEMSTEP_H

#include "ROL_NewtonKrylovStep.hpp"
#include "ROL_PrimalDualInteriorPointOperator.hpp"
#include "ROL_SchurComplememt.hpp"

/** @ingroup step_group
    \class ROL::PrimalDualSystemStep
    \brief Provides the interface to compute approximate
           solutions to 2x2 block systems arising from primal-dual
           interior point methods

           Note that as we do not need an additional Lagrange multiplier
           for the primal dual system, the vector expected to be passed
           in its place is the primal-dual residual
 */

namespace ROL {

template<class Real>
class PrimalDualSystemStep : public Step<Real> {

  typedef Vector<Real>             V;
  typedef PartitionedVector<Real>  PV;
  typedef Objective<Real>          OBJ;
  typedef BoundConstraint<Real>    BND;
  typedef Constraint<Real> CON;
  typedef AlgorithmState<Real>     AS;
  typedef SchurComplement<Real>    SCHUR;

  typedef PrimalDualInteriorPointBlock11  OP11;
  typedef PrimalDualInteriorPointBlock12  OP12;
  typedef PrimalDualInteriorPointBlock21  OP21;
  typedef PrimalDualInteriorPointBlock22  OP22;


private:

  // Block indices
  static const size_type OPT   = 0;
  static const size_type EQUAL = 1;
  static const size_type LOWER = 2;
  static const size_type UPPER = 3;

  // Super block indices
  static const size_type OPTMULT = 0;  // Optimization and equality multiplier components
  static const size_type BNDMULT = 1;  // Bound multiplier components

  std::shared_ptr<Secant<Real> > secant_;
  std::shared_ptr<Krylov<Real> > krylov_;
  std::shared_ptr<V> scratch1_;           // scratch vector
  std::shared_ptr<V> scratch_;

  std::shared_ptr<OP11> A_;
  std::shared_ptr<OP12> B_;
  std::shared_ptr<OP21> C_;
  std::shared_ptr<OP22> D_;

  std::shared_ptr<SCHUR> schur_; // Allows partial decoupling of (x,lambda) and (zl,zu)
  std::shared_ptr<OP>    op_;    // Solve fully coupled system

  int iterKrylov_; ///< Number of Krylov iterations (used for inexact Newton)
  int flagKrylov_; ///< Termination flag for Krylov method (used for inexact Newton)
  int verbosity_;  ///< Verbosity level

  bool useSecantPrecond_;
  bool useSchurComplement_;



  // Repartition (x,lambda,zl,zu) as (xlambda,z) = ((x,lambda),(zl,zu))
  std::shared_ptr<PV> repartition( V &x ) {

    PV &x_pv = dynamic_cast<PV&>(x);
    std::shared_ptr<V> xlambda = CreatePartitionedVector(x_pv.get(OPT),x_pv.get(EQUAL));
    std::shared_ptr<V> z = CreatePartitionedVector(x_pv.get(LOWER),x_pv.get(UPPER));

    std::shared_ptr<V> temp[] = {xlambda,z};

    return std::make_shared<PV( std::vector<std::shared_ptr<V> >>(temp,temp+2) );

  }

  // Repartition (x,lambda,zl,zu) as (xlambda,z) = ((x,lambda),(zl,zu))
  std::shared_ptr<const PV> repartition( const V &x ) {
    const PV &x_pv = dynamic_cast<const PV&>(x);
    std::shared_ptr<const V> xlambda = CreatePartitionedVector(x_pv.get(OPT),x_pv.get(EQUAL));
    std::shared_ptr<const V> z = CreatePartitionedVector(x_pv.get(LOWER),x_pv.get(UPPER));

    std::shared_ptr<const V> temp[] = {xlambda,z};

    return std::make_shared<PV( std::vector<std::shared_ptr<const V> >>(temp,temp+2) );

  }

public:

  using Step<Real>::initialize;
  using Step<Real>::compute;
  using Step<Real>::update;


  PrimalDualSystemStep( Teuchos::ParameterList &parlist,
                        const std::shared_ptr<Krylov<Real> > &krylov,
                        const std::shared_ptr<Secant<Real> > &secant,
                        std::shared_ptr<V> &scratch1 ) : Step<Real>(),
    krylov_(krylov), secant_(secant), scratch1_(scratch1), schur_(nullptr),
    op_(nullptr), useSchurComplement_(false) {

    PL &iplist = parlist.sublist("Step").sublist("Primal Dual Interior Point");
    PL &syslist = iplist.sublist("System Solver");

    useSchurComplement_ = syslist.get("Use Schur Complement",false);

  }

  PrimalDualSystemStep( Teuchos::ParameterList &parlist,
                        std::shared_ptr<V> &scratch1_ ) : Step<Real>() {
    PrimalDualSystemStep(parlist,nullptr,nullptr,scratch1);
  }

  void initialize( V &x, const V &g, V &res, const V &c,
                   OBJ &obj, CON &con, BND &bnd, AS &algo_state ) {

    Step<Real>::initialize(x,g,res,c,obj,con,bnd,algo_state);

    std::shared_ptr<OBJ> pObj(&obj);
    std::shared_ptr<CON> pCon(&con);
    std::shared_ptr<BND> pBnd(&bnd);

    std::shared_ptr<PV> x_pv = repartition(x);

    std::shared_ptr<V> xlambda = x_pv->get(OPTMULT);
    std::shared_ptr<V> z = x_pv->get(BNDMULT);

    A_ = std::make_shared<OP11>( pObj, pCon, *xlambda, scratch1_ );
    B_ = std::make_shared<OP12>( );
    C_ = std::make_shared<OP21>( *z );
    D_ = std::make_shared<OP22>( pBnd, *xlambda );

    if( useSchurComplement_ ) {
      schur_ = std::make_shared<SCHUR>(A_,B_,C_,D_,scratch1_);
    }
    else {
      op_ = BlockOperator2<Real>(A_,B_,C_,D_);
    }
  }

  void compute( V &s, const V &x, const V &res, OBJ &obj, CON &con,
                BND &bnd, AS &algo_state ) {

    std::shared_ptr<StepState<Real> > step_state = Step<Real>::getState();


    if( useSchurComplement_ ) {

      std::shared_ptr<const PV> x_pv = repartition(x);
      std::shared_ptr<const PV> res_pv = repartition(res);
      std::shared_ptr<PV> s_pv = repartition(s);


      // Decouple (x,lambda) from (zl,zu) so that s <- L

      std::shared_ptr<V> sxl   = s_pv->get(OPTMULT);
      std::shared_ptr<V> sz    = s_pv->get(BNDMULT);



    }
    else {

    }

  }

  void update( V &x, V &res, const V &s, OBJ &obj, CON &con,
               BND &bnd, AS &algo_state ) {

    std::shared_ptr<StepState<Real> > step_state = Step<Real>::getState();


  }


};

} // namespace ROL

#endif  // ROL_PRIMALDUALSYSTEMSTEP_H
