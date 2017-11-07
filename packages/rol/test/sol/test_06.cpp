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

/*! \file  test_01.cpp
    \brief Test SROMvector interface.
*/


#include "ROL_SROMVector.hpp"
#include "ROL_TeuchosBatchManager.hpp"

#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_DefaultComm.hpp"

typedef double RealT;

int main(int argc, char *argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  std::shared_ptr<const Teuchos::Comm<int> > comm
    = Teuchos::DefaultComm<int>::getComm();

  int iprint = argc - 1;
  Teuchos::oblackholestream bhs; // outputs nothing
  std::ostream& outStream = (iprint > 0 && !Teuchos::rank<int>(*comm)) ? std::cout : bhs;

  int errorFlag = 0;

  RealT errtol = ROL::ROL_THRESHOLD<RealT>();

  try {
    // Batch manager
    std::shared_ptr<ROL::BatchManager<RealT> > bman =
      std::make_shared<ROL::TeuchosBatchManager<RealT,int>>(comm);

    // Dimension of the optimization vector
    int dimension = 5, numMyAtoms = 10;
    int size = dimension*numMyAtoms;

    // Create batch std vectors 
    std::shared_ptr<std::vector<RealT> > b1_rcp = std::make_shared<std::vector<RealT>>(size);
    std::shared_ptr<std::vector<RealT> > b2_rcp = std::make_shared<std::vector<RealT>>(size);
    std::shared_ptr<std::vector<RealT> > b3_rcp = std::make_shared<std::vector<RealT>>(size);
    for (int i = 0; i < size; ++i) {
      (*b1_rcp)[i] = static_cast<RealT>(rand())/static_cast<RealT>(RAND_MAX);
      (*b2_rcp)[i] = static_cast<RealT>(rand())/static_cast<RealT>(RAND_MAX);
      (*b3_rcp)[i] = static_cast<RealT>(rand())/static_cast<RealT>(RAND_MAX);
    }
    std::shared_ptr<ROL::BatchStdVector<RealT> > b1
      = std::make_shared<ROL::BatchStdVector<RealT>>(b1_rcp,bman);
    std::shared_ptr<ROL::BatchStdVector<RealT> > b2
      = std::make_shared<ROL::BatchStdVector<RealT>>(b2_rcp,bman);
    std::shared_ptr<ROL::BatchStdVector<RealT> > b3
      = std::make_shared<ROL::BatchStdVector<RealT>>(b3_rcp,bman);

    // Create atom vectors 
    std::shared_ptr<std::vector<RealT> > a1_rcp = std::make_shared<std::vector<RealT>>(size);
    std::shared_ptr<std::vector<RealT> > a2_rcp = std::make_shared<std::vector<RealT>>(size);
    std::shared_ptr<std::vector<RealT> > a3_rcp = std::make_shared<std::vector<RealT>>(size);
    std::shared_ptr<std::vector<RealT> > aW_rcp = std::make_shared<std::vector<RealT>>(size);
    for (int i = 0; i < size; ++i) {
      (*a1_rcp)[i] = static_cast<RealT>(rand())/static_cast<RealT>(RAND_MAX);
      (*a2_rcp)[i] = static_cast<RealT>(rand())/static_cast<RealT>(RAND_MAX);
      (*a3_rcp)[i] = static_cast<RealT>(rand())/static_cast<RealT>(RAND_MAX);
      (*aW_rcp)[i] = static_cast<RealT>(2);
    }
    std::shared_ptr<ROL::PrimalAtomVector<RealT> > a1
      = std::make_shared<ROL::PrimalAtomVector<RealT>>(a1_rcp,bman,numMyAtoms,dimension,aW_rcp);
    std::shared_ptr<ROL::PrimalAtomVector<RealT> > a2
      = std::make_shared<ROL::PrimalAtomVector<RealT>>(a2_rcp,bman,numMyAtoms,dimension,aW_rcp);
    std::shared_ptr<ROL::PrimalAtomVector<RealT> > a3
      = std::make_shared<ROL::PrimalAtomVector<RealT>>(a3_rcp,bman,numMyAtoms,dimension,aW_rcp);

    // Create probability vectors
    std::shared_ptr<std::vector<RealT> > p1_rcp = std::make_shared<std::vector<RealT>>(numMyAtoms);
    std::shared_ptr<std::vector<RealT> > p2_rcp = std::make_shared<std::vector<RealT>>(numMyAtoms);
    std::shared_ptr<std::vector<RealT> > p3_rcp = std::make_shared<std::vector<RealT>>(numMyAtoms);
    std::shared_ptr<std::vector<RealT> > pW_rcp = std::make_shared<std::vector<RealT>>(numMyAtoms);
    for (int i = 0; i < numMyAtoms; ++i) {
      (*p1_rcp)[i] = static_cast<RealT>(rand())/static_cast<RealT>(RAND_MAX);
      (*p2_rcp)[i] = static_cast<RealT>(rand())/static_cast<RealT>(RAND_MAX);
      (*p3_rcp)[i] = static_cast<RealT>(rand())/static_cast<RealT>(RAND_MAX);
      (*pW_rcp)[i] = static_cast<RealT>(2);
    }
    std::shared_ptr<ROL::PrimalProbabilityVector<RealT> > p1
      = std::make_shared<ROL::PrimalProbabilityVector<RealT>>(p1_rcp,bman,pW_rcp);
    std::shared_ptr<ROL::PrimalProbabilityVector<RealT> > p2
      = std::make_shared<ROL::PrimalProbabilityVector<RealT>>(p2_rcp,bman,pW_rcp);
    std::shared_ptr<ROL::PrimalProbabilityVector<RealT> > p3
      = std::make_shared<ROL::PrimalProbabilityVector<RealT>>(p3_rcp,bman,pW_rcp);

    // Create SROM vectors
    ROL::SROMVector<RealT> x1(p1,a1);
    ROL::SROMVector<RealT> x2(p2,a2);
    ROL::SROMVector<RealT> x3(p3,a3);

    // Standard tests.
    std::vector<RealT> consistencyBMAN = b1->checkVector(*b2, *b3, true, outStream);
    ROL::StdVector<RealT> checkvecBMAN(&consistencyBMAN, false);
    if (checkvecBMAN.norm() > std::sqrt(errtol)) {
      errorFlag++;
    }
    std::vector<RealT> consistencyAtom = a1->checkVector(*a2, *a3, true, outStream);
    ROL::StdVector<RealT> checkvecAtom(&consistencyAtom, false);
    if (checkvecAtom.norm() > std::sqrt(errtol)) {
      errorFlag++;
    }
    std::vector<RealT> consistencyProb = p1->checkVector(*p2, *p3, true, outStream);
    ROL::StdVector<RealT> checkvecProb(&consistencyProb, false);
    if (checkvecProb.norm() > std::sqrt(errtol)) {
      errorFlag++;
    }
    std::vector<RealT> consistencySROM = x1.checkVector(x2, x3, true, outStream);
    ROL::StdVector<RealT> checkvecSROM(&consistencySROM, false);
    if (checkvecSROM.norm() > std::sqrt(errtol)) {
      errorFlag++;
    }

    RealT numProcs = static_cast<RealT>(Teuchos::size<int>(*comm));
    RealT anorm = std::sqrt(numProcs*size), pnorm = std::sqrt(numProcs*numMyAtoms);
    RealT norm = std::sqrt(anorm*anorm + pnorm*pnorm);
    RealT sqrt2 = static_cast<RealT>(std::sqrt(2.));

    // Create batch std vectors 
    std::shared_ptr<std::vector<RealT> > b_rcp = std::make_shared<std::vector<RealT>>(size,1);
    std::shared_ptr<ROL::BatchStdVector<RealT> > b
      = std::make_shared<ROL::BatchStdVector<RealT>>(b_rcp,bman);
    RealT bnorm = b->norm();
    outStream << "BatchStdVector Norm Error:          "
              << std::abs(bnorm - anorm) << std::endl;
    if ( std::abs(bnorm - anorm) > std::sqrt(errtol) ) {
      errorFlag++;
    }

    // Create atom vectors 
    std::shared_ptr<std::vector<RealT> > ap_rcp = std::make_shared<std::vector<RealT>>(size,1);
    std::shared_ptr<ROL::PrimalAtomVector<RealT> > ap
      = std::make_shared<ROL::PrimalAtomVector<RealT>>(ap_rcp,bman,numMyAtoms,dimension,aW_rcp);
    RealT apnorm = ap->norm();
    outStream << "PrimalAtomVector Norm Error:        "
              << std::abs(apnorm - sqrt2*anorm) << std::endl;
    if ( std::abs(apnorm - sqrt2*anorm) > std::sqrt(errtol) ) {
      errorFlag++;
    }
    std::shared_ptr<std::vector<RealT> > ad_rcp = std::make_shared<std::vector<RealT>>(size,1);
    std::shared_ptr<ROL::DualAtomVector<RealT> > ad
      = std::make_shared<ROL::DualAtomVector<RealT>>(ad_rcp,bman,numMyAtoms,dimension,aW_rcp);
    RealT adnorm = ad->norm();
    outStream << "DualAtomVector Norm Error:          "
              << std::abs(adnorm - anorm/sqrt2) << std::endl;
    if ( std::abs(adnorm - anorm/sqrt2) > std::sqrt(errtol) ) {
      errorFlag++;
    }

    // Create probability vectors
    std::shared_ptr<std::vector<RealT> > pp_rcp = std::make_shared<std::vector<RealT>>(numMyAtoms,1);
    std::shared_ptr<ROL::PrimalProbabilityVector<RealT> > pp
      = std::make_shared<ROL::PrimalProbabilityVector<RealT>>(pp_rcp,bman,pW_rcp);
    RealT ppnorm = pp->norm();
    outStream << "PrimalProbabilityVector Norm Error: "
              << std::abs(ppnorm - sqrt2*pnorm) << std::endl;
    if ( std::abs(ppnorm - sqrt2*pnorm) > std::sqrt(errtol) ) {
      errorFlag++;
    }
    std::shared_ptr<std::vector<RealT> > pd_rcp = std::make_shared<std::vector<RealT>>(numMyAtoms,1);
    std::shared_ptr<ROL::DualProbabilityVector<RealT> > pd
      = std::make_shared<ROL::DualProbabilityVector<RealT>>(pd_rcp,bman,pW_rcp);
    RealT pdnorm = pd->norm();
    outStream << "DualProbabilityVector Norm Error:   "
              << std::abs(pdnorm - pnorm/sqrt2) << std::endl;
    if ( std::abs(pdnorm - pnorm/sqrt2) > std::sqrt(errtol) ) {
      errorFlag++;
    }
    
    // Create SROM vectors
    ROL::SROMVector<RealT> xp(pp,ap);
    RealT xpnorm = xp.norm();
    outStream << "PrimalSROMVector Norm Error:        "
              << std::abs(xpnorm - sqrt2*norm) << std::endl;
    if ( std::abs(xpnorm - sqrt2*norm) > std::sqrt(errtol) ) {
      errorFlag++;
    }
    ROL::SROMVector<RealT> xd(pd,ad);
    RealT xdnorm = xd.norm();
    outStream << "DualSROMVector Norm Error:          "
              << std::abs(xdnorm - norm/sqrt2) << std::endl;
    if ( std::abs(xdnorm - norm/sqrt2) > std::sqrt(errtol) ) {
      errorFlag++;
    }
    outStream << std::endl;
  }

  catch (std::logic_error err) {
    outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;
}
