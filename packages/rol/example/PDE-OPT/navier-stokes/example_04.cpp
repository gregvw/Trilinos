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

/*! \file  example_04.cpp
    \brief Shows how to solve the stochastic Navier-Stokes problem.
*/

#include "Teuchos_Comm.hpp"
#ifdef HAVE_MPI
#include "Teuchos_DefaultMpiComm.hpp"
#endif
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Version.hpp"

#include <iostream>
#include <algorithm>
//#include <fenv.h>

#include "ROL_Algorithm.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_MonteCarloGenerator.hpp"
#include "ROL_OptimizationProblem.hpp"
#include "ROL_TpetraTeuchosBatchManager.hpp"

#include "../TOOLS/meshmanager.hpp"
#include "../TOOLS/pdeconstraint.hpp"
#include "../TOOLS/pdeobjective.hpp"
#include "../TOOLS/pdevector.hpp"
#include "../TOOLS/batchmanager.hpp"
#include "pde_navier-stokes.hpp"
#include "obj_navier-stokes.hpp"

typedef double RealT;

template<class Real>
void setUpAndSolve(ROL::OptimizationProblem<Real> &opt,
                   Teuchos::ParameterList &parlist,
                   std::ostream &outStream) {
  ROL::Algorithm<RealT> algo("Trust Region",parlist,false);
  Teuchos::Time timer("Optimization Time", true);
  algo.run(opt,true,outStream);
  timer.stop();
  outStream << "Total optimization time = " << timer.totalElapsedTime() << " seconds." << std::endl;
}

template<class Real>
void print(ROL::Objective<Real> &obj,
           const ROL::Vector<Real> &z,
           ROL::SampleGenerator<Real> &sampler,
           const int ngsamp,
           const std::shared_ptr<const Teuchos::Comm<int> > &comm,
           const std::string &filename) {
  Real tol(1e-8);
  // Build objective function distribution
  int nsamp = sampler.numMySamples();
  std::vector<Real> myvalues(nsamp), myzerovec(nsamp, 0);
  std::vector<double> gvalues(ngsamp), gzerovec(ngsamp, 0);
  std::vector<Real> sample = sampler.getMyPoint(0);
  int sdim = sample.size();
  std::vector<std::vector<Real> > mysamples(sdim, myzerovec);
  std::vector<std::vector<double> > gsamples(sdim, gzerovec);
  for (int i = 0; i < nsamp; ++i) {
    sample = sampler.getMyPoint(i);
    obj.setParameter(sample);
    myvalues[i] = static_cast<double>(obj.value(z,tol));
    for (int j = 0; j < sdim; ++j) {
      mysamples[j][i] = static_cast<double>(sample[j]);
    }
  }

  // Send data to root processor
#ifdef HAVE_MPI
  std::shared_ptr<const Teuchos::MpiComm<int> > mpicomm
    = std::dynamic_pointer_cast<const Teuchos::MpiComm<int> >(comm);
  int nproc = Teuchos::size<int>(*mpicomm);
  std::vector<int> sampleCounts(nproc, 0), sampleDispls(nproc, 0);
  MPI_Gather(&nsamp,1,MPI_INT,&sampleCounts[0],1,MPI_INT,0,*(mpicomm->getRawMpiComm())());
  for (int i = 1; i < nproc; ++i) {
    sampleDispls[i] = sampleDispls[i-1] + sampleCounts[i-1];
  }
  MPI_Gatherv(&myvalues[0],nsamp,MPI_DOUBLE,&gvalues[0],&sampleCounts[0],&sampleDispls[0],MPI_DOUBLE,0,*(mpicomm->getRawMpiComm())());
  for (int j = 0; j < sdim; ++j) {
    MPI_Gatherv(&mysamples[j][0],nsamp,MPI_DOUBLE,&gsamples[j][0],&sampleCounts[0],&sampleDispls[0],MPI_DOUBLE,0,*(mpicomm->getRawMpiComm())());
  }
#else
  gvalues.assign(myvalues.begin(),myvalues.end());
  for (int j = 0; j < sdim; ++j) {
    gsamples[j].assign(mysamples[j].begin(),mysamples[j].end());
  }
#endif

  // Print
  int rank  = Teuchos::rank<int>(*comm);
  if ( rank==0 ) {
    std::ofstream file;
    file.open(filename);
    file << std::scientific << std::setprecision(15);
    for (int i = 0; i < ngsamp; ++i) {
      for (int j = 0; j < sdim; ++j) {
        file << std::setw(25) << std::left << gsamples[j][i];
      }
      file << std::setw(25) << std::left << gvalues[i] << std::endl;
    }
    file.close();
  }
}

int main(int argc, char *argv[]) {
//  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing

  /*** Initialize communicator. ***/
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &bhs);
  std::shared_ptr<const Teuchos::Comm<int> > comm
    = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  std::shared_ptr<const Teuchos::Comm<int> > serial_comm
    = std::make_shared<Teuchos::SerialComm<int>>();
  const int myRank = comm->getRank();
  if ((iprint > 0) && (myRank == 0)) {
    outStream = &std::cout, false;
  }
  else {
    outStream = &bhs, false;
  }
  int errorFlag  = 0;

  // *** Example body.
  try {

    /*** Read in XML input ***/
    std::string filename = "input_ex04.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );

    // Problem dimensions
    const int stochDim = 2;
    const RealT one(1); 

    /*************************************************************************/
    /***************** BUILD GOVERNING PDE ***********************************/
    /*************************************************************************/
    /*** Initialize main data structure. ***/
    std::shared_ptr<MeshManager<RealT> > meshMgr
      = std::make_shared<MeshManager_BackwardFacingStepChannel<RealT>>(*parlist);
    // Initialize PDE describing advection-diffusion equation
    std::shared_ptr<PDE_NavierStokes<RealT> > pde
      = std::make_shared<PDE_NavierStokes<RealT>>(*parlist);
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > con
      = std::make_shared<PDE_Constraint<RealT>>(pde,meshMgr,serial_comm,*parlist,*outStream);
    // Cast the constraint and get the assembler.
    std::shared_ptr<PDE_Constraint<RealT> > pdecon
      = std::dynamic_pointer_cast<PDE_Constraint<RealT> >(con);
    std::shared_ptr<Assembler<RealT> > assembler = pdecon->getAssembler();
    assembler->printMeshData(*outStream);
    con->setSolveParameters(*parlist);

    /*************************************************************************/
    /***************** BUILD VECTORS *****************************************/
    /*************************************************************************/
    std::shared_ptr<Tpetra::MultiVector<> >  u_rcp = assembler->createStateVector();
    std::shared_ptr<Tpetra::MultiVector<> >  p_rcp = assembler->createStateVector();
    std::shared_ptr<Tpetra::MultiVector<> > du_rcp = assembler->createStateVector();
    u_rcp->randomize();  //u_rcp->putScalar(static_cast<RealT>(1));
    p_rcp->randomize();  //p_rcp->putScalar(static_cast<RealT>(1));
    du_rcp->randomize(); //du_rcp->putScalar(static_cast<RealT>(0));
    std::shared_ptr<ROL::Vector<RealT> > up
      = std::make_shared<PDE_PrimalSimVector<RealT>>(u_rcp,pde,assembler,*parlist);
    std::shared_ptr<ROL::Vector<RealT> > pp
      = std::make_shared<PDE_PrimalSimVector<RealT>>(p_rcp,pde,assembler,*parlist);
    std::shared_ptr<ROL::Vector<RealT> > dup
      = std::make_shared<PDE_PrimalSimVector<RealT>>(du_rcp,pde,assembler,*parlist);
    // Create residual vectors
    std::shared_ptr<Tpetra::MultiVector<> > r_rcp = assembler->createResidualVector();
    r_rcp->randomize(); //r_rcp->putScalar(static_cast<RealT>(1));
    std::shared_ptr<ROL::Vector<RealT> > rp
      = std::make_shared<PDE_DualSimVector<RealT>>(r_rcp,pde,assembler,*parlist);
    // Create control vector and set to ones
    std::shared_ptr<Tpetra::MultiVector<> >  z_rcp = assembler->createControlVector();
    std::shared_ptr<Tpetra::MultiVector<> > dz_rcp = assembler->createControlVector();
    std::shared_ptr<Tpetra::MultiVector<> > yz_rcp = assembler->createControlVector();
    z_rcp->randomize();  z_rcp->putScalar(static_cast<RealT>(0));
    dz_rcp->randomize(); //dz_rcp->putScalar(static_cast<RealT>(0));
    yz_rcp->randomize(); //yz_rcp->putScalar(static_cast<RealT>(0));
    std::shared_ptr<ROL::TpetraMultiVector<RealT> > zpde
      = std::make_shared<PDE_PrimalOptVector<RealT>>(z_rcp,pde,assembler,*parlist);
    std::shared_ptr<ROL::TpetraMultiVector<RealT> > dzpde
      = std::make_shared<PDE_PrimalOptVector<RealT>>(dz_rcp,pde,assembler,*parlist);
    std::shared_ptr<ROL::TpetraMultiVector<RealT> > yzpde
      = std::make_shared<PDE_PrimalOptVector<RealT>>(yz_rcp,pde,assembler,*parlist);
    std::shared_ptr<ROL::Vector<RealT> > zp  = std::make_shared<PDE_OptVector<RealT>>(zpde);
    std::shared_ptr<ROL::Vector<RealT> > dzp = std::make_shared<PDE_OptVector<RealT>>(dzpde);
    std::shared_ptr<ROL::Vector<RealT> > yzp = std::make_shared<PDE_OptVector<RealT>>(yzpde);
    // Create ROL SimOpt vectors
    ROL::Vector_SimOpt<RealT> x(up,zp);
    ROL::Vector_SimOpt<RealT> d(dup,dzp);

    /*************************************************************************/
    /***************** BUILD COST FUNCTIONAL *********************************/
    /*************************************************************************/
    std::vector<std::shared_ptr<QoI<RealT> > > qoi_vec(2,nullptr);
    qoi_vec[0] = Teuchos::rcp(new QoI_State_NavierStokes<RealT>(*parlist,
                                                                pde->getVelocityFE(),
                                                                pde->getPressureFE(),
                                                                pde->getFieldHelper()));
    qoi_vec[1] = std::make_shared<QoI_L2Penalty_NavierStokes<RealT>(pde->getVelocityFE(>(),
                                                                    pde->getPressureFE(),
                                                                    pde->getVelocityBdryFE(),
                                                                    pde->getBdryCellLocIds(),
                                                                    pde->getFieldHelper()));
    std::shared_ptr<StdObjective_NavierStokes<RealT> > std_obj
      = std::make_shared<StdObjective_NavierStokes<RealT>>(*parlist);
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > obj
      = std::make_shared<PDE_Objective<RealT>>(qoi_vec,std_obj,assembler);
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > objState
      = std::make_shared<PDE_Objective<RealT>>(qoi_vec[0],assembler);
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > objCtrl
      = std::make_shared<PDE_Objective<RealT>>(qoi_vec[1],assembler);
    std::shared_ptr<ROL::Reduced_Objective_SimOpt<RealT> > objRed
      = std::make_shared<ROL::Reduced_Objective_SimOpt<RealT>>(obj, con, up, zp, pp, true, false);

    /*************************************************************************/
    /***************** BUILD BOUND CONSTRAINT ********************************/
    /*************************************************************************/
    std::shared_ptr<Tpetra::MultiVector<> >  zlo_rcp = assembler->createControlVector();
    std::shared_ptr<Tpetra::MultiVector<> >  zhi_rcp = assembler->createControlVector();
    zlo_rcp->putScalar(static_cast<RealT>(0));
    zhi_rcp->putScalar(ROL::ROL_INF<RealT>());
    std::shared_ptr<ROL::TpetraMultiVector<RealT> > zlopde
      = std::make_shared<PDE_PrimalOptVector<RealT>>(zlo_rcp,pde,assembler,*parlist);
    std::shared_ptr<ROL::TpetraMultiVector<RealT> > zhipde
      = std::make_shared<PDE_PrimalOptVector<RealT>>(zhi_rcp,pde,assembler,*parlist);
    std::shared_ptr<ROL::Vector<RealT> > zlop = std::make_shared<PDE_OptVector<RealT>>(zlopde);
    std::shared_ptr<ROL::Vector<RealT> > zhip = std::make_shared<PDE_OptVector<RealT>>(zhipde);
    std::shared_ptr<ROL::BoundConstraint<RealT> > bnd
      = std::make_shared<ROL::Bounds<RealT>>(zlop,zhip);
    bool useBounds = parlist->sublist("Problem").get("Use bounds", false);
    if (!useBounds) {
      bnd->deactivate();
    }

    /*************************************************************************/
    /***************** BUILD SAMPLER *****************************************/
    /*************************************************************************/
    int nsamp = parlist->sublist("Problem").get("Number of samples",100);
    int nsamp_dist = parlist->sublist("Problem").get("Number of output samples",100);
    std::vector<RealT> tmp = {-one,one};
    std::vector<std::vector<RealT> > bounds(stochDim,tmp);
    std::shared_ptr<ROL::BatchManager<RealT> > bman
      = std::make_shared<PDE_OptVector_BatchManager<RealT>>(comm);
    std::shared_ptr<ROL::SampleGenerator<RealT> > sampler
      = std::make_shared<ROL::MonteCarloGenerator<RealT>>(nsamp,bounds,bman);
    std::shared_ptr<ROL::SampleGenerator<RealT> > sampler_dist
      = std::make_shared<ROL::MonteCarloGenerator<RealT>>(nsamp_dist,bounds,bman);

    /*************************************************************************/
    /***************** BUILD STOCHASTIC PROBLEM ******************************/
    /*************************************************************************/
    std::shared_ptr<ROL::OptimizationProblem<RealT> > opt;
    std::vector<RealT> ctrl;
    std::vector<RealT> var;

    Teuchos::Array<RealT> alphaArray
      = Teuchos::getArrayFromStringParameter<RealT>(parlist->sublist("Problem"),"bPOE Thresholds");
    std::vector<RealT> alpha = alphaArray.toVector();
    std::sort(alpha.begin(),alpha.end());
    int N = alpha.size();

    /*************************************************************************/
    /***************** SOLVE MEAN PLUS CVAR **********************************/
    /*************************************************************************/
    RealT tol(1e-8);
    parlist->sublist("SOL").set("Stochastic Component Type","Risk Averse");
    parlist->sublist("SOL").sublist("Risk Measure").set("Name","bPOE");
    parlist->sublist("SOL").sublist("Risk Measure").sublist("bPOE").set("Moment Order",2.0);
    for (int i = 0; i < N; ++i) {
      // Solve.
      parlist->sublist("SOL").sublist("Risk Measure").sublist("bPOE").set("Threshold",alpha[i]);
      opt = std::make_shared<ROL::OptimizationProblem<RealT>>(objRed,zp,bnd);
      RealT stat(1);
      if ( i > 0 ) {
        stat = var[i];
      }
      parlist->sublist("SOL").set("Initial Statistic",stat);
      opt->setStochasticObjective(*parlist,sampler);
      setUpAndSolve<RealT>(*opt,*parlist,*outStream);
      // Output.
      ctrl.push_back(objCtrl->value(*up,*zp,tol));
      var.push_back(opt->getSolutionStatistic());
      std::stringstream nameCtrl;
      nameCtrl << "control_bPOE_" << i+1 << ".txt";
      pdecon->outputTpetraVector(z_rcp,nameCtrl.str().c_str());
      std::stringstream nameObj;
      nameObj << "obj_samples_bPOE_" << i+1 << ".txt";
      print<RealT>(*objRed,*zp,*sampler_dist,nsamp_dist,comm,nameObj.str());
    }

    /*************************************************************************/
    /***************** PRINT CONTROL OBJ AND VAR *****************************/
    /*************************************************************************/
    const int rank = Teuchos::rank<int>(*comm);
    if ( rank==0 ) {
      std::stringstream nameCTRL, nameVAR;
      nameCTRL << "ctrl.txt";
      nameVAR << "var.txt";
      std::ofstream fileCTRL, fileVAR;
      fileCTRL.open(nameCTRL.str());
      fileVAR.open(nameVAR.str());
      fileCTRL << std::scientific << std::setprecision(15);
      fileVAR << std::scientific << std::setprecision(15);
      int size = var.size();
      for (int i = 0; i < size; ++i) {
        fileCTRL << std::setw(25) << std::left << ctrl[i] << std::endl;
        fileVAR << std::setw(25) << std::left << var[i] << std::endl;
      }
      fileCTRL.close();
      fileVAR.close();
    }

    // Get a summary from the time monitor.
    Teuchos::TimeMonitor::summarize();
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
