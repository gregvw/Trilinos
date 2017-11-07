
// Burgers includes
#include "example_02.hpp"
// ROL includes
#include "ROL_Algorithm.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_StdTeuchosBatchManager.hpp"
#include "ROL_MonteCarloGenerator.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_RiskNeutralObjective.hpp"
#include "ROL_Vector_SimOpt.hpp"
#include "ROL_Bounds.hpp"
// Teuchos includes
#include "Teuchos_Time.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_CommHelpers.hpp"

int main( int argc, char *argv[] ) {  

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  std::shared_ptr<const Teuchos::Comm<int> > comm =
    Teuchos::DefaultComm<int>::getComm();

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

    /***************************************************************************/
    /***************** GRAB INPUTS *********************************************/
    /***************************************************************************/
    // Get finite element parameter list
    std::string filename = "example_02.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*parlist) );
    if ( parlist->get("Display Option",0) && (comm->getRank() > 0) ) {
      parlist->set("Display Option",0);
    }
    // Get ROL parameter list
    filename = "input.xml";
    std::shared_ptr<Teuchos::ParameterList> ROL_parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*ROL_parlist) );
  
    /***************************************************************************/
    /***************** INITIALIZE SAMPLERS *************************************/
    /***************************************************************************/
    int dim    = 2;
    bool useSA = parlist->get("Use Stochastic Approximation",false);
    int nSamp  = 1;
    if ( !useSA ) {
      nSamp  = parlist->get("Number of Monte Carlo Samples",1000);
    }
    std::vector<double> tmp(2); tmp[0] = -1.0; tmp[1] = 1.0;
    std::vector<std::vector<double> > bounds(dim,tmp);
    std::shared_ptr<ROL::BatchManager<double> > bman
      = std::make_shared<ROL::StdTeuchosBatchManager<double,int>>(comm);
    std::shared_ptr<ROL::SampleGenerator<double> > sampler
      = std::make_shared<ROL::MonteCarloGenerator<double>>(nSamp,bounds,bman,useSA);
  
    /***************************************************************************/
    /***************** INITIALIZE CONTROL VECTOR *******************************/
    /***************************************************************************/
    int nx = parlist->get("Number of Elements", 128);
    std::shared_ptr<std::vector<double> > z_rcp = std::make_shared<std::vector<double>>(nx+1, 0.0);
    std::shared_ptr<ROL::Vector<double> > z = std::make_shared<ROL::StdVector<double>>(z_rcp);
    std::shared_ptr<std::vector<double> > u_rcp = std::make_shared<std::vector<double>>(nx-1, 0.0);
    std::shared_ptr<ROL::Vector<double> > u = std::make_shared<ROL::StdVector<double>>(u_rcp);
    ROL::Vector_SimOpt<double> x(u,z);
    std::shared_ptr<std::vector<double> > p_rcp = std::make_shared<std::vector<double>>(nx-1, 0.0);
    std::shared_ptr<ROL::Vector<double> > p = std::make_shared<ROL::StdVector<double>>(p_rcp);
    std::shared_ptr<std::vector<double> > U_rcp = std::make_shared<std::vector<double>>(nx+1, 35.0);
    std::shared_ptr<ROL::Vector<double> > U = std::make_shared<ROL::StdVector<double>>(U_rcp);
    std::shared_ptr<std::vector<double> > L_rcp = std::make_shared<std::vector<double>>(nx+1, -5.0);
    std::shared_ptr<ROL::Vector<double> > L = std::make_shared<ROL::StdVector<double>>(L_rcp);
    ROL::Bounds<double> bnd(L,U);
  
    /***************************************************************************/
    /***************** INITIALIZE OBJECTIVE FUNCTION ***************************/
    /***************************************************************************/
    double alpha = parlist->get("Penalty Parameter", 1.e-4);
    std::shared_ptr<FEM<double> > fem = std::make_shared<FEM<double>>(nx);
    std::shared_ptr<ROL::Objective_SimOpt<double> > pObj
      = std::make_shared<DiffusionObjective<double>>(fem, alpha);
    std::shared_ptr<ROL::Constraint_SimOpt<double> > pCon
      = std::make_shared<DiffusionConstraint<double>>(fem);
    std::shared_ptr<ROL::Objective<double> > robj
      = std::make_shared<ROL::Reduced_Objective_SimOpt<double>>(pObj,pCon,u,z,p);
    ROL::RiskNeutralObjective<double> obj(robj,sampler);
  
    /***************************************************************************/
    /***************** RUN DERIVATIVE CHECK ************************************/
    /***************************************************************************/
    if (parlist->get("Run Derivative Check",false)) {
      // Direction to test finite differences
      std::shared_ptr<std::vector<double> > dz_rcp = std::make_shared<std::vector<double>>(nx+1, 0.0);
      std::shared_ptr<ROL::Vector<double> > dz = std::make_shared<ROL::StdVector<double>>(dz_rcp);
      std::shared_ptr<std::vector<double> > du_rcp = std::make_shared<std::vector<double>>(nx-1, 0.0);
      std::shared_ptr<ROL::Vector<double> > du = std::make_shared<ROL::StdVector<double>>(du_rcp);
      ROL::Vector_SimOpt<double> d(du,dz);
      // Set to random vectors
      srand(12345);
      for (int i=0; i<nx+1; i++) {
        (*dz_rcp)[i] = 2.0*(double)rand()/(double)RAND_MAX - 1.0;
        (*z_rcp)[i] = 2.0*(double)rand()/(double)RAND_MAX - 1.0;
      }
      for (int i=0; i<nx-1; i++) {
        (*du_rcp)[i] = 2.0*(double)rand()/(double)RAND_MAX - 1.0;
        (*u_rcp)[i] = 2.0*(double)rand()/(double)RAND_MAX - 1.0;
      }
      // Run derivative checks
      std::vector<double> param(dim,0.0);
      robj->setParameter(param);
      if ( comm->getRank() == 0 ) {
        std::cout << "\nRUN DERIVATIVE CHECK FOR PARAMETRIZED OBJECTIVE FUNCTION SIMOPT\n";
      }
      pObj->checkGradient(x,d,(comm->getRank()==0));
      pObj->checkHessVec(x,d,(comm->getRank()==0));
      if ( comm->getRank() == 0 ) {
        std::cout << "\nRUN DERIVATIVE CHECK FOR PARAMETRIZED EQUALITY CONSTRAINT SIMOPT\n";
      }
      pCon->checkApplyJacobian(x,d,*p,(comm->getRank()==0));
      pCon->checkApplyAdjointJacobian(x,*du,*p,x,(comm->getRank()==0));
      pCon->checkApplyAdjointHessian(x,*du,d,x,(comm->getRank()==0));
      if ( comm->getRank() == 0 ) {
        std::cout << "\nRUN DERIVATIVE CHECK FOR PARAMETRIZED OBJECTIVE FUNCTION\n";
      }
      robj->checkGradient(*z,*dz,(comm->getRank()==0));
      robj->checkHessVec(*z,*dz,(comm->getRank()==0));
      // Run derivative checks
      if ( comm->getRank() == 0 ) {
        std::cout << "\nRUN DERIVATIVE CHECK FOR RISK-NEUTRAL OBJECTIVE FUNCTION\n";
      }
      obj.checkGradient(*z,*dz,(comm->getRank()==0));
      obj.checkHessVec(*z,*dz,(comm->getRank()==0));
    }
  
    /***************************************************************************/
    /***************** INITIALIZE ROL ALGORITHM ********************************/
    /***************************************************************************/
    std::shared_ptr<ROL::Algorithm<double> > algo; 
    if ( useSA ) {
      ROL_parlist->sublist("General").set("Recompute Objective Function",false);
      ROL_parlist->sublist("Step").sublist("Line Search").set("Initial Step Size",0.1/alpha);
      ROL_parlist->sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
      ROL_parlist->sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Iteration Scaling");
      ROL_parlist->sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Steepest Descent");
      ROL_parlist->sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Null Curvature Condition");
      algo = std::make_shared<ROL::Algorithm<double>>("Line Search",*ROL_parlist,false);
    } 
    else {
      algo = std::make_shared<ROL::Algorithm<double>>("Trust Region",*ROL_parlist,false);
    }
  
    /***************************************************************************/
    /***************** PERFORM OPTIMIZATION ************************************/
    /***************************************************************************/
    Teuchos::Time timer("Optimization Time",true);
    z->zero();
    algo->run(*z,obj,bnd,(comm->getRank()==0));
    double optTime = timer.stop();
  
    /***************************************************************************/
    /***************** PRINT RESULTS *******************************************/
    /***************************************************************************/
    int my_number_samples = sampler->numMySamples(), number_samples = 0;
    Teuchos::reduceAll<int,int>(*comm,Teuchos::REDUCE_SUM,1,&my_number_samples,&number_samples);
    int my_number_solves  = std::dynamic_pointer_cast<DiffusionConstraint<double> >(pCon)->getNumSolves(), number_solves = 0;
    Teuchos::reduceAll<int,int>(*comm,Teuchos::REDUCE_SUM,1,&my_number_solves,&number_solves);
    if (comm->getRank() == 0) {
      std::cout << "Number of Samples    = " << number_samples << "\n";
      std::cout << "Number of Solves     = " << number_solves  << "\n";
      std::cout << "Optimization Time    = " << optTime        << "\n\n";
    }
  
    if ( comm->getRank() == 0 ) {
      std::ofstream file;
      if (useSA) {
        file.open("control_SA.txt");
      }
      else {
        file.open("control_SAA.txt");
      }
      std::vector<double> xmesh(fem->nz(),0.0);
      fem->build_mesh(xmesh);
      for (int i = 0; i < fem->nz(); i++ ) {
        file << std::setprecision(std::numeric_limits<double>::digits10) << std::scientific << xmesh[i] << "  "  
             << std::setprecision(std::numeric_limits<double>::digits10) << std::scientific << (*z_rcp)[i] 
             << "\n";
      }
      file.close();
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




