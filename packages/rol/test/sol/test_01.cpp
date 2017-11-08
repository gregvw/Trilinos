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

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "ROL_StdVector.hpp"
#include "ROL_StdBoundConstraint.hpp"
#include "ROL_Types.hpp"
#include "ROL_Algorithm.hpp"

#include "ROL_OptimizationProblem.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BatchManager.hpp"
#include "ROL_MonteCarloGenerator.hpp"

typedef double RealT;

template<class Real>
class ParametrizedObjectiveEx1 : public ROL::Objective<Real> {
public:
  Real value( const ROL::Vector<Real> &x, Real &tol ) {
    std::shared_ptr<const std::vector<Real> > ex =
      (dynamic_cast<ROL::StdVector<Real>&>(const_cast<ROL::Vector<Real>&>(x))).getVector();
    Real quad = 0.0, lin = 0.0;
    std::vector<Real> p = this->getParameter();
    unsigned size = ex->size();
    for ( unsigned i = 0; i < size; i++ ) {
      quad += (*ex)[i]*(*ex)[i];
      lin  += (*ex)[i]*p[i+1];
    }
    return std::exp(p[0])*quad + lin + p[size+1];
  }

  void gradient( ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real &tol ) {
    std::shared_ptr<const std::vector<Real> > ex =
      (dynamic_cast<ROL::StdVector<Real>&>(const_cast<ROL::Vector<Real>&>(x))).getVector();
    std::shared_ptr<std::vector<Real> > eg =
      std::const_pointer_cast<std::vector<Real> >((dynamic_cast<ROL::StdVector<Real>&>(g)).getVector());
    std::vector<Real> p = this->getParameter();
    unsigned size = ex->size();
    for ( unsigned i = 0; i < size; i++ ) {
      (*eg)[i] = 2.0*std::exp(p[0])*(*ex)[i] + p[i+1];
    }
  }

  void hessVec( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real &tol ) {
    std::shared_ptr<const std::vector<Real> > ex =
      (dynamic_cast<ROL::StdVector<Real>&>(const_cast<ROL::Vector<Real>&>(x))).getVector();
    std::shared_ptr<const std::vector<Real> > ev =
      (dynamic_cast<ROL::StdVector<Real>&>(const_cast<ROL::Vector<Real>&>(v))).getVector();
    std::shared_ptr<std::vector<Real> > ehv =
      std::const_pointer_cast<std::vector<Real> >((dynamic_cast<ROL::StdVector<Real>&>(hv)).getVector());
    std::vector<Real> p = this->getParameter();
    unsigned size = ex->size();
    for ( unsigned i = 0; i < size; i++ ) {
      (*ehv)[i] = 2.0*std::exp(p[0])*(*ev)[i];
    }
  }
};

void setUpAndSolve(Teuchos::ParameterList &list,
                   std::shared_ptr<ROL::Objective<RealT> > &pObj,
                   std::shared_ptr<ROL::SampleGenerator<RealT> > &sampler,
                   std::shared_ptr<ROL::Vector<RealT> > &x,
                   std::shared_ptr<ROL::BoundConstraint<RealT> > &bnd,
                   std::ostream & outStream) {
  ROL::OptimizationProblem<RealT> opt(pObj,x,bnd);
  opt.setStochasticObjective(list,sampler);
  outStream << "\nCheck Derivatives of Stochastic Objective Function\n";
  opt.check(outStream);
  // Run ROL algorithm
  ROL::Algorithm<RealT> algo("Trust Region",list,false);
  algo.run(opt,true,outStream);
}

void setRandomVector(std::vector<RealT> &x) {
  unsigned dim = x.size();
  for ( unsigned i = 0; i < dim; i++ ) {
    x[i] = (RealT)rand()/(RealT)RAND_MAX;
  }
}

void printSolution(const std::vector<RealT> &x,
                   std::ostream & outStream) {
  unsigned dim = x.size();
  outStream << "x = (";
  for ( unsigned i = 0; i < dim-1; i++ ) {
    outStream << x[i] << ", ";
  }
  outStream << x[dim-1] << ")\n";
}

int main(int argc, char* argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  std::ostream* outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = &std::cout;
  else
    outStream = &bhs;

  int errorFlag  = 0;

  try {
    /**********************************************************************************************/
    /************************* CONSTRUCT ROL ALGORITHM ********************************************/
    /**********************************************************************************************/
    // Get ROL parameterlist
    const std::string filename = "input_01.xml";
    auto parlist = Teuchos::getParametersFromXmlFile(filename);
    Teuchos::ParameterList list = *parlist;
    /**********************************************************************************************/
    /************************* CONSTRUCT SOL COMPONENTS *******************************************/
    /**********************************************************************************************/
    // Build vectors
    unsigned dim = 4;
    std::shared_ptr<std::vector<RealT> > x_rcp = std::make_shared<std::vector<RealT>>(dim,0.0);
    std::shared_ptr<ROL::Vector<RealT> > x = std::make_shared<ROL::StdVector<RealT>>(x_rcp);
    std::shared_ptr<std::vector<RealT> > d_rcp = std::make_shared<std::vector<RealT>>(dim,0.0);
    std::shared_ptr<ROL::Vector<RealT> > d = std::make_shared<ROL::StdVector<RealT>>(d_rcp);
    setRandomVector(*d_rcp);
    // Build samplers
    int nSamp = 1000;
    unsigned sdim = dim + 2;
    std::vector<RealT> tmp(2,0.); tmp[0] = -1.; tmp[1] = 1.;
    std::vector<std::vector<RealT> > bounds(sdim,tmp);
    std::shared_ptr<ROL::BatchManager<RealT> > bman =
      std::make_shared<ROL::BatchManager<RealT>>();
    std::shared_ptr<ROL::SampleGenerator<RealT> > sampler =
      std::make_shared<ROL::MonteCarloGenerator<RealT>>(nSamp,bounds,bman,false,false,100);
    // Build risk-averse objective function
    std::shared_ptr<ROL::Objective<RealT> > pObj =
      std::make_shared<ParametrizedObjectiveEx1<RealT>>();
    // Build bound constraints
    std::vector<RealT> l(dim,0.0);
    std::vector<RealT> u(dim,1.0);
    std::shared_ptr<ROL::BoundConstraint<RealT> > bnd =
      std::make_shared<ROL::StdBoundConstraint<RealT>>(l,u);
    bnd->deactivate();
    // Test parametrized objective functions
    *outStream << "Check Derivatives of Parametrized Objective Function\n";
    pObj->setParameter(sampler->getMyPoint(0));
    pObj->checkGradient(*x,*d,true,*outStream);
    pObj->checkHessVec(*x,*d,true,*outStream);
    /**********************************************************************************************/
    /************************* MEAN VALUE *********************************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN VALUE\n";
    list.sublist("SOL").set("Stochastic Component Type","Mean Value");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* RISK NEUTRAL *******************************************************/
    /**********************************************************************************************/
    *outStream << "\nRISK NEUTRAL\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Neutral");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS DEVIATION ************************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS DEVIATION\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mean Plus Deviation");
    list.sublist("SOL").sublist("Risk Measure").sublist("Mean Plus Deviation").set("Deviation Type","Absolute");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS VARIANCE *************************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS VARIANCE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mean Plus Variance");
    list.sublist("SOL").sublist("Risk Measure").sublist("Mean Plus Variance").set("Deviation Type","Absolute");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS DEVIATION FROM TARGET ************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS DEVIATION FROM TARGET\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mean Plus Deviation From Target");
    list.sublist("SOL").sublist("Risk Measure").sublist("Mean Plus Deviation From Target").set("Deviation Type","Absolute");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS VARIANCE FROM TARGET *************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS VARIANCE FROM TARGET\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mean Plus Variance From Target");
    list.sublist("SOL").sublist("Risk Measure").sublist("Mean Plus Variance From Target").set("Deviation Type","Absolute");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS SEMIDEVIATION ********************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS SEMIDEVIATION\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mean Plus Deviation");
    list.sublist("SOL").sublist("Risk Measure").sublist("Mean Plus Deviation").set("Deviation Type","Upper");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS SEMIVARIANCE *********************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS SEMIVARIANCE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mean Plus Variance");
    list.sublist("SOL").sublist("Risk Measure").sublist("Mean Plus Variance").set("Deviation Type","Upper");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS SEMIDEVIATION FROM TARGET ********************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS SEMIDEVIATION FROM TARGET\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mean Plus Deviation From Target");
    list.sublist("SOL").sublist("Risk Measure").sublist("Mean Plus Deviation From Target").set("Deviation Type","Upper");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS SEMIVARIANCE FROM TARGET *********************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS SEMIVARIANCE FROM TARGET\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mean Plus Variance From Target");
    list.sublist("SOL").sublist("Risk Measure").sublist("Mean Plus Variance From Target").set("Deviation Type","Upper");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS CVAR *****************************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS CONDITIONAL VALUE AT RISK\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","CVaR");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN PLUS HMCR *****************************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN PLUS HIGHER MOMENT COHERENT RISK\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","HMCR");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* SMOOTHED CVAR QUADRANGLE *******************************************/
    /**********************************************************************************************/
    *outStream << "\nQUANTILE-BASED QUADRANGLE RISK MEASURE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Quantile-Based Quadrangle");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MIXED-QUANTILE QUADRANGLE ******************************************/
    /**********************************************************************************************/
    *outStream << "\nMIXED-QUANTILE QUADRANGLE RISK MEASURE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mixed-Quantile Quadrangle");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* SUPER QUANTILE QUADRANGLE ******************************************/
    /**********************************************************************************************/
    *outStream << "\nSUPER QUANTILE QUADRANGLE RISK MEASURE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Super Quantile Quadrangle");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* CHEBYSHEV 1 KUSUOKA ************************************************/
    /**********************************************************************************************/
    *outStream << "\nCHEBYSHEV 1 KUSUOKA RISK MEASURE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Chebyshev-Kusuoka");
    list.sublist("SOL").sublist("Risk Measure").sublist("Chebyshev-Kusuoka").set("Weight Type",1);
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* CHEBYSHEV 2 KUSUOKA ************************************************/
    /**********************************************************************************************/
    *outStream << "\nCHEBYSHEV 2 KUSUOKA RISK MEASURE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Chebyshev-Kusuoka");
    list.sublist("SOL").sublist("Risk Measure").sublist("Chebyshev-Kusuoka").set("Weight Type",2);
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* CHEBYSHEV 3 KUSUOKA ************************************************/
    /**********************************************************************************************/
    *outStream << "\nCHEBYSHEV 3 KUSUOKA RISK MEASURE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Chebyshev-Kusuoka");
    list.sublist("SOL").sublist("Risk Measure").sublist("Chebyshev-Kusuoka").set("Weight Type",3);
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* CHEBYSHEV 3 KUSUOKA ************************************************/
    /**********************************************************************************************/
    *outStream << "\nSPECTRAL RISK MEASURE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Spectral Risk");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* MEAN-VARIANCE QUADRANGLE *******************************************/
    /**********************************************************************************************/
    *outStream << "\nMEAN-VARIANCE QUADRANGLE RISK MEASURE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Mean-Variance Quadrangle");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* QUANTILE-RANGE QUADRANGLE ******************************************/
    /**********************************************************************************************/
    *outStream << "\nQUANTILE-RADIUS QUADRANGLE RISK MEASURE\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Quantile-Radius Quadrangle");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* CHI-SQUARED DIVERGENCE *********************************************/
    /**********************************************************************************************/
    *outStream << "\nCHI-SQUARED DIVERGENCE DISTRIBUTIONALLY ROBUST\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Chi-Squared Divergence");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* KL DIVERGENCE ******************************************************/
    /**********************************************************************************************/
    *outStream << "\nKL DIVERGENCE DISTRIBUTIONALLY ROBUST\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","KL Divergence");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* COHERENT EXPONENTIAL UTILITY FUNCTION ******************************/
    /**********************************************************************************************/
    *outStream << "\nCOHERENT EXPONENTIAL UTILITY FUNCTION\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Coherent Exponential Utility");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* EXPONENTIAL UTILITY FUNCTION ***************************************/
    /**********************************************************************************************/
    *outStream << "\nEXPONENTIAL UTILITY FUNCTION\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Exponential Utility");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
    /**********************************************************************************************/
    /************************* CONVEX COMBINATION OF RISK MEASURES ********************************/
    /**********************************************************************************************/
    *outStream << "\nCONVEX COMBINATION OF RISK MEASURES\n";
    list.sublist("SOL").set("Stochastic Component Type","Risk Averse");
    list.sublist("SOL").sublist("Risk Measure").set("Name","Convex Combination Risk Measure");
    setRandomVector(*x_rcp);
    setUpAndSolve(list,pObj,sampler,x,bnd,*outStream);
    printSolution(*x_rcp,*outStream);
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
