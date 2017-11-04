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

#ifndef PYROL_OPTIMIZATION_PROBLEM_HPP
#define PYROL_OPTIMIZATION_PROBLEM_HPP

#ifdef __cplusplus 
extern "C" {
#endif

#ifndef Py_LIMITED_API


using OPT_PROB = ROL::OptimizationProblem<double>;

typedef struct {
  PyObject_HEAD
  Teuchos::RCP<OPT_PROB> optprob;
  static char *kwlist[] = 
    { // Detailed names
      C_TEXT("Objective"), 
      C_TEXT("Optimization Vector"),
      C_TEXT("Lower Bound Vector"),
      C_TEXT("Upper Bound Vector"), 
      C_TEXT("Equality Constraint"),
      C_TEXT("Equality Multiplier Vector"),
      C_TEXT("Inequality Constraint"), 
      C_TEXT("Inequality Multiplier Vector"),
      // Short names
      C_TEXT("obj"),
      C_TEXT("x"),
      C_TEXT("lo"),
      C_TEXT("up"),
      C_TEXT("eqcon"),
      C_TEXT("eqmul"),
      C_TEXT("incon"),
      C_TEXT("inmul") };
} PyOptimizationProblemObject;

PyAPI_DATA(PyTypeObject) PyOptimizationProblem_Type;

#endif // Py_LIMITED_API


#ifdef __cplusplus
} // extern "C"
#endif


static PyObject* 
OptimizationProblem_new(PyTypeObject *type, PyObject &args, PyObject *keywds) {
  PyOptimizationProblemObject *self;
  
  self = (PyOptimizationProblemObject*)type->tp_alloc(type,0);
  if( self != NULL) {

  }
  return (PyObject*)self;
}


static int 
OptimizationProblem_init(PyOptimizationProblemObject* self, 
                         PyObject* args, PyObject* keywds) {

  using Teuchos::RCP; using Teuchos::rcp;

  auto INF  = ROL::ROL_INF<double>();
  auto NINF = ROL::ROL_NINF<double>();
  
  ROL::Elementwise::Fill<double> Fill_INF(INF);
  ROL::Elementwise::Fill<double> Fill_NINF(NINF);

  // Required arguments
  PyObject* pyObjective;
  PyObject* pyX;

  // Optional (keyword) arguments
  PyObject* pyLo    = NULL;
  PyObject* pyUp    = NULL;
  PyObject* pyEqCon = NULL;
  PyObject* pyEqMul = NULL;
  PyObject* pyInCon = NULL;
  PyObject* pyInMul = NULL;

  int parseCheck = PyArg_ParseTupleAndKeywords(args,keywds,"OO|OOOOOO",kwlist,
    &pyObjective, &pyX, &pyLo, &pyUp, &pyEqCon, &pyEqMul, &incon, &inmul);
 
  TEUCHOS_TEST_FOR_EXCEPTION(!parseCheck,std::logic_error, "Failed to parse input tuple."
    << " Expected OptimizationProblem(obj,x) with optional keyword arguments lo, up, "
    << "eqcon, eqmul, incon, inmul.");

  auto obj = rcp( new PyROL::OptimizationProblem(pyObjective) );

  auto x = PyObject_AsVector(pyX);

  auto lo  = Teuchos::null;
  auto up  = Teuchos::null;
  auto bnd = Teuchos::null;

  auto eqcon = pyEqCon == NULL ? Teuchos::null : rcp( new PyROL::EqualityConstraint(pyEqCon) );
  auto eqmul = pyEqMul == NULL ? Teuchos::null : PyObject_AsVector(pyEqMul);

  auto incon = pyInCon == NULL ? Teuchos::null : rcp( new PyROL::InequalityConstraint(pyInCon) );
  auto inmul = pyInMul == NULL ? Teuchos::null : PyObject_AsVector(pyInMul);

  // Create bound constraint from upper and lower bound vectors
  if(lower != NULL) {
    lo = PyObject_AsVector(lower);
    if(upper != NULL) {
      up = PyObject_AsVector(upper);
    }
    else { // no upper bound
      up = x->clone();
      up->applyUnary(Fill_INF);
    }
    bnd = rcp( new ROL::BoundConstraint<double>(lo,up) );    
  }
  else { // no lower bound
    if( upper != NULL ) {
      up = PyObject_AsVector(upper);
      lo = x->clone();
      lo = applyUnary(Fill_NINF);
      bnd = rcp( new ROL::BoundConstraint<double>(lo,up) );
    }
  }

  opt_prob = rcp( new OPT_PROB(obj,x,bnd,eqcon,eqmul,incon,inmul) ); 

  return 0;
}


static PyObject*
OptimizationProblem_check( PyOptimizationProblem* pyOptProb ) {
  std::stringstream outputStream;
  PyFPE_START_PROTECT("OptimizationProblem_check", return 0)
  outputStream = pyOptProb->opt_prob->check(outputStream);
  PyFPE_END_PROTECT(outputStream);  
  PyObject* pyOutput = PyString_FromString(C_TEXT(outputStream.str()));
  return pyOutput;
}


static PyObject* 
OptimizationProblem_solve( PyOptimizationProblem* pyOptProb, PyDictObject* pyOptions ) {
  std::stringstream outputStream;
  Teuchos::ParameterList parlist;
  PyROL::dictToParameterList(pyOptions,parlist);
  ROL::OptimizationSolver<double> solver( pyOptProb->opt_prob, parlist);
  PyFPE_START_PROTECT("OptimizationProblem_solve", return 0)
  solver.solve( outputStream );
  PyFPE_END_PROTECT(outputStream);
  PyObject* pyOutput = PyString_FromString(C_TEXT(outputStream.str()));
  return pyOutput;
}


static PyMethodDef OptimizationProblem_methods[] = {
  {"check",(PyCFunction)OptimizationProblem_check, METH_VARARGS, "Validate linear algebra and derivatives"},
  {"solve",(PyCFunction)OptimizationProblem_solve, METH_VARARGS, "Solve the optimization problem"}
};



PyTypeObject PyOptimizationProblem_Type = {
#if PY_MAJOR_VERSION >= 3
  PyVarObject_HEAD_INIT(&PyType_Type,0)
#else
  PyObject_HEAD_INIT(NULL)
  "pyrol.OptimizationProblem",              /* tp_name           */
  sizeof(PyOptimizationProblemObject),      /* tp_basicsize      */
  0,                                        /* tp_itemsize       */
  0,                                        /* tp_dealloc        */
  0,                                        /* tp_print          */
  0,                                        /* tp_getattr        */
  0,                                        /* tp_setattr        */
  0,                                        /* tp_reserved       */
  0,                                        /* tp_repr           */
  0,                                        /* tp_as_number      */
  0,                                        /* tp_as_sequence    */
  0,                                        /* tp_as_mapping     */
  0,                                        /* tp_hash           */
  0,                                        /* tp_call           */
  0,                                        /* tp_str            */
  0,                                        /* tp_getattro       */
  0,                                        /* tp_setattro       */
  0,                                        /* tp_as_buffer      */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags          */  
  "Documentation Placeholder",              /* tp_doc            */
  0,		                            /* tp_traverse       */     
  0,		                            /* tp_clear          */     
  0,		                            /* tp_richcompare    */     
  0,		                            /* tp_weaklistoffset */     
  0,		                            /* tp_iter           */    
  0,		                            /* tp_iternext       */     
  OptimizationProblem_methods,              /* tp_methods        */     
  0,                                        /* tp_members        */     
  0,                                        /* tp_getset         */     
  0,                                        /* tp_base           */     
  0,                                        /* tp_dict           */     
  0,                                        /* tp_descr_get      */     
  0,                                        /* tp_descr_set      */     
  0,                                        /* tp_dictoffset     */     
  (initproc)OptimizationProblem_init,       /* tp_init           */     
  0,                                        /* tp_alloc          */     
  0,                                        /* tp_new            */ 
  PyObject_Del,                             /* tp_free           */
};


#endif // PYROL_OPTIMIZATION_PROBLEM_HPP
