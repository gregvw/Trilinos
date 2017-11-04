import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import pyrol
from problems import equality


if __name__ == '__main__':

    print("Testing solution of an equality constrained problem using NumPy array")

    obj = equality.Objective()
    con = equality.EqualityConstraint()
    x = np.array([-1.8,1.7,1.9,-0.8,-0.8])
    l = np.random.rand(3) # Lagrange multipliers

    # Given solution
    x_sol = np.array([-1.717143570394391e+00,
                       1.595709690183565e+00,
                       1.827245752927178e+00,
                      -7.636430781841294e-01,
                      -7.636430781841294e-01])
 

    test_opt = { "Check Gradient"                     : True,
                 "Check HessVec"                      : True,
                 "Check HessSym"                      : True,
                 "Check Jacobian"                     : True, 
                 "Check Adjoint Jacobian Consistency" : True,
                 "Check Adjoint Hessian"              : True,
                 "Order"            : 2,
                 "Steps"            : [0.1**i for i in range(7)]
               }




    solve_opt = { "Algorithm" : "Composite Step",
                  "Return Iterates" : "True", 
                  "Step" : {
                      "Composite Step" : {
                          "Optimality System Solver" : {
                              "Nominal Relative Tolerance" : 1.e-4,
                              "Fix Tolerance" : True
                          },
                          "Tangential Subproblem Solver" : {
			      "Iteration Limit" : 20,
                              "Relative Tolerance" : 1.0e-2
                          },
                          "Output Level" : 0
                      }   
                  },
                  "Status Test" : {
                      "Gradient Tolerance" :   1.e-12,
                      "Constraint Tolerance" : 1.e-12,
                      "Step Tolerance" :       1.e-18,
                      "Iteration Limit" :      100
                  }
              }


    test_output  = pyrol.testObjective(obj,x,test_opt)
    test_output += pyrol.testEqualityConstraint(con,x,l,test_opt)
    print(test_output)

    solve_output = pyrol.solveEqualityConstrained(obj,con,x,l,solve_opt)

    if len(solve_output)>1:
        dim = len(x)
        print(solve_output[0])
        print("Optimization vectors\n")
        xv = lambda i: "".join(["x[",str(i),"]"])
        print("".join(["iter/element  "]+["{0:14}".format(xv(i)) for i in range(dim)]))
        print("-"*((1+dim)*14))
        for i,line in enumerate(solve_output[1].split('\n')):
            if len(line)>0:
                print("".join(["{0:7}{1:7}".format(i,"")]+["{0:14}".format(l) for l in line.split()]))
    else:
        print(solve_output)
    
    print("\nExact Solution: " + str(x_sol)) 
