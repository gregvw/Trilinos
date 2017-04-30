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
    x = np.ones(5) # Optimization vector
    l = np.ones(3) # Lagrange multipliers

    solve_opt = { "Algorithm" : "Composite Step",
                  "Step" : {
                      "Composite Step" : {
                          "Optimality System Solver" : {
                              "Nominal Relative Tolerance" : 1.e-4,
                              "Fix Tolerance" : True
                          },
                          "Tangential Subproblem Solver" : {
				      "Iteration Limit" : 20,
                              "Relative Tolerance" : 1e-2
                          },
                          "Output Level" : 0 
                      }   
                  },
                  "Status Test" : {
                      "Gradient Tolerance" : 1.e-12,
                      "Constraint Tolerance" : 1.e-12,
                      "Step Tolerance" : 1.e-18,
                      "Iteration Limit" : 100
                  }
              }

    output,iterates = pyrol.solveEqualityConstrained(obj,con,x,l,solve_opt)
    print(output)
     
