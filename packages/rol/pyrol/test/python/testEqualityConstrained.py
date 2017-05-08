import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from vector import vector
import numpy as np
import pyrol
from problems import equality


if __name__ == '__main__':

    print("Testing solution of an equality constrained problem using Python Vector")

    obj = equality.Objective()
    con = equality.EqualityConstraint()
    x = vector(0.1*np.ones(5)) # Optimization vector
    l = vector(np.random.rand(3)) # Lagrange multipliers

    solve_opt = { "Algorithm" : "Composite Step",
                  "Step" : {
                      "Composite Step" : {
                          "Optimality System Solver" : {
                              "Nominal Relative Tolerance" : 1.e-4,
                              "Fix Tolerance" : True
                          },
                          "Tangential Subproblem Solver" : {
				      "Iteration Limit" : 4,
                              "Relative Tolerance" : 1.0e-3
                          },
                          "Output Level" : 0
                      }
                  },
                  "Status Test" : {
                      "Gradient Tolerance" :   1.e-8,
                      "Constraint Tolerance" : 1.e-8,
                      "Step Tolerance" :       1.e-10,
                      "Iteration Limit" :      5
                  }
              }

    output,iterates = pyrol.solveEqualityConstrained(obj,con,x,l,solve_opt)
    print(output)
