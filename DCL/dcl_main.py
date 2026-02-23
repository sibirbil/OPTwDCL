# -*- coding: utf-8 -*-

import dcl_core as dcl
import dcl_problem as dclp


def main():

    #set problem  
    problem_name = "wfp"
    adjustable_problem_parameters = 0.6 #palatability threshold for the wfp problem
    learning_model_type = "cart"
    problem, problem_model_type, X, y = dclp.dcl_setup_problem(problem_name, learning_model_type, adjustable_problem_parameters)
    #x_init = SET #TODO

    #set nondefault options
    options = dcl.dcl_options()
    options.trust_region = "convex_hull"
    options.learning_model_type = learning_model_type
    options.max_iterations = 10
    #options.check_options()
    
    #solve!    
    dcls = dcl.dcl_solver(problem, problem_model_type, options, X,y, None)
    dcls.solve()


if __name__ == "__main__":
    main()
