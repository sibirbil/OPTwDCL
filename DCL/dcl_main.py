# -*- coding: utf-8 -*-
import numpy as np
import random
import dcl_core as dcl
import dcl_problem as dclp


#def _set_seeds(seed: int = SEED) -> None:
#    random.seed(seed)
#    np.random.seed(seed)

def main():
    #_set_seeds(SEED)

    #set problem  
    problem_name = "wfp"
    learning_model_type = "cart"
    problem, problem_model_type, X, y = dclp.dcl_setup_problem(problem_name, learning_model_type)
    #x_init = SET #TODO

    #set nondefault options
    options = dcl.dcl_options()
    options.trust_region = "convex_hull"
    options.learning_model_type = learning_model_type
    options.max_iterations = 10
    #options.check_options()

    #solve!    
    dcls = dcl.dcl_solver(problem, problem_model_type, options, X,y, x_init)
    dcls.solve()


if __name__ == "__main__":
    main()
