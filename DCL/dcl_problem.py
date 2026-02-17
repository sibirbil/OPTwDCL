# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB


#This file contains procedures for setting up problems to be solved by dcl

class SyriaInstance:
    nutr_val: pd.DataFrame   #index=Food, columns=nutrients
    nutr_req: pd.DataFrame   #index=Type, columns=nutrients
    cost_p: pd.Series        #index=Food

def load_syria_instance(xlsx_path: str | Path) -> SyriaInstance:
    xlsx_path = Path(xlsx_path)

    nutr_val = pd.read_excel(
        xlsx_path,
        sheet_name="nutr_val",
        index_col="Food",
        engine="openpyxl",
    )
    nutr_req = pd.read_excel(
        xlsx_path,
        sheet_name="nutr_req",
        index_col="Type",
        engine="openpyxl",
    )
    cost_p = pd.read_excel(
        xlsx_path,
        sheet_name="FoodCost",
        index_col="Supplier",
        engine="openpyxl",
    ).iloc[0, :]

    return SyriaInstance(nutr_val=nutr_val, nutr_req=nutr_req, cost_p=cost_p)


def load_wfp_dataset(csv_path: str | Path, *, shuffle_seed: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)  #FGN:??
    y = df["label"].astype(float)
    X = df.drop(columns=["label"])
    return X, y


def setup_problem_wfp(problem_model_type):
    instance = load_syria_instance(syria_xlsx_path)
    X, y = load_wfp_dataset(wfp_csv_path, shuffle_seed=0)
    if(problem_model_type=="gurobi"):
        model = gp.Model()
        #model.addVar...
        #model.setObjective...
        #CONTINUE: Add regular constraints, bounds, etc 
    #else if(problem_model_type=="pyomo"): #TODO 
    return model, problem_model_type, X, y


def dcl_setup_problem(problem_name, learning_model_type):
#read problem data and put into input and output data structures
#construct the objective / additional constraints of the problem
    #set problem_model_type
    if(learning_model_type=="cart"): #TODO: Add more MIP-representable types
        problem_model_type = "gurobi"
    else:
        problem_model_type = "pyomo" 

    if(problem_name=="wfp"):
        return setup_problem_wfp(problem_model_type)
    #elif(problem_name=="rsnbrk"):
    #    return setup_problem_rsnbrk(problem_model_type)
    #else:
    #  throw error


