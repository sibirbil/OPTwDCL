# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
#import random
#import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

#This file contains procedures for setting up problems to be solved by dcl

syria_xlsx_path = Path("../../OptiCL/notebooks/WFP/processed-data/Syria_instance.xlsx")
wfp_csv_path = Path("../../OptiCL/notebooks/WFP/processed-data/WFP_dataset.csv")

@dataclass
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
    df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)  
    y = df["label"].astype(float)
    X = df.drop(columns=["label"])
    return X, y


#In this routine we add regular constraints, bounds, etc all components of the 
#optimization problem except the learned constraint
#We shuffle the data set here before training
def setup_problem_wfp(
    problem_model_type,
    *,
    shuffle_seed: int = 0,
    pal_req: float = 0.0,          #palatability requirement (lower bound)
):
    instance = load_syria_instance(syria_xlsx_path)
    X, y = load_wfp_dataset(wfp_csv_path, shuffle_seed=shuffle_seed)
    if(problem_model_type=="gurobi"):
        model = gp.Model()
        
        foods = list(instance.nutr_val.index)
        nutrients = list(instance.nutr_req.columns)

        A = instance.nutr_val.loc[foods, nutrients].to_numpy(dtype=float)    
        req = instance.nutr_req.iloc[0][nutrients].to_numpy(dtype=float)     
        cost = instance.cost_p.loc[foods].to_numpy(dtype=float)              
        
        F = len(foods)
        
        #change notation to x = [z;output_var], output_var = g(z)
        input_vars = model.addMVar(shape=F, lb=0.0, vtype=GRB.CONTINUOUS, name="input_vars") #z
        output_var = model.addMVar(shape=1, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="output_vars") 
        #add output_vars (palatability), add palatability lower_bound=palatability_req
        
        z = input_vars

        #pal = model.addMVar(shape=1, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pal")
        #model.addConstr(pal[0] >= float(pal_req), name="pal_lb")
        model.addConstr(output_var[0] >= float(pal_req), name="pal_lb")

        model.addConstr(A.T @ z >= req, name="nutr_req")
        
        sugar_i = foods.index("Sugar")
        salt_i = foods.index("Salt")
        model.addConstr(z[sugar_i] == 0.2, name="fix_sugar")
        model.addConstr(z[salt_i]  == 0.05, name="fix_salt")
    
        model.setObjective(cost @ z, GRB.MINIMIZE)

        model.update()
        
    #else if(problem_model_type=="pyomo"): #TODO 
    return model, problem_model_type, X, y


def dcl_setup_problem(problem_name, learning_model_type, adjustable_problem_parameters,seed):
#read problem data and put into input and output data structures
#construct the objective / additional constraints of the problem
        
    #set problem_model_type
    if(learning_model_type=="cart" or learning_model_type=="nn"): #TODO: Add more MIP-representable types
        problem_model_type = "gurobi"
    else:
        problem_model_type = "pyomo" 

    if(problem_name=="wfp"):
        return setup_problem_wfp(problem_model_type, shuffle_seed=seed, 
                                 pal_req=adjustable_problem_parameters)
    #elif(problem_name=="rsnbrk"):
    #    return setup_problem_rsnbrk(problem_model_type)
    #else:
    #  throw error


