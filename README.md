# Optimization with Dynamic Constraint Learning

Ezgi Öztekin, Figen Öztoprak & İlker Birbil

The repository includes Rosenbrock, WFP, and concrete case studies.

## Files

- `dcl_main.py` — run settings and results.
- `dcl_problem.py` — datasets and optimization problems.
- `dcl_core.py` — local models and optimization routine.

## Run

Use Python 3.10+ with NumPy, pandas, SciPy, and scikit-learn. WFP and concrete also require their datasets, Gurobi, `gurobipy`, and `gurobi_ml`.

Set `problem_name` in `dcl_main.py` to `"rosenbrock"`, `"wfp"`, or `"concrete"` (default: `"rosenbrock"`), then run:

```bash
python dcl_main.py
```
