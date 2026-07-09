# evaluate_one_iter_poly_at_dcl_points.py

from pathlib import Path
import csv


# =========================================================
# 1. Load the polynomial function printed from the
#    1-iteration/full-data DCL run
# =========================================================
FUNCTION_PATH = Path("outputs/one_iter_full_data_dcl_polynomial_function.py")

if not FUNCTION_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {FUNCTION_PATH}. "
        "First run the 1-iteration/full-data DCL and print the polynomial."
    )

namespace = {}
exec(FUNCTION_PATH.read_text(encoding="utf-8"), namespace)

if "alt_dcl_global_ghat" not in namespace:
    raise RuntimeError(
        f"{FUNCTION_PATH} does not define alt_dcl_global_ghat(x0, x1, x2)."
    )

alt_dcl_global_ghat = namespace["alt_dcl_global_ghat"]


# =========================================================
# 2. DCL points from your long Rosenbrock table
#
# Format:
#   Iter, x0, x1, x2, True g from your table
#
# I include True g from the table so that Err(AltDCL)
# matches the displayed table values.
# =========================================================
points = [
    (0,  -1.637274,  0.213708,  0.043281,   2.256100),
    (1,  -0.115282,  0.717479,  0.512713,   0.195526),
    (2,  -4.116976,  1.823774,  3.320674,   4.798581),
    (3,   0.047118,  0.873055,  0.763275,  -0.030893),
    (4, -70.825445,  0.097393, -0.449349,  92.693060),
    (5,  -0.334257,  1.173416,  1.375228,   0.364612),
    (6,   1.997862, -1.575745,  2.263422,   9.456777),
    (7,   1.014799, -1.499744,  2.342268,   6.099473),
    (8,  -0.978632,  0.870817,  0.754314,   0.996926),
    (9,  33.781077,  1.424944, -0.449015, 581.181900),
    (10, -53.178887, 0.243228,  1.488836, 258.148800),
    (11, -0.795148,  0.848111,  0.720261,   0.818312),
    (12,  2.343113, -1.536541,  2.229598,   5.816481),
    (13, -0.174924,  0.893664,  0.795797,   0.187037),
    (14, -4.136969,  2.013453,  4.045723,   5.170892),
    (15, -0.083565,  1.473804,  2.174784,   0.308776),
    (16, 33.781084,  1.424944, -0.449015, 581.181800),
    (17, -0.483039,  1.266421,  1.601137,   0.554740),
    (18, -0.345725,  0.797180,  0.636499,   0.386962),
    (19,  1.770870,  0.099280, -0.002205,  -0.945024),
    (20,  0.907112,  0.206192,  0.040816,  -0.276691),
    (21, -39.950604, 0.395936,  3.212889, 974.304500),
    (22, -1.543993,  1.185625,  1.419304,   1.596936),
    (23,  0.546095,  0.571606,  0.320842,  -0.359102),
    (24,  0.314580,  1.228705,  1.508510,  -0.262128),
    (25,  1.260102,  0.362930,  0.123614,  -0.847675),
    (26, -0.325442,  0.989257,  0.978052,   0.325591),
    (27, -0.538175,  0.908265,  0.822768,   0.547064),
    (28, -0.043381,  1.139250,  1.294796,   0.063729),
    (29, -16.795782, 0.396968,  0.025406,  18.906530),
]


# =========================================================
# 3. Optional: recompute true Rosenbrock g from rounded x's
#
# This may differ slightly from the table True g because your table
# x-values are rounded to 6 decimals.
# =========================================================
def rosenbrock_true_g(x0, x1, x2):
    return (
        -x0
        + 100.0 * (x2 - x1 ** 2) ** 2
        + (1.0 - x1) ** 2
    )


# =========================================================
# 4. Evaluate AltDCL-global at all DCL points
# =========================================================
rows = []

for iter_id, x0, x1, x2, true_g_table in points:
    alt_g = float(alt_dcl_global_ghat(x0, x1, x2))

    true_g_recomputed = float(rosenbrock_true_g(x0, x1, x2))

    err_alt_table = abs(alt_g - true_g_table)
    err_alt_recomputed = abs(alt_g - true_g_recomputed)

    rows.append({
        "Iter": iter_id,
        "x0": x0,
        "x1": x1,
        "x2": x2,
        "True g table": true_g_table,
        "True g recomputed": true_g_recomputed,
        "AltDCL Global ghat": alt_g,
        "Err AltDCL table": err_alt_table,
        "Err AltDCL recomputed": err_alt_recomputed,
    })


# =========================================================
# 5. Print simple console table
# =========================================================
print()
print("ALT DCL GLOBAL MODEL EVALUATED AT LONG-DCL POINTS")
print(
    "Iter & x0 & x1 & x2 & True g(table) & "
    "AltDCL Global ghat & Err(AltDCL) \\\\"
)

for r in rows:
    print(
        f"{r['Iter']} & "
        f"{r['x0']:.6f} & {r['x1']:.6f} & {r['x2']:.6f} & "
        f"{r['True g table']:.6f} & "
        f"{r['AltDCL Global ghat']:.6e} & "
        f"{r['Err AltDCL table']:.6e} \\\\"
    )


# =========================================================
# 6. Save CSV
# =========================================================
OUT_CSV = Path("outputs/alt_dcl_global_evaluated_at_long_dcl_points.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Iter",
            "x0",
            "x1",
            "x2",
            "True g table",
            "True g recomputed",
            "AltDCL Global ghat",
            "Err AltDCL table",
            "Err AltDCL recomputed",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print("Saved CSV to:")
print(OUT_CSV)


# =========================================================
# 7. Save LaTeX rows/table
# =========================================================
OUT_TEX = Path("outputs/alt_dcl_global_evaluated_at_long_dcl_points.tex")

latex_lines = []

latex_lines.append(r"\begin{tabular}{c rrr r rr}")
latex_lines.append(r"\toprule")
latex_lines.append(
    r"Iter & $x_0$ & $x_1$ & $x_2$ & True $g$ & "
    r"AltDCL Global $\hat g$ & Err(AltDCL) \\"
)
latex_lines.append(r"\midrule")

for r in rows:
    latex_lines.append(
        f"{r['Iter']} & "
        f"{r['x0']:.6f} & {r['x1']:.6f} & {r['x2']:.6f} & "
        f"{r['True g table']:.6f} & "
        f"{r['AltDCL Global ghat']:.6e} & "
        f"{r['Err AltDCL table']:.6e} \\\\"
    )

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")

OUT_TEX.write_text("\n".join(latex_lines), encoding="utf-8")

print()
print("Saved LaTeX table to:")
print(OUT_TEX)


# =========================================================
# 8. Optional: print only the two columns you would append
#    to your existing landscape table
# =========================================================
print()
print("ONLY THE TWO COLUMNS TO APPEND TO YOUR EXISTING TABLE:")
print("AltDCL Global ghat & Err(AltDCL) \\\\")

for r in rows:
    print(
        f"{r['AltDCL Global ghat']:.6e} & "
        f"{r['Err AltDCL table']:.6e} \\\\"
    )