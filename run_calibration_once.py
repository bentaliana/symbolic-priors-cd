from experiments.selection_study.calibration import run_calibration

p = run_calibration(
    "experiments/selection_study/configs/calibration",
    "results",
)
print(p)
