"""Compare the pestpp-sm emulated outputs against the known true Michalewicz
values for the prediction set, and report accuracy metrics.

The prediction inputs (mic.dv_pop.csv) have known true outputs stored in
mic.obs_pop.csv (the ``func`` column), so we can measure how well the GPR
emulator - trained only on the mic.0 dataset - reproduces them.

Run after pestpp-sm has produced mic_sm.obs.csv:

    pestpp-sm mic_sm.pst
    python compare.py
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)  # the gpr_michalewicz_data directory


def compare():
    truth = pd.read_csv(os.path.join(DATA, "mic.obs_pop.csv")).set_index("real_name")
    emu = pd.read_csv(os.path.join(HERE, "mic_sm.obs.csv")).set_index("real_name")

    idx = truth.index
    y_true = truth.loc[idx, "func"].values.astype(float)
    y_emu = emu.loc[idx, "func"].values.astype(float)
    y_std = emu.loc[idx, "func_sd"].values.astype(float)

    err = y_emu - y_true
    rng = float(y_true.max() - y_true.min())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nrmse = rmse / rng
    maxabs = float(np.abs(err).max())
    corr = float(np.corrcoef(y_emu, y_true)[0, 1])
    # fraction of true values within +/- 2 predictive std of the emulated mean
    cover = float(np.mean(np.abs(err) <= 2.0 * y_std))

    print("pestpp-sm emulator vs true Michalewicz ({0} prediction points)".format(len(idx)))
    print("  true output range : [{0:.4f}, {1:.4f}]".format(y_true.min(), y_true.max()))
    print("  correlation       : {0:.6f}".format(corr))
    print("  RMSE              : {0:.4e}".format(rmse))
    print("  NRMSE (RMSE/range): {0:.4e}".format(nrmse))
    print("  max |error|       : {0:.4e}".format(maxabs))
    print("  mean pred. std    : {0:.4e}".format(y_std.mean()))
    print("  within +/-2 std   : {0:.1%}".format(cover))


if __name__ == "__main__":
    compare()
