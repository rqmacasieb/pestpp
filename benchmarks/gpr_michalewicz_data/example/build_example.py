"""Build a runnable pestpp-sm example from the vendored Michalewicz dataset.

This example emulates the 10-D Michalewicz function with the Gaussian Process
Regression (GPR) surrogate built into pestpp-sm.

Data roles (all CSVs live in the parent gpr_michalewicz_data directory):

  * mic.0.dv_pop.csv  -> TRAINING inputs  (columns x1..x10) via gpr_training_input_file
  * mic.0.obs_pop.csv -> TRAINING outputs (column  func)        [optional gpr_training_output_file]
  * mic.dv_pop.csv    -> PREDICTION inputs (columns x1..x10)     [ 100 points]

The forward model (``forward_run.py``) evaluates the same 10-D Michalewicz
function used in laGPy's ``MIC.py`` (``benchmark_functions.Michalewicz`` with
``m=10``).  pestpp-sm runs that model for every training parameter set, records
the outputs, fits a GPR emulator, and predicts the prediction suite without
running the model.

Running this script writes a self-contained PEST(++) dataset into this folder:

    forward_run.py     the Michalewicz model
    par.dat.tpl        template file (x1..x10)
    obs.dat.ins        instruction file (func)
    mic_sm.pst         the control file with the ++gpr_* options
    gpr_train_in.csv   training parameter sets (x1..x10)
    gpr_in.csv         prediction parameter sets (x1..x10)

Then run the emulator with:   pestpp-sm mic_sm.pst
"""

import os
import pandas as pd
import pyemu

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)  # the gpr_michalewicz_data directory
PARS = ["x{0}".format(i) for i in range(1, 11)]


def build():
    # 1. load the source data ------------------------------------------------
    train_in = pd.read_csv(os.path.join(DATA, "mic.0.dv_pop.csv")).set_index("real_name")
    pred_in = pd.read_csv(os.path.join(DATA, "mic.dv_pop.csv")).set_index("real_name")

    # 2. write the template / instruction files ------------------------------
    tpl = ["ptf ~"] + ["{0} ~  {0}  ~".format(p) for p in PARS]
    with open(os.path.join(HERE, "par.dat.tpl"), "w") as f:
        f.write("\n".join(tpl) + "\n")
    with open(os.path.join(HERE, "obs.dat.ins"), "w") as f:
        f.write("pif ~\nl1 w !func!\n")

    # 3. build the control file from the io file pair ------------------------
    pst = pyemu.helpers.pst_from_io_files(
        os.path.join(HERE, "par.dat.tpl"),
        os.path.join(HERE, "par.dat"),
        os.path.join(HERE, "obs.dat.ins"),
        os.path.join(HERE, "obs.dat"),
        pst_path=".",
    )

    par = pst.parameter_data
    par.loc[:, "partrans"] = "none"
    par.loc[:, "parlbnd"] = -10.0
    par.loc[:, "parubnd"] = 10.0
    par.loc[:, "parval1"] = 0.0
    par.loc[:, "parchglim"] = "relative"

    pst.observation_data.loc[:, "weight"] = 1.0
    pst.model_command = ["python forward_run.py"]
    pst.control_data.noptmax = 0

    # 4. pestpp-sm control variables -----------------------------------------
    pst.pestpp_options = {}
    pst.pestpp_options["gpr_training_input_file"] = "gpr_train_in.csv"
    pst.pestpp_options["gpr_input_file"] = "gpr_in.csv"
    # 1000 training points -> use a local approximate GP (laGP) per prediction
    pst.pestpp_options["gpr_local"] = True
    pst.pestpp_options["gpr_local_start"] = 10
    pst.pestpp_options["gpr_local_end"] = 50
    pst.pestpp_options["gpr_local_method"] = "alc"
    pst.pestpp_options["gpr_lengthscale"] = -1.0   # estimate by MLE
    pst.pestpp_options["gpr_nugget"] = 1.0e-4      # fixed
    pst.pestpp_options["sweep_forgive"] = True
    pst.write(os.path.join(HERE, "mic_sm.pst"))

    # 5. training and prediction inputs (x1..x10 only) -----------------------
    train_in[PARS].to_csv(os.path.join(HERE, "gpr_train_in.csv"), index_label="real_name")
    pred_in[PARS].to_csv(os.path.join(HERE, "gpr_in.csv"), index_label="real_name")

    print("wrote pestpp-sm example to:", HERE)
    print("  training points  :", len(train_in))
    print("  prediction points:", len(pred_in))
    print("run with:  pestpp-sm mic_sm.pst")


if __name__ == "__main__":
    build()
