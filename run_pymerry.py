# -*- coding: utf-8 -*-
"""
Created on Tuesday July 18 - 2023

Run code for PyMERRY
"""


# IMPORTS:
import os
import warnings
import numpy as np
import pygimli as pg
from time import time
import tools.PyMERRY as PM
from datetime import timedelta
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Turn off the automatic figure display from PyGIMLi:
pg.viewer.mpl.noShow(on=True)

if __name__ == "__main__":  # check if main run for parallel computing
    # %% 1 - LOAD INPUT DATA FOR PyMERRY from parameter file:
    parameter_file_name = "parameters.txt"

    # %% 2 - INSTANCIATE PyMERRY:
    # 2.1 - Initialize:
    inputs = PM.InputTools.load_parameters(parameter_file_name)
    data = inputs["data"]
    model = inputs["model"]
    mesh = inputs["mesh"]

    # 2.2 - Create saving directory:
    save_dir = inputs["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    PM.InputTools.save_parameters_file(inputs)

    # %% 3 - RUN PyMERRY:
    to = time()

    # 3.1 - Set input data:
    pm = PM.MERRY(data=inputs["data"], model=inputs["model"],
                  mesh=inputs["mesh"], DOI_DD=inputs["doi_DD"],
                  DOI_W=inputs["doi_W"], DOI_WS=inputs["doi_WS"],
                  alpha=inputs["alpha"], beta=inputs["beta"])

    # 3.2 - Compute masks and coverage:
    pm.create_masks(verbose=True)

    # 3.3 - Compute errors, ("__name__" is requiered for parallel runs):
    pm.error_assesment(__name__, verbose=True)

    tf = time()
    print(f"Total runtime {timedelta(seconds=tf-to)}\n")

    # %% 4 - SAVE RESULTS AND PLOTS:
    # 4.1 - Display PyMERRY status in console:
    print(pm)

    # 4.2 Save results as .csv files :
    pm.save("quadrupoles", path=save_dir)
    pm.save("u", path=save_dir)
    pm.save("j", path=save_dir)
    pm.save("frechet", path=save_dir)
    pm.save("masks", path=save_dir)
    pm.save("coverage", path=save_dir)
    pm.save("profile_mask", path=save_dir)
    pm.save("rhoa_th", path=save_dir)
    pm.save("error_absolute", path=save_dir)
    pm.save("error_relative", path=save_dir)
    pm.save("error_min", path=save_dir)
    np.savetxt(os.path.join(save_dir, "py_merry_run_time_s.txt"),
               np.array([tf-to]), delimiter=";")

    # %% 5 - FIGURES:
    # 5.1 - Coverage plot:
    fig1, ax1 = plt.subplots(figsize=(16, 9))

    pg.show(mesh, data=pm.coverage, ax=ax1, cMin=0, cMax=1,
            cMap="Greys", label="Coverage")

    fig1.savefig(os.path.join(save_dir, "coverage.png"), bbox_inches='tight')

    # 5.2 - Error plot: low resistivity / model / high resistivity:
    fig2, ax2 = plt.subplots(3, figsize=(16, 9))

    PM.Plots.plot_error_bars(
        pm.model_mask, pm.profile_mask, pm.error_absolute, pm.mesh,
        inputs["cmin"], inputs["cmax"], inputs["gamma"], inputs["logscale"],
        fig2, ax2[0], ax2[1], ax2[2])

    fig2.savefig(os.path.join(save_dir, "errors.png"), bbox_inches='tight')

    # Turn on the automatic figure display from PyGIMLi:
    pg.viewer.mpl.noShow(on=False)
