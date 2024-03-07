# -*- coding: utf-8 -*-
"""
Created on Tuesday July 18 - 2023

PyMERRY Source code: v1.0

PyMERRY: a python solution for improved interpretation of electrical
resistivity tomography images” article in Geophysics.

    Maxime Gautier
    Stéphanie Gautier
    Rodolphe Cattin

University of Montpellier, Géosciences Montpellier – CNRS, Montpellier, France
E-mail: maxime.gautier@umontpellier.fr;
        stephanie.gautier-raux@umontpellier.fr;
        rodolphe.cattin@umontpellier.fr.
"""


import os
import shutil
import warnings
import itertools
import numpy as np
import pandas as pd
import tqdm as tqdm
import pygimli as pg
from time import time
import numpy.ma as ma
from sys import platform
import multiprocessing as mp
from collections import Counter
from datetime import datetime, timedelta


class MERRY:
    """
    PyMERRY Class:

    Attributes:
        - data: pg.DataContainerERT
            PyGimli object containing the position of each electrode and the
            measurement of apparent resistivity from each quadrupole.
        - model: np.ndarray
            Electrical resistivity model in ohm-m. Shape: (nb cells, ).
        - mesh: pg.Mesh
            Pygimli mesh object of the domain.
        - DOI_W: float
            Depth of investigation coefficient for Wenner array.
        - DOI_WS: float
            Depth of investigation coefficient for Wenner-Schlumberger array.
        - DOI_DD: float
            Depth of investigation coefficient for Dipole-Dipole array.
        - alpha: float
            Resistivity-meter injection accuracy.
        - beta: float
            Resistivity-meter potential accuracy.
        - model_var_p : float
            Negative variation of resistivity to apply in each cell of the
            model in %. Set 1 for 100 % and 0.5 for 50 %.
        - nb_cpu: int
            Nb of cpu to use for parallel runs.
        - parallel_start_method : str
            Name of the parallel starting method.
        - k: np.ndarray
            Geometrical factor in m associated with each quadrupole.
            Without taking account of the term "2*pi".
            Shape: (nb measurement, ).
        - u: np.ndarray
            Potential field u in V computed at each node of the mesh and for
            each electrode. Shape: (nb electrodes, nb nodes).
        - j: list(np.ndarray)
            A list containing j vectors associated with each electrode.
            len: nb electrodes. Each item of j is a "np.ndarray" with 3 columns
            for x, y, and z components of j in A/m², respectively.
            Shape: (nb cells, 3).
        - j_norm: np.ndarray
            Norms of current density vectors j in A/m².
            Shape: (nb electrodes, nb cells).
        - frechet: np.ndarray
            The Frechet derivative for each quadrupole.
            Shape: (nb measurements, nb cells).
        - masks: np.ndarray
            Binary masks (0; 1) for each quadrupole in the survey.
            Shape: (nb measurements, nb cells).
        - coverage: np.ndarray
            Coverage array. Shape: (nb cells,).
        - profile_mask: np.ndarray
            Binary mask at the scale of the profile. Shape: (nb cells,).
        - covered_cells_id: np.ndarray
            List of index if the covered cells in the mesh.
        - model_mask: np.ndarray
            Resistivity model with enpty cells for non covered cells.
            Shape: (nb cells, ).
        - rhoa_th: np.ndarray
            Theoretical apparent resistivity for each quadrupole of the
            profile. Shape: (nb measurements,).
        - error_neg_min:np.ndarray
            Minimal of negative error in %. Shape: (nb cells, ).
        - error_pos_min:np.ndarray
            Minimal of positive error in %. Shape: (nb cells, ).
        - error_min:np.ndarray
            Minimal error between abs(error_neg, error_pos).
            Shape: (nb cells, ).
        - error_relative:np.ndarray
            Same as error_min.
        - error_absolute: np.ndarray
            Absolute error in ohm-m. Shape: (nb cells, ).

    """

    def __init__(self, data, model, mesh,
                 DOI_DD=0.195, DOI_W=0.11, DOI_WS=0.125,
                 alpha=0.002, beta=0.002, model_var_p=0.95,
                 nb_cpu=mp.cpu_count()-2):

        # Requiered attributes:
        self.data = data
        self.model = model
        self.mesh = mesh

        # Default attributes
        self.DOI_W, self.DOI_WS, self.DOI_DD = DOI_W, DOI_WS, DOI_DD
        self.alpha, self.beta = alpha, beta
        self.model_var_p = model_var_p
        self.nb_cpu = nb_cpu

        # Check inputs and generate a quadrupole table:
        self._load_check_inputs()
        self._generate_quadrupoles_table()

        # Attributes concerning mask and coverage:
        self.u = None
        self.j = None
        self.frechet = None
        self.masks = None
        self.coverage = None
        self.profile_mask = None
        self.covered_cells_id = None

        # Attributes concerning errors:
        self.rhoa_th = None
        self.error_neg_min = None
        self.error_pos_min = None
        self.error_min = None
        self.error_relative = None
        self.error_absolute = None

        # Set starts methods:
        if platform == "win32":  # run on Windows
            try:
                mp.set_start_method("spawn", force=True)
            except:
                pass
        elif platform == "linux":  # run on Linux
            try:
                mp.set_start_method("fork", force=True)
            except:
                pass
        elif platform == "???":  # run on MacOS
            try:
                mp.set_start_method("fork", force=True)
            except:
                pass
        else:
            try:
                mp.set_start_method("spawn", force=True)
            except:
                pass
        self.parallel_start_method = mp.get_start_method()

    def __repr__(self) -> str:
        # Set state of some attributes:
        u_status = "valid" if self.u is not None else None
        j_status = "valid" if self.j is not None else None
        frechet_status = "valid" if self.frechet is not None else None
        masks_status = "valid" if self.masks is not None else None
        coverage_status = "valid" if self.coverage is not None else None
        profile_mask_s = "valid" if self.profile_mask is not None else None
        rhoa_th_s = "valid" if self.rhoa_th is not None else None
        error_neg_min_s = "valid" if self.error_neg_min is not None else None
        error_pos_min_s = "valid" if self.error_pos_min is not None else None
        error_min_s = "valid" if self.error_min is not None else None
        error_relative = "valid" if self.error_relative is not None else None
        error_absolute_s = "valid" if self.error_absolute is not None else None

        # Set state of nc-number of covered cells:
        nc = None
        cov_percentage = "ND"
        if self.profile_mask is not None:
            cov = ((nc := len(self.covered_cells_id)) / self.mesh.cellCount())
            cov_percentage = f" ({round(cov*100, 2)}%)"

        return (f"PyMERRY with\n {self.mesh}\n model: {self.model.shape[0]}"
                + f"  values\n data: {self.data}"
                + f"\n ERT profile: nb data: {self.nb_data}, "
                + f"nb electrodes: {self.nb_elecs}, length: {self.length}m, "
                + f" spacing: {self.spacing}m, {self.array_types}\n"
                + f" DOI coefs (W, WS, DD): {self.DOI_W}, {self.DOI_WS},"
                + f" {self.DOI_DD}\n"
                + f" Instrumental accuracy (I, U): {self.alpha}, {self.beta}"
                + f"\n Cells covered: {nc} / {self.mesh.cellCount()} "
                + cov_percentage
                + f"\n u status: {u_status}"
                + f"\n j status: {j_status}"
                + f"\n frechet status: {frechet_status}"
                + f"\n masks status: {masks_status}"
                + f"\n coverage status: {coverage_status}"
                + f"\n profile mask status: {profile_mask_s}"
                + f"\n rhoa th status: {rhoa_th_s}"
                + f"\n deviation neg min status: {error_neg_min_s}"
                + f"\n deviation pos min status: {error_pos_min_s}"
                + f"\n deviation min status: {error_min_s}"
                + f"\n deviation relative status: {error_relative}"
                + f"\n deviation absolute status: {error_absolute_s}\n")

    def __str__(self) -> str:
        # Set state of some attributes:
        u_status = "valid" if self.u is not None else None
        j_status = "valid" if self.j is not None else None
        frechet_status = "valid" if self.frechet is not None else None
        masks_status = "valid" if self.masks is not None else None
        coverage_status = "valid" if self.coverage is not None else None
        profile_mask_s = "valid" if self.profile_mask is not None else None
        rhoa_th_s = "valid" if self.rhoa_th is not None else None
        error_neg_min_s = "valid" if self.error_neg_min is not None else None
        error_pos_min_s = "valid" if self.error_pos_min is not None else None
        error_min_s = "valid" if self.error_min is not None else None
        error_relative = "valid" if self.error_relative is not None else None
        error_absolute_s = "valid" if self.error_absolute is not None else None

        # Set state of nc-number of covered cells:
        nc = None
        cov_percentage = "ND"
        if self.profile_mask is not None:
            cov = ((nc := len(self.covered_cells_id)) / self.mesh.cellCount())
            cov_percentage = f" ({round(cov*100, 2)}%)"

        return (f"PyMERRY with\n {self.mesh}\n model: {self.model.shape[0]}"
                + f"  values\n data: {self.data}"
                + f"\n ERT profile: nb data: {self.nb_data}, "
                + f"nb electrodes: {self.nb_elecs}, length: {self.length}m, "
                + f" spacing: {self.spacing}m, {self.array_types}\n"
                + f" DOI coefs (W, WS, DD): {self.DOI_W}, {self.DOI_WS},"
                + f" {self.DOI_DD}\n"
                + f" Instrumental accuracy (I, U): {self.alpha}, {self.beta}"
                + f"\n Cells covered: {nc} / {self.mesh.cellCount()} "
                + cov_percentage
                + f"\n u status: {u_status}"
                + f"\n j status: {j_status}"
                + f"\n frechet status: {frechet_status}"
                + f"\n masks status: {masks_status}"
                + f"\n coverage status: {coverage_status}"
                + f"\n profile mask status: {profile_mask_s}"
                + f"\n rhoa th status: {rhoa_th_s}"
                + f"\n deviation neg min status: {error_neg_min_s}"
                + f"\n deviation pos min status: {error_pos_min_s}"
                + f"\n deviation min status: {error_min_s}"
                + f"\n deviation relative status: {error_relative}"
                + f"\n deviation absolute status: {error_absolute_s}\n")

    def _load_check_inputs(self) -> None:
        """Check if input parameters are compatible with each other."""

        # Check if the model is a 1D array:
        if self.model.shape != (self.mesh.cellCount(), ):
            print("Proceed to flatten model array.")
            # Flat the array if not:
            self.model = self.model.flatten()

        # Check if the model size fits the mesh:
        if (ms := self.model.shape[0]) != (mc := self.mesh.cellCount()):
            raise (Exception(
                f"Model ({ms}) does not fit the number of cells ({mc})."))

        # Check if sensors are on a mesh node:
        if not all(p in [[n.x(), n.y(), n.z()] for n in self.mesh.nodes()]
                   for p in np.array(self.data.sensorPositions()).tolist()):
            raise (Exception(
                "Some sensors are not associated with a node of the mesh."))

    def _generate_quadrupoles_table(self) -> None:
        """
        Create a Panda DataFrame with information on each quadrupole.
        A line is a quadrupole with their index.

        Columns are:
            - index of the data (or quadrupole used) ID
            - potisions, xa, xb, xm, xn, ya, yb, ym, un, za, zb, zm, zn
            - index in the PyGimli DataContainerERT: idx_a, idx_b, idx_m, idx_n
            - apparent resistivity in ohm-m measured on the field: rhoa
            - standard deviation on each measurement: err
            - geometrical factor (without 2*pi) k in m, computed if not given
            - array: Wenner (W), Wenner-Schlumberger(WS), Dipole-Dipole (DD)
            - length between outer electrodes of the array L in m
            - length between inner electrodes of the array b in m
            - ratio b on L
            - Depth of investigation coefficient DOI_coef
            - coordinates of the depth of investigation in m: xDOI, yDOI
            - index of cell at the depth of investigation: idx_cell_DOI

        Some columns will be added later:
            - value of the threshold (frechet) for DOI point: frechet_DOI
            - theoretical apparent resistivity computed on model
              in ohm-m: rhoa_th

        Add information on PyMERRY instance:
            - number of measurements
            - profile length in m
            - number of electrodes used
            - electrode spacing in m
            - the quadrupole DataFrame descripted above named "quadrupoles"
            - count of each type of array used

        """

        quads = pd.DataFrame()
        quads.index.rename("ID", inplace=True)

        # Electrodes position:
        x = np.array(self.data.sensorPositions())[:, 0]
        y = np.array(self.data.sensorPositions())[:, 1]
        z = np.array(self.data.sensorPositions())[:, 2]

        quads["xa"] = x[self.data["a"]]
        quads["xb"] = x[self.data["b"]]
        quads["xm"] = x[self.data["m"]]
        quads["xn"] = x[self.data["n"]]

        quads["ya"] = y[self.data["a"]]
        quads["yb"] = y[self.data["b"]]
        quads["ym"] = y[self.data["m"]]
        quads["yn"] = y[self.data["n"]]

        quads["za"] = z[self.data["a"]]
        quads["zb"] = z[self.data["b"]]
        quads["zm"] = z[self.data["m"]]
        quads["zn"] = z[self.data["n"]]

        # Electrodes indexes:
        quads["idx_a"] = self.data["a"]
        quads["idx_b"] = self.data["b"]
        quads["idx_m"] = self.data["m"]
        quads["idx_n"] = self.data["n"]

        # Field data:
        quads['rhoa'] = np.array(self.data['rhoa'])
        quads['err'] = np.array(self.data['err'])

        # Geometrical factor k:
        k = np.array(self.data['k'])
        if np.all(k == 0):
            self.data['k'] = pg.physics.ert.createGeometricFactors(
                self.data, verbose=False)
        quads["k"] = self.data['k'] / (2*np.pi)
        self.k = np.array(quads["k"])

        # Array type, length, depth of investigation (DOI):
        quads["array"] = ["ND"] * quads.shape[0]
        quads["L"] = ["ND"] * quads.shape[0]
        quads["b"] = ["ND"] * quads.shape[0]
        quads["b_on_L"] = ["ND"] * quads.shape[0]
        quads["DOI_coef"] = ["ND"]*quads.shape[0]
        quads["xDOI"] = ["ND"]*quads.shape[0]
        quads["yDOI"] = ["ND"]*quads.shape[0]
        quads["idx_cell_DOI"] = ["ND"]*quads.shape[0]

        # Adaptative DOI coeff for DD (polynomial interpolation deg=3)
        # from maximum of Beta NDIC Curves in Baker, 1989, figure 2, page 1033:
        bL_data = [0, 0.2, 0.4, 0.6, 0.8]
        DICN_max = [0, 0.072782874617737, 0.118042813455657,
                    0.152293577981651, 0.173088685015291]
        poly = np.polyfit(bL_data, DICN_max, 3)
        # r2 = (1 - ((sum(DICN_max-np.polyval(poly, bL_data))**2) /
        #       sum(DICN_max-np.polyval(poly, bL_data))))

        # Set information on each quadrupole:
        for i, q in enumerate(quads.itertuples()):

            # Get quadrupole type: W, WS, or DD:
            q_type = MERRY.get_array_type(q.xa, q.xb, q.xm, q.xn)

            # Wenner configuration:
            if q_type == "W":
                quads.loc[i, "array"] = "W"
                quads.loc[i, "L"] = abs(q.xa - q.xb)
                quads.loc[i, "b"] = abs(q.xm - q.xn)
                quads.loc[i, "b_on_L"] = abs(q.xm - q.xn) / abs(q.xa - q.xb)
                quads.loc[i, "xDOI"] = (xd := q.xa + abs(q.xa - q.xb) / 2)
                quads.loc[i, "yDOI"] = (yd := - self.DOI_W*abs(q.xa - q.xb))
                doi = (xd, yd)
                quads.loc[i, "idx_cell_DOI"] = self.mesh.findCell(doi).id()
                quads.loc[i, "DOI_coef"] = self.DOI_W

            # Wenner-Schlumberger configuration:
            elif q_type == "WS":
                quads.loc[i, "array"] = "WS"
                quads.loc[i, "L"] = abs(q.xa - q.xb)
                quads.loc[i, "b"] = abs(q.xm - q.xn)
                quads.loc[i, "b_on_L"] = abs(q.xm - q.xn) / abs(q.xa - q.xb)
                quads.loc[i, "xDOI"] = (xd := q.xa + abs(q.xa - q.xb) / 2)
                quads.loc[i, "yDOI"] = (yd := - self.DOI_WS * abs(q.xa - q.xb))
                doi = (xd, yd)
                quads.loc[i, "idx_cell_DOI"] = self.mesh.findCell(doi).id()
                quads.loc[i, "DOI_coef"] = self.DOI_WS

            # Dipole-Dipole configuration:
            elif q_type == "DD":
                quads.loc[i, "array"] = "DD"
                Ldoi = abs(q.xa - q.xn)
                quads.loc[i, "L"] = abs(q.xa - q.xn)
                quads.loc[i, "b"] = abs(q.xb - q.xm)
                bL = abs(q.xb - q.xm) / abs(q.xa - q.xn)
                quads.loc[i, "b_on_L"] = bL
                coef = np.polyval(poly, bL)
                quads.loc[i, "xDOI"] = (xd := q.xa + Ldoi / 2)
                quads.loc[i, "yDOI"] = (yd := - coef * Ldoi)
                doi = (xd, yd)
                quads.loc[i, "idx_cell_DOI"] = self.mesh.findCell(doi).id()
                quads.loc[i, "DOI_coef"] = coef

        # Profile informaiton (added to MERRY instance):
        self.nb_data = i + 1
        self.length = abs(np.max(x) - np.min(x))
        self.nb_elecs = x.shape[0]
        self.spacing = abs(x[1] - x[0])
        self.quadrupoles = quads
        self.array_types = dict(Counter(self.quadrupoles["array"]))

    def get_array_type(xa: "float", xb: "float",
                       xm: "float", xn: "float") -> "str":
        """
        Estimate the quadripole configuration between Wenner (W),
        Wenner-Schlumberger (WS) or Dipole-dipole (DD).

        Parameters
        ----------
        xa : float
            Position of A electrode along x-axis in m.
        xb : float
            Position of B electrode along x-axis in m.
        xm : float
            Position of M electrode along x-axis in m.
        xn : float
            Position of N electrode along x-axis in m.

        Returns
        -------
        return_type : str
            'W', 'WS, or 'DD'

        """
        # Get electrodes names in ascending order along x-axis: eg. 'amnb'
        x = {"a": xa, "b": xb, "m": xm, "n": xn}
        x_sorted = dict(sorted(x.items(), key=lambda e: e[1]))

        # Turn 'amnb' into IPPI for injection (I) and potential (P) electrodes:
        x_IP = "".join(list(x_sorted.keys()))
        x_IP = x_IP.replace("a", "I").replace("b", "I")
        x_IP = x_IP.replace("m", "P").replace("n", "P")

        # Get the quadrupole type:
        if x_IP == "IPPI":  # W or WS.
            if abs(x["a"]-x["m"]) == abs(x["m"]-x["n"]) == abs(x["n"]-x["b"]):
                return_type = "W"
            else:
                return_type = "WS"
        elif x_IP == "IIPP" or x_IP == "PPII":  # DD
            return_type = "DD"

        return return_type

    def create_masks(self, **kwargs) -> None:
        """Function for create masks and coverage."""

        verbose = kwargs.pop("verbose", True)

        to = time()
        if verbose:
            print(f"START MASK COMPUTING AT: {datetime.now()}")

        # Compute potential fields u of each electrode:
        if self.u is None:
            self.u = MaskComputation.compute_potential(
                self.data, self.model, self.mesh, verbose=verbose)

        # Compute current density vectors j of each electrode:
        if self.j is None:
            self.j, self.j_norm = MaskComputation.compute_current_density(
                self.u, self.model, self.mesh, verbose=verbose)

        # Assign u and j to each electrode ABMN of each quadrupole:
        quad_u, quad_j = MaskComputation.assign_uj(
            self.quadrupoles, self.u, self.j)

        # Compute the Frechet derivative of each quadrupole:
        if self.frechet is None:
            to_f = time()
            self.frechet = np.array([MaskComputation.frechet_from_j(
                quad_j.loc[q, "a"], quad_j.loc[q, "b"],
                quad_j.loc[q, "m"], quad_j.loc[q, "n"])
                for q in range(self.data.size())])
            tf_f = time()
            if verbose:
                print("  Frechet derivatives computed: " +
                      f"{round(tf_f-to_f, 4)}.")

        # Set value of Frechet derivative at the deph of investigation:
        frechet_DOI = np.array(
            [self.frechet[q, :][self.quadrupoles.loc[q, 'idx_cell_DOI']]
             for q in range(self.data.size())])
        self.quadrupoles["frechet_DOI"] = frechet_DOI  # add to quadrupole tab.

        # Compute binary masks of each quadrupole:
        if self.masks is None:
            self.masks = MaskComputation.compute_masks(
                frechet_DOI, self.frechet, self.data, self.mesh,
                verbose=verbose)

        # Compute the profile coverage, profile mask and get covered cell ids:
        if self.coverage is None:
            self.coverage = MaskComputation.compute_coverage(
                self.masks, verbose=verbose)
            self.covered_cells_id = np.where(self.coverage > 0)[0]
            self.profile_mask = np.where(self.coverage > 0, 1, 0)

            # Apply profile mask on resistivity model:
            nan_mask = np.copy(self.profile_mask).astype(float)
            nan_mask[nan_mask == 0] = np.nan
            self.model_mask = self.model * nan_mask

        tf = time()
        if verbose:
            print(f"END MASK COMPUTING: {timedelta(seconds=(tf - to))}"+"\n")

    def error_assesment(self, check_main: "__name__", **kwargs) -> None:
        """
        Error assessments with parallel runs.

        Parameters
        ----------
        check_main : "str"
            The special variable "__name__" needed for parallel run.

        """

        verbose = kwargs.pop('verbose', True)

        to = time()
        if verbose:
            print(f"START DEVIATION ASSESSMENT AT: {datetime.now()}")

        # Check is coverage exists:
        if self.coverage is None:
            raise (Exception("A coverage is need to compute errors."))

        # Compute theoretical apparent resistivity:
        if self.rhoa_th is None:
            self.rhoa_th = ErrorComputation.compute_rhoa_theoric(
                self.data, self.model, self.mesh, verbose=verbose)
            self.quadrupoles['rhoa_th'] = self.rhoa_th

        # Save temporary files for sharing them in parallel computing:
        os.makedirs("temp", exist_ok=True)
        mesh_path = os.path.join("temp", "mesh.bms")
        data_path = os.path.join("temp", "data.dat")
        model_path = os.path.join("temp", "model.txt")
        self.mesh.save(mesh_path)
        self.data.save(data_path)
        np.savetxt(model_path, self.model, delimiter=";")

        # ERRORS WITH NEGATIVE VARIATIONS OF THE MODEL:
        # Compute rhoa on the model with negative variations:
        if self.error_neg_min is None:
            rhoa_var_neg = ErrorComputation.run_neg_var(
                self.data, self.model, self.mesh, self.model_var_p,
                self.covered_cells_id, check_main, self.nb_cpu,
                mesh_path, model_path, data_path, self.parallel_start_method,
                verbose=verbose)

        if check_main == '__main__':  # needed for future parallel runs

            # Negative errors with interpolation from negative variations:
            error_neg = ErrorComputation.interpolation_neg(
                self.rhoa_th, rhoa_var_neg, self.nb_data,
                self.covered_cells_id, self.alpha, self.beta, self.k,
                self.model_var_p)

            # Compute negative minimal errors:
            self.error_neg_min = ErrorComputation.extract_min(
                error_neg, self.nb_data, self.mesh, self.masks,
                self.covered_cells_id)

            # ERRORS WITH POSITIVE VARIATIONS OF THE MODEL:
            # Compute rhoa on the model with positive variations:
            if self.error_pos_min is None:
                rhoa_var_pos = ErrorComputation.run_pos_var(
                    self.data, self.model, self.mesh, self.error_neg_min,
                    self.covered_cells_id, check_main, self.nb_cpu,
                    mesh_path, model_path, data_path,
                    self.parallel_start_method, verbose=verbose)

                # positive errors with interpolation from positive variations:
                error_pos = ErrorComputation.interpolation_pos(
                    self.rhoa_th, rhoa_var_pos, self.nb_data,
                    self.covered_cells_id, self.alpha, self.beta, self.k,
                    self.error_neg_min)

                # Compute positive minimal errors:
                self.error_pos_min = ErrorComputation.extract_min(
                    error_pos, self.nb_data, self.mesh, self.masks,
                    self.covered_cells_id)

            # KEEP MINIMAL ERRORS FROM NEGATIVES AND POSITIVES ONES:
            if self.error_min is None:
                # Kept minimal values from 2 arrays:
                self.error_min = np.fmin(self.error_neg_min,
                                         self.error_pos_min)
                self.error_relative = self.error_min

            # COMPUTE THE ABOSLUTE ERROR:
            if self.error_absolute is None:
                self.error_absolute = ErrorComputation.absolute_error(
                    self.model, self.error_relative)

            # Delete temp directory and contents:
            shutil.rmtree("temp")

            tf = time()
            if verbose:
                print("END DEVIATION ASSESSMENT: " +
                      f"{timedelta(seconds=(tf - to))}" + "\n")

    def set_u(self, u: "np.ndarray") -> None:
        self._check_u(u)
        self.u = u

    def set_j(self, j: "np.ndarray") -> None:
        self._check_j(j)
        self.j = j
        self.j_norm = np.array([np.sum(elt**2, axis=1)**.5 for elt in j])

    def set_frechet(self, frechet: "np.ndarray") -> None:
        self._check_frechet(frechet)
        self.frechet = frechet

    def set_masks(self, masks: "np.ndarray") -> None:
        self._check_masks(masks)
        self.masks = masks

    def set_coverage(self, coverage: "np.ndarray") -> None:
        coverage = self._check_coverage(coverage)
        self.coverage = coverage
        self.covered_cells_id = np.where(self.coverage > 0)[0]
        self.profile_mask = np.where(self.coverage > 0, 1, 0)

        # Apply profile mask on resistivity model:
        nan_mask = np.copy(self.profile_mask).astype(float)
        nan_mask[nan_mask == 0] = np.nan
        self.model_mask = self.model * nan_mask

    def set_rhoa_th(self, rhoa_th: "np.ndarray") -> None:
        rhoa_th = self._check_rhoa_th(rhoa_th)
        self.rhoa_th = rhoa_th
        self.quadrupoles['rhoa_th'] = rhoa_th

    def set_error_min(self, error_min: "np.ndarray") -> None:
        error_min = self._check_error_map(error_min)
        self.error_min = error_min

    def set_error_relative(self, error_relative: "np.ndarray") -> None:
        error_relative = self._check_error_map(error_relative)
        self.error_relative = error_relative

    def set_error_absolute(self, error_absolute: "np.ndarray") -> None:
        error_absolute = self._check_error_map(error_absolute)
        self.error_absolute = error_absolute

    def _check_u(self, u: "np.ndarray") -> bool:
        if not isinstance(u, np.ndarray):
            raise (Exception(f"u must be a '{np.ndarray}', given '{type(u)}'"))
        if (ps := u.shape) != (s := (self.data.sensorCount(),
                                     self.mesh.nodeCount())):
            raise (Exception(f"Potential shape {ps} does not match shape"
                             + f" (nb elecs: {s[0]}, nb nodes: {s[1]})."))
        return True

    def _check_j(self, j: "np.ndarray") -> bool:
        if not isinstance(j, list):
            raise (Exception(f"j must be a '{list}', given '{type(j)}'"))
        if (nj := len(j)) != (ns := self.data.sensorCount()):
            raise (Exception(f"Number of j vectors ({nj}) in given list does"
                             + f" not match number of sensors ({ns})."))
        if not all(isinstance(elt, np.ndarray) for elt in j):
            raise (Exception(f"Items of j list must be {np.ndarray}."))
        if not all(elt.shape == (self.mesh.cellCount(), 3) for elt in j):
            raise (Exception("Items of j list shape does not" +
                             " match cell number in mesh with 3 components" +
                             f" ({self.mesh.cellCount()}, 3)."))
        return True

    def _check_frechet(self, frechet: "np.ndaray") -> bool:
        if not isinstance(frechet, np.ndarray):
            raise (Exception(f"frechet must be a '{np.ndarray}', " +
                             f"given '{type(frechet)}'"))
        if (fs := frechet.shape) != (s := (self.data.size(),
                                     self.mesh.cellCount())):
            raise (Exception(f"frechet shape {fs} does not match shape"
                             + f" (nb data: {s[0]}, nb cells: {s[1]})."))
        return True

    def _check_masks(self, masks: "np.ndarray") -> bool:
        if not isinstance(masks, np.ndarray):
            raise (Exception(f"masks must be a '{np.ndarray}', " +
                             f"given '{type(masks)}'"))
        if (ms := masks.shape) != (s := (self.data.size(),
                                         self.mesh.cellCount())):
            raise (Exception(f"masks shape {ms} does not match shape"
                             + f" (nb data: {s[0]}, nb cells: {s[1]})."))
        m = masks.astype(int)
        if len(np.where(m == 0)[0]) + len(np.where(m == 1)[0]) != m.size:
            raise (Exception("masks must contains only 0 or 1."))
        return True

    def _check_coverage(self, coverage: "np.ndarray") -> "np.ndarray":
        if not isinstance(coverage, np.ndarray):
            raise (Exception(f"coverage must be a '{np.ndarray}', " +
                             f"given '{type(coverage)}'"))
        if (cs := coverage.shape[0]) != (s := self.mesh.cellCount()):
            raise (Exception(f"coverage shape {cs} does not nb cell in mesh "
                             + f"({s})."))
        if len(np.where((coverage >= 0) &
                        (coverage <= 1))[0]) != coverage.size:
            raise (Exception("coverage must contains values in [0; 1]."))
        coverage = coverage.reshape((self.mesh.cellCount(),))
        return coverage

    def _check_rhoa_th(self, rhoa_th: "np.ndarray") -> "np.ndarray":
        if not isinstance(rhoa_th, np.ndarray):
            raise (Exception(f"rhoa_th must be a '{np.ndarray}', given"
                             + f" '{type(rhoa_th)}'"))
        if (nv := rhoa_th.shape[0]) != self.nb_data:
            raise (Exception(f"Number of values ({nv}) does not fit number of"
                             + f"data ({self.nb_data})."))
        rhoa_th = rhoa_th.reshape((self.nb_data,))
        return rhoa_th

    def _check_error_map(self, error: "np.ndarray") -> "np.ndarray":
        if not isinstance(error, np.ndarray):
            raise (Exception(f"error must be a '{np.ndarray}', given"
                             + f" '{type(error)}'"))
        if (nv := error.shape[0]) != (nc := self.mesh.cellCount()):
            raise (Exception(f"Number of values ({nv}) does not fit number of"
                             + f"cells ({nc})."))
        error = error.reshape((self.mesh.cellCount(),))
        return error

    def save_quadrupoles(self, path: "str", delimiter=';') -> None:
        """
        Save the quadrupole table into a .csv file.

        Parameters
        ----------
        path : "str"
            Path for save the file.
        delimiter : str, optional
            Delimiter for the .csv file. The default is ';'.

        Returns
        -------
        None

        """
        self.quadrupoles.to_csv(path+'.csv', sep=delimiter)

    def save(self, name: "str", delimiter=";", **kwargs) -> None:
        """
        Save automatically a PyMERRY attribute in a file.

        Parameters
        ----------
        name : "str"
            Name of the attibute to save.
        delimiter : str, optional
            Delimiter for .csv files. The default is ";".
        **kwargs : str
            path: path for save the file.

        Returns
        -------
        None.

        """
        path = kwargs.pop('path', False)

        # For saving norms of j and not 3-components vectors:
        if name == 'j':
            name = 'j_norm'

        # Check if object to save exist in the current instance of MERRY:
        if name not in self.__dict__.keys():
            raise (Exception(f"'{name}' attr does not exist in the PyMERRY" +
                             "instance."))

        # Get attribute name in the PyMERRY instance:
        obj = self.__dict__[name]

        # Check if object to save is not empty:
        if obj is None:
            raise (Exception(f"{name} is empty in the PyMERRY instance."))

        # Set file name and path if given:
        fname = name + '.csv'
        if path is not False:
            fname = os.path.join(path, name + '.csv')

        # Save with right format:
        if isinstance(obj, np.ndarray):

            if obj.shape[0] == self.nb_elecs:

                if obj.shape[1] == (cols := self.mesh.nodeCount()):
                    df = pd.DataFrame(obj, columns=[f"Node {i}"
                                                    for i in range(cols)])
                    df.index.rename("Electrode ID", inplace=True)
                    df.to_csv(fname, sep=delimiter)

                elif obj.shape[1] == (cols := self.mesh.cellCount()):
                    df = pd.DataFrame(obj, columns=[f"Cell {i}"
                                                    for i in range(cols)])
                    df.index.rename("Electrode ID", inplace=True)
                    df.to_csv(fname, sep=delimiter)

            elif obj.shape[0] == self.nb_data:
                try:
                    if obj.shape[1] == (cols := self.mesh.cellCount()):
                        df = pd.DataFrame(obj, columns=[f"Cell {i}"
                                                        for i in range(cols)])
                        df.index.rename("Quadrupole ID", inplace=True)
                        df.to_csv(fname, sep=delimiter)

                    elif obj.shape[1] == (col := len(self.covered_cells_id())):
                        df = pd.DataFrame(obj, columns=[f"Covered cell {i}"
                                                        for i in range(col)])
                        df.index.rename("Quadrupole ID", inplace=True)
                        df.to_csv(fname, sep=delimiter)
                except:
                    df = pd.DataFrame(obj, columns=["rhoa th"])
                    df.index.rename("Quadrupole ID", inplace=True)
                    df.to_csv(fname, sep=delimiter)

            elif obj.shape[0] == self.mesh.cellCount():
                df = pd.DataFrame(obj, columns=[name])
                df.index.rename("Cell ID", inplace=True)
                df.to_csv(fname, sep=delimiter)

        if isinstance(obj, pd.DataFrame):
            obj.to_csv(fname, sep=delimiter)


class MaskComputation:
    """
    Class containing functions for computation of the coverage.
    """

    def compute_potential(survey: "pg.DataContainerERT", model: "np.ndarray",
                          mesh: "pg.Mesh", verbose=False) -> "np.ndarray":
        """
        Compute the electrical potential field u associated with each electrode
        in a survey. u is computed by solving equations of electrical
        propagation in soil by the finite-element solver in PyGimli library.

        Parameters
        ----------
        survey : "pg.DataContainerERT"
            PyGimli object containing the position of each electrode.
        model : "np.ndarray"
            Electrical resistivity model in ohm-m. Shape: (nb cells, ).
        mesh : "pg.Mesh"
            Pygimli mesh object of the domain.
        verbose : "bool", optional
            Write computational time is set to True. The default is False.

        Returns
        -------
        "np.ndarray"
            Potential field u in V computed at each node of the mesh and for
            each electrode.
            Shape: (nb electrodes, nb nodes).

        """
        to = time()
        u = pg.physics.ert.simulate(mesh, survey, model, returnFields=True,
                                    verbose=False)
        tf = time()
        if verbose:
            print(f"  Potential u computed: {round(tf-to, 4)}s.")
        return pg.utils.gmat2numpy(u)

    def compute_current_density(u: "np.ndarray", model: "np.ndarray",
                                mesh: "pg.Mesh",
                                verbose=False) -> "list, np.ndarray":
        """
        Compute current density field j associated with all electrodes in the
        survey. j = -(1/rho)*grad(U).

        Parameters
        ----------
        u : "np.ndarray"
            Potential field u, in V, computed at each node of the mesh and for
            each electrode. Shape: (nb electrodes, nb nodes).
        model : "np.ndarray"
            Electrical resistivity model in ohm-m. Shape: (nb cells, ).
        mesh : "pg.Mesh"
            Pygimli mesh object of the domain.
        verbose : "bool", optional
            Write computational time is set to True. The default is False.

        Returns
        -------
        j : "list(np.ndarray)"
            A list containing j vectors associated with each electrode.
            len: nb electrodes.
            Each item of j is a "np.ndarray" with 3 columns for x, y, and
            z components of j in A/m², respectively.
            Shape: (nb cells, 3).
        j_norm : "np.ndarray"
            Norms of current density vectors j in A/m².
            Shape: (nb electrodes, nb cells).

        """

        to = time()

        # Duplicate model for matrix calculations:
        model_rep = np.repeat(model.reshape(mesh.cellCount(), 1), 3, axis=1)

        # Current density field j = -(1/rho)*grad(U):
        j = [-pg.solver.grad(mesh, u[q, :]) / model_rep
             for q in range(u.shape[0])]

        # Norm of each vector j:
        j_norm = np.array([np.sum(elt**2, axis=1)**.5 for elt in j])

        tf = time()
        if verbose:
            print(f"  Current density j computed: {round(tf-to, 4)}s.")

        return j, j_norm

    def assign_uj(quadrupoles: "pd.DataFrame", u: "np.ndarray",
                  j: "np.ndarray") -> "tuple(pd.DataFrame)":
        """
        Assign the right potential field u and current density j to each
        electrode used in each quadrupole of the survey.

        Parameters
        ----------
        quadrupoles : "pd.DataFrame"
            Table describing all quadrupoles of the survey. In particular
            indexes of all ABMN electrodes used for each quadrupole.
        u : "np.ndarray"
             Potential field u, in V, computed at each node of the mesh and for
             each electrode. Shape: (nb electrodes, nb nodes).
        j : "list(np.ndarray)"
            A list containing j vectors associated with each electrode.
            len: nb electrodes.
            Each item of j is a "np.ndarray" with 3 columns for x, y, and
            z components of j in A/m², respectively.
            Shape: (nb cells, 3).

        Returns
        -------
        u_assignments : "pd.DataFrame"
            A table containing potential field u (np.ndarray) in V associated
            with all ABMN electrodes for each quadrupole.
            Shape: (nb measurements, 4 for ABMN electrodes).
        j_assignments : "pd.DataFrame"
            A table containing 3-component current density vectors j in A/m²
            (np.ndarray) associated with all ABMN electrodes for each
            quadrupole.
            Shape: (nb measurements, 4 for ABMN electrodes).

        """
        u_assignments = pd.DataFrame([[u[quadrupoles.loc[i, "idx_a"]],
                                       u[quadrupoles.loc[i, "idx_b"]],
                                       u[quadrupoles.loc[i, "idx_m"]],
                                       u[quadrupoles.loc[i, "idx_n"]]]
                                      for i in quadrupoles.index],
                                     columns=["a", "b", "m", "n"])

        j_assignments = pd.DataFrame([[j[quadrupoles.loc[i, "idx_a"]],
                                       j[quadrupoles.loc[i, "idx_b"]],
                                       j[quadrupoles.loc[i, "idx_m"]],
                                       j[quadrupoles.loc[i, "idx_n"]]]
                                      for i in quadrupoles.index],
                                     columns=["a", "b", "m", "n"])
        return u_assignments, j_assignments

    def frechet_from_j(ja: "np.ndarray", jb: "np.ndarray", jm: "np.ndarray",
                       jn: "np.ndarray") -> "np.ndarray":
        """
        Compute the Frechet derivative F associated with each quadrupole in
        the survey. F = jA·jM - jB·jM -  jA·jN +  jB·jN in A/m^(-4).

        Parameters
        ----------
        ja : "np.ndarray"
            3-Component current density vector (x, y, z) associated to the
            electrode A. Shape: (nb cells, 3).
        jb : "np.ndarray"
            3-Component current density vector (x, y, z) associated to the
            electrode B. Shape: (nb cells, 3).
        jm : "np.ndarray"
            3-Component current density vector (x, y, z) associated to the
            electrode M. Shape: (nb cells, 3).
        jn : "np.ndarray"
            3-Component current density vector (x, y, z) associated to the
            electrode N. Shape: (nb cells, 3).

        Returns
        -------
        "np.ndarray"
            The Frechet derivative for each quadrupole.
            Shape: (nb measurements, nb cells).

        """
        return (np.sum(ja*jm, axis=1) - np.sum(jb*jm, axis=1) -
                np.sum(ja*jn, axis=1) + np.sum(jb*jn, axis=1))

    def compute_masks(threshold_values_DOI: "np.ndarray",
                      values: "np.ndarray", survey: "pg.DataContainerERT",
                      mesh: "pg.Mesh", verbose=False) -> "np.ndarray":
        """
        Compute binary masks (0; 1) for each quadrupole based on a threshold of
        a quantity named "value" (e.g., the Frechet derivative, jacobian, ...).
        A value equal to 1 seems the cell is investigated by the quadrupole,
        and a value equal to 0 seems the cell is not investigated.

        Parameters
        ----------
        threshold_values_DOI : "np.ndarray"
            Threshold value in the cell located at the depth of investigation
            (DOI) of quadrupoles. Shape: (nb measurements,).
        values : "np.ndarray"
            Values on which the computation of the binary mask is based.
            Shape: (nb measurements, nb cells).
        survey : "pg.DataContainerERT"
            PyGimli object containing the position of each electrode.
        mesh : "pg.Mesh"
            Pygimli mesh object of the domain.
        verbose : "bool", optional
            Write computational time is set to True. The default is False.

        Returns
        -------
        masks : "np.ndarray"
            Binary masks (0; 1) for each quadrupole in the survey.
            Shape: (nb measurements, nb cells).

        """
        to = time()

        # Duplicate threshold values for matrix calculations:
        threshold_DOI_rep = np.repeat(
            threshold_values_DOI.reshape((survey.size(), 1)),
            mesh.cellCount(), axis=1)

        # Compute binary masks:
        masks = np.where(np.abs(values) >= np.abs(threshold_DOI_rep), 1, 0)

        tf = time()
        if verbose:
            print(f"  Masks computed: {round(tf-to, 5)}s.")
        return masks

    def compute_coverage(masks: "np.ndarray", verbose=False) -> "np.ndarray":
        """
        Compute the coverage defined as a normalized sum of each binary
        quadrupole mask. The highest the value, the more the cell is
        covered. 1: fully covered, and 0: not covered.

        Parameters
        ----------
        masks : "np.ndarray"
            Binary masks  (0; 1) for each quadrupole.
            Shape: (nb measurements, nb cells).
        verbose : "bool", optional
            Write computational time is set to True. The default is False.

        Returns
        -------
        coverage : "np.ndarray"
            Coverage array.
            Shape: (nb cells,).

        """
        to = time()

        # Sum of all binary masks:
        coverage = np.sum(masks, axis=0)

        # Set 0 in all minimal value to avoid near-zero numbers:
        min_value = np.nanmin(coverage)
        coverage[coverage <= min_value] = 0

        # Normalization:
        coverage = coverage / np.max(coverage)

        tf = time()
        if verbose:
            print(f"  Coverage computed: {round(tf-to, 4)}s.")
        return coverage


class ErrorComputation:
    """
    Class containing functions for computation of errors.
    """

    def compute_rhoa_theoric(survey: "pg.DataContainerERT",
                             model: "np.ndarray", mesh: "pg.Mesh",
                             verbose=False) -> "np.ndarray":
        """
        Compute theoretical apparent resistivities values in ohm-m for each
        quadrupole on the resistivity model.

        Parameters
        ----------
        survey : "pg.DataContainerERT"
            PyGimli object containing the position of each electrode.
        model : "np.ndarray"
            Electrical resistivity model in ohm-m. Shape: (nb cells, ).
        mesh : "pg.Mesh"
            Pygimli mesh object of the domain.
        verbose : "bool", optional
            Write computational time is set to True. The default is False.

        Returns
        -------
        "np.ndarray"
            Theoretical apparent resistivity for each quadrupole of the
            profile.
            Shape: (nb measurements,).

        """
        to = time()
        rhoa_th = pg.physics.ert.simulate(mesh, survey, model,
                                          returnArray=True, verbose=False)
        tf = time()
        if verbose:
            print(f"  Theoretical rhoa computed: {round(tf-to, 4)}s.")
        return np.array(rhoa_th)

    def extract_min(rho_var: "np.ndarray", nb_data: "int", mesh: "pg.Mesh",
                    masks: "np.ndarray",
                    covered_cells_id: "np.ndarray") -> "np.ndarray":
        """
        Transform rho variations in covered cells to all cells to get vectors
        of the same size as the model, apply masks, and extract the minimal
        variation.

        Parameters
        ----------
        rho_var : "np.ndarray"
            variation of rho in ohm-m in each covered cell of the model.
            Shape: (nb measurment, nb covered cells).
        nb_data : "int"
            The number of measurements (or data, or quadrupoles used).
        mesh : "pg.Mesh"
            Pygimli mesh object of the domain.
        masks : "np.ndarray"
            Binary masks (0; 1) for each quadrupole in the survey.
            Shape: (nb measurements, nb cells).
        covered_cells_id : "np.ndarray"
            Indexes of all covered cells in the mesh.
            Shape: (nb covered cells,).

        Returns
        -------
        err_min : "np.ndarray"
            Minimal rho variation in model defined as minimal error in %:
                Shape: (nb cells,).

        """
        # Set a nan vector of right shape (nb measurements, nb cells):
        rho_var_in_mesh = np.nan * np.ones((nb_data, mesh.cellCount()))
        for q in range(nb_data):
            # Get rho variation and mask for quadrupole q:
            rvq, mq = rho_var[q, :], masks[q, :]

            # Assign rho variation to quadrupole q:
            rho_var_in_mesh[q, covered_cells_id] = rvq

            # Apply quadrupole q mask:
            rho_var_in_mesh[q, np.where(mq == 0)] = np.nan

        # Disable warnings (nan, zero division, ...):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Get minimal variation in each cell as the minimal error:
            err_min = np.nanmin(rho_var_in_mesh, axis=0)

        return err_min

    def _simulation_neg(cell_id: "int", variation: "float", mesh_path: "str",
                        model_path: "str", data_path: "str") -> "np.ndarray":
        """
        Compute apparent resistivity rhoa in ohm-m on a modified model in the
        cell_id-cell. Here, the modification is a diminishing  of the
        resistivity in the cell_id-cell. Will be run with multiprocessing.

        Parameters
        ----------
        cell_id : "int"
            Index of the cell where the model will be modified.
        variation : float
            Percentage to modify the resistivity model in the i-th cell.
        mesh_path : str
            Path to the mesh.bms file.
        model_path : str
            Path to the model.csv file.
        data_path : str
            Path to the data.dat file.

        Returns
        -------
        "np.ndarray"
            Apparent resistivity computed on the modified model.

        """
        # Load files (because no sharing memory with parallel spawn):
        mesh = pg.load(mesh_path)
        model = np.loadtxt(model_path, delimiter=";")
        survey = pg.load(data_path)

        # Modify model:
        model_mod = np.copy(model)
        model_mod[cell_id] = model_mod[cell_id] * (1 - variation)
        return np.array(pg.physics.ert.simulate(mesh, survey,
                                                model_mod, returnArray=True,
                                                verbose=False))

    def _simulation_pos(cell_id: "int", variation: "float", mesh_path: "str",
                        model_path: "str", data_path: "str") -> "np.ndarray":
        """
        Compute apparent resistivity rhoa in ohm-m on a modified model in the
        cell_id-cell. Here, the modification is a increasing  of the
        resistivity in the cell_id-cell. Will be run with multiprocessing.

        Parameters
        ----------
        cell_id : "int"
            Index of the cell where the model will be modified.
        variation : float
            Percentage to modify the resistivity model in the i-th cell.
        mesh_path : str
            Path to the mesh.bms file.
        model_path : str
            Path to the model.csv file.
        data_path : str
            Path to the data.dat file.

        Returns
        -------
        "np.ndarray"
            Apparent resistivity computed on the modified model.

        """
        # Load files (because no sharing memory with parallel spawn):
        mesh = pg.load(mesh_path)
        model = np.loadtxt(model_path, delimiter=";")
        survey = pg.load(data_path)

        # Modify model:
        model_mod = np.copy(model)
        model_mod[cell_id] = model_mod[cell_id] * (1 + variation)
        return np.array(pg.physics.ert.simulate(mesh, survey,
                                                model_mod, returnArray=True,
                                                verbose=False))

    def interpolation_neg(rhoa_th: "np.ndarray", rhoa_var_neg: "np.ndarray",
                          nb_data: "int", covered_cells_id: "np.ndarray",
                          alpha: "float", beta: "float", k: "np.ndarray",
                          model_var_p: "float") -> "np.ndarray":
        """
        Proceed to linear interpolations in 2 segments depending on threshold
        T:
            - AB: for variation between 0 and rhoa_var_neg(model_var_p)
                  if 0 <= T < rhoa_var_neg(model_var_p)
                  var_p = a*delta_rhoa
            - BC: for variation between rhoa_var_neg(model_var_p)
                  if rhoa_var_neg(model_var_p) < T <= rhoa_th
                  var_p = a*delta_rhoa + b
            - C  : -> 1 if T > rhoa_th
        Linear interpolation is used to gain computing time.


        Parameters
        ----------
        rhoa_th : "np.ndarray"
            Theoretical apparent resistivities in ohm-m computed on inversion
            model.
            Shape: (nb measurements,).
        rhoa_var_neg : "np.ndarray"
            Apparent resistivities in ohm-m computed on the modifed model.
            Shape: (nb measurements, nb covered cells).
        nb_data : "int"
            The number of measurements (or data, or quadrupoles used).
        covered_cells_id : "np.ndarray"
            Indexes of all covered cells in the mesh.
            Shape: (nb covered cells,).
        alpha : "float"
            Nominal accuracy of resistivity meter in % concerning current
            intensity injected.
        beta : "float"
            Nominal accuracy of resistivity meter in % concerning current
            potential measured.
        k : "np.ndarray"
            Geometrical factor in m associated with each quadrupole.
            Without taking account of the term "2*pi".
            Shape: (nb measurement, ).
        model_var_p : "float"
            Negative variation of resistivity to apply in each cell of the
            model in %. Set 1 for 100 % and 0.5 for 50 %.

        Returns
        -------
        rho_var_neg_to_t : "np.ndarray"
            Resistivity variation in ohm-m in a cell to reach the instrumental
            threshold.
            Shape: (nb measurements, nb covered cells).

        """
        # Duplicate rhoa_th and k values for matrix calculations:
        rhoa_th_rep = np.repeat(rhoa_th.reshape((nb_data, 1)),
                                len(covered_cells_id), axis=1)

        k = np.repeat(np.transpose(k).reshape((nb_data, 1)),
                      len(covered_cells_id), axis=1)

        # Difference of apparent resistivity between theoretical and modified:
        delta_rhoa = np.abs(rhoa_th_rep - rhoa_var_neg)

        # Instrumental detection threshold T:
        T = (alpha + beta)*np.abs(k)*rhoa_th_rep

        # Set type of interpolations depending on threshold values:
        ab = np.where(T <= delta_rhoa)
        bc = np.where(((delta_rhoa < T) & (T <= rhoa_th_rep)))
        c = np.where(T > rhoa_th_rep)

        # Disable warnings (nan, zero division, ...):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Proceed to interpolations:
            rho_var_neg_to_t = np.zeros(rhoa_th_rep.shape)

            rho_var_neg_to_t[ab] = model_var_p * T[ab] / delta_rhoa[ab]

            rho_var_neg_to_t[bc] = (((1-model_var_p) / (rhoa_th_rep[bc] -
                                                        delta_rhoa[bc])) *
                                    (T[bc] - delta_rhoa[bc] +
                                    (((rhoa_th_rep[bc] -
                                     delta_rhoa[bc])/(1-model_var_p)) *
                                     model_var_p)))

            rho_var_neg_to_t[c] = 1

        return rho_var_neg_to_t

    def interpolation_pos(rhoa_th: "np.ndarray", rhoa_var_pos: "np.ndarray",
                          nb_data: "int", covered_cells_id: "np.ndarray",
                          alpha: "float", beta: "float", k: "np.ndarray",
                          model_var_p: "float"):
        """
        Proceed to linear interpolations in 1 segments.

        Parameters
        ----------
        rhoa_th : "np.ndarray"
            Theoretical apparent resistivities in ohm-m computed on inversion
            model.
            Shape: (nb measurements,).
        rhoa_var_pos : "np.ndarray"
            Apparent resistivities in ohm-m computed on the modifed model.
            Shape: (nb measurements, nb covered cells).
        nb_data : "int"
            The number of measurements (or data, or quadrupoles used).
        covered_cells_id : "np.ndarray"
            Indexes of all covered cells in the mesh.
            Shape: (nb covered cells,).
        alpha : "float"
            Nominal accuracy of resistivity meter in % concerning current
            intensity injected.
        beta : "float"
            Nominal accuracy of resistivity meter in % concerning current
            potential measured.
        k : "np.ndarray"
            Geometrical factor in m associated with each quadrupole.
            Without taking account of the term "2*pi".
            Shape: (nb measurement,).
        model_var_p : "np.ndarray"
            Variation of resistivity appliyed in each cell of the
            model in %.

        Returns
        -------
        rho_var_pos_to_t : "np.ndarray"
            Resistivity variation in ohm-m induced in a cell.
            Shape: (nb measurements, nb covered cells).

        """
        # Remove nan values from model_var_p which is error_neg_min:
        model_var_p = model_var_p[~np.isnan(model_var_p)]

        # Duplicate rhoa_th and k values for matrix calculations:
        rhoa_th_rep = np.repeat(rhoa_th.reshape((nb_data, 1)),
                                len(covered_cells_id), axis=1)

        k = np.repeat(np.transpose(k).reshape((nb_data, 1)),
                      len(covered_cells_id), axis=1)

        # Difference of apparent resistivity between theoretical and modified:
        delta_rhoa = np.abs(rhoa_th_rep - rhoa_var_pos)

        # Instrumental detection threshold T:
        T = (alpha + beta)*np.abs(k)*rhoa_th_rep

        # Disable warnings (nan, zero division, ...):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Proceed to interpolations:
            rho_var_pos_to_t = model_var_p * T / delta_rhoa

        return rho_var_pos_to_t

    def run_neg_var(survey: "pg.DataContainerERT", model: "np.ndarray",
                    mesh: "pg.Mesh", model_var_p: "float",
                    covered_cell_id: "np.ndarray", run_main_check: "str",
                    nb_cpu: "int", mesh_path, model_path, data_path,
                    start_method, verbose=False):
        """
        Run data acquisition simulation in parallel with a negative
        variation of rho in each covered cell.

        Parameters
        ----------
        survey : "pg.DataContainerERT"
            PyGimli object containing the position of each electrode.
        model : "np.ndarray"
            Electrical resistivity model in ohm-m. Shape: (nb cells, ).
        mesh : "pg.Mesh"
            Pygimli mesh object of the domain.
        model_var_p : "float"
            Resistivity variaiton in % to apply at each cell of the model.
            Set 1 for 100 % and 0.95 for 95 %.
        covered_cells_id : "np.ndarray"
            Indexes of all covered cells in the mesh.
            Shape: (nb covered cells,).
        run_main_check : "str"
            Need to be the special variable '__name__' for parallel runs.
        nb_cpu : "int"
            Number of CPU to use for parallel run.
        verbose : "bool", optional
            Write computational time is set to True. The default is False.

        Returns
        -------
        "np.ndarray"
            Apparent resistivity  in ohm-m computed on model modified for each
            quadrupole and for each covered cell.
            Shape(nb measurements, nb covered cells).

        """
        # Check if the variation in lower than 100 %:
        if model_var_p > 1:
            raise (Exception("Model variation must be inferior than 100 %."))

        # List of arguments for each parallel run:
        params = [[i, model_var_p, mesh_path, model_path, data_path]
                  for i in covered_cell_id]

        # Protect entry point of the code:
        if run_main_check == "__main__":
            if verbose:
                print("  Negative model variation test: ... please wait ...")

            # Set a pool of workers, and run it asynchronously:
            if start_method == "spawn":
                to = time()
                with mp.Pool(processes=nb_cpu) as pool:
                    result = pool.starmap_async(
                        ErrorComputation._simulation_neg, params).get()
                    pool.close()
                    pool.join()
                tf = time()

            elif start_method == "fork":
                to = time()
                with mp.Pool(processes=nb_cpu) as pool:
                    result = pool.starmap_async(
                        ErrorComputation._simulation_neg,
                        tqdm.tqdm(params, total=len(params))).get()
                    pool.close()
                    pool.join()
                tf = time()

            if verbose:
                print("  Negative model variation test finished: "
                      + f"{timedelta(seconds=(tf-to))}")

            return np.transpose(np.array(result))

    def run_pos_var(survey: "pg.DataContainerERT", model: "np.ndarray",
                    mesh: "pg.Mesh", model_var_pos: "np.ndarray",
                    covered_cell_id: "np.ndarray", run_main_check: "str",
                    nb_cpu: "int", mesh_path, model_path, data_path,
                    start_method, verbose=False):
        """
        Run data acquisition simulation in parallel with a positive variation
        of rho in each covered cell.

        Parameters
        ----------
        survey : "pg.DataContainerERT"
            PyGimli object containing the position of each electrode.
        model : "np.ndarray"
            Electrical resistivity model in ohm-m. Shape: (nb cells, ).
        mesh : "pg.Mesh"
            Pygimli mesh object of the domain.
        model_var_pos : "np.ndarray"
            Resistivity variaiton in % to apply at each cell of the model.
            Set 1 for 100 % and 0.95 for 95 %.
            Shape: (nb cells, ).
        covered_cells_id : "np.ndarray"
            Indexes of all covered cells in the mesh.
            Shape: (nb covered cells,).
        run_main_check : "str"
            Need to be the special variable '__name__' for parallel runs.
        nb_cpu : "int"
            Number of CPU to use for parallel run.
        verbose : "bool", optional
            Write computational time is set to True. The default is False.

        Returns
        -------
        "np.ndarray"
            Apparent resistivity  in ohm-m computed on model modified for each
            quadrupole and for each covered cell.
            Shape(nb measurements, nb covered cells).

        """
        # Check if the variation in lower than 100 %:
        if np.all(model_var_pos) > 1:
            raise (Exception("Model variation must be inferior than 100 %."))

        # Extract variation in covered cells only:
        var = model_var_pos[~np.isnan(model_var_pos)]

        # List of arguments for each parallel run:
        params = [[i, j, mesh_path, model_path, data_path]
                  for i, j in zip(covered_cell_id, var)]

        # Protect entry point of the code:
        if run_main_check == "__main__":
            if verbose:
                print("  Positive model variation test: ... please wait ...")

            # Set a pool of workers, and run it asynchronously:
            if start_method == "spawn":
                to = time()
                with mp.Pool(processes=nb_cpu) as pool:
                    result = pool.starmap_async(
                        ErrorComputation._simulation_pos, params).get()
                    pool.close()
                    pool.join()
                tf = time()

            elif start_method == "fork":
                to = time()
                with mp.Pool(processes=nb_cpu) as pool:
                    result = pool.starmap_async(
                        ErrorComputation._simulation_pos,
                        tqdm.tqdm(params, total=len(params))).get()
                    pool.close()
                    pool.join()
                tf = time()

            if verbose:
                print("  Positive model variation test finished: "
                      + f"{timedelta(seconds=(tf-to))}")

            return np.transpose(np.array(result))

    def absolute_error(model: "np.ndarray", error: "np.ndarray"):
        """
        Compute absolute error in ohm-m on a resistivity model.

        Parameters
        ----------
         model : "np.ndarray"
             Electrical resistivity model in ohm-m. Shape: (nb cells, ).
        error : "np.ndarray"
            Relative error in %. Shape: (nb cells, ).

        Returns
        -------
        "np.ndarray"
            Absolute error in ohm-m. Shape: (nb cells, ).

        """
        return model * error


class InputTools:
    """
    Class containing functions for reading input files and parameters files.
    """
    def read_param_file(file_name: str) -> dict:
        """
        Read specific lines from a parameter .txt file.
        And retur a difctionary with parameter value or str.

        Parameters
        ----------
        file_name : str
            Path and/or name of the parameter.txt file.

        Returns
        -------
        dict_log : dict
            A dictionary containing value or str from parameter file:
                'cells_file', 'nodes_file', 'model_file', 'data_file',
                'save_dir', 'alpha', 'beta', 'DOI', 'DOI_custom',
                'gamma', 'cmin', 'cmax' and 'logscale'.

        """
        # Open file and read specific lines:
        file = open(file_name, "r")
        lines = file.readlines()
        lines = [lines[i-1]
                 for i in [10, 11, 12, 13, 19, 24, 25, 28, 29, 35, 36, 37, 38]]

        # Prepare a dictionary:
        dict_log = {}
        keys = ["cells_file", "nodes_file", "model_file", "data_file",
                "save_dir", "alpha", "beta", "DOI", "DOI_custom",
                "gamma", "cmin", "cmax", "logscale"]

        # Browse the selected lines and fill the dictionary:
        for i, line in enumerate(lines):
            line.replace(" ", "")
            sep = line.split(':', 1)
            value = sep[1].strip()
            try:
                dict_log[keys[i]] = float(value)
            except:
                dict_log[keys[i]] = value
        return dict_log

    def mesh_tables(mesh: "pg.Mesh") -> "tuple(np.ndarray)":
        """
        Get nodes table and cells table from a mesh.
        Only for a triangular cells mesh.

        nodes_table columns: node id, x, y, z
        cells_table columns: cell id, node1_id, node2_id, node3_id

        Parameters
        ----------
        mesh : pg.Mesh
            Pygimli mesh object of the domain.

        Returns
        -------
        nodes_table, cells, table"tuple(np.ndarray)"
            np.ndarray, np.ndarray describing the mesh.

        """
        # Create table of nodes: [id, x, y, z]
        nodes_table = [[n.id(), n.x(), n.y(), n.z()]
                       for n in mesh.nodes()]

        # Create table of cells: [if, n1, n2, n3]
        cells_table = [[c.id(), c.node(0).id(), c.node(1).id(), c.node(2).id()]
                       for c in mesh.cells()]
        return np.array(nodes_table), np.array(cells_table)

    def get_boundaries_table(nodes_table: "np.ndarray",
                             cells_table: "np.ndarray") -> "np.ndarray":
        """
        Get boundaries table from nodes and cells table.

        Parameters
        ----------
        nodes_table : np.ndarray
            Nodes_table columns: node id, x, y, z.
        cells_table : np.ndarray
            Cells_table columns: cell id, node1_id, node2_id, node3_id.

        Returns
        -------
        boundaries_table : np.ndarray
            Boundaries table with columns: bound_id, node1_id, node2_id.

        """
        # Extract the 3 boundaries from each triangular cells:
        boundaries_table = []
        for line in range(cells_table.shape[0]):
            cell = cells_table[line, :][1:]
            cell_bounds = list(itertools.combinations(cell, 2))
            [boundaries_table.append(bounds) for bounds in cell_bounds]

        # Delete redundent boundaries (set <=> np.unique for tuples):
        boundaries_table = set(tuple(sorted(tup)) for tup in boundaries_table)

        # Turn the table into a list of lists [node 1 id, node 2 id]:
        boundaries_table = np.array([[e[0], e[1]] for e in boundaries_table])

        # Tranfromt into numpy array:
        bounds_id = np.arange(boundaries_table.shape[0])
        bounds_id = bounds_id.reshape((boundaries_table.shape[0], 1))

        # Add boundary ID at first column:
        boundaries_table = np.concatenate(
            (bounds_id, boundaries_table), axis=1)

        return boundaries_table

    def get_box_bounds_id(mesh: "pg.Mesh") -> "np.ndarray":
        """
        Get the list of boundaries ID at the border of the domain.

        Parameters
        ----------
        mesh : pg.Mesh
            Pygimli mesh object of the domain.

        Returns
        -------
        border_bounds_id: np.ndarray
            Array containing all boudaries ID at the border of the domain.

        """
        # Extract the 3 boundaries from each triangular cells:
        bound_by_cells = np.array([[c.boundary(0).id(), c.boundary(1).id(),
                                    c.boundary(2).id()] for c in mesh.cells()])

        # Count occurence of each boundary:
        counts = Counter(list(np.array(bound_by_cells).flatten()))

        # Kept only boundaries with occurence = 1:
        border_bounds_id = [k for k in counts.keys() if counts[k] == 1]

        border_bounds_id.sort()
        return np.array(border_bounds_id)

    def retrive_mesh(nodes_table: "np.ndarray",
                     cells_table: "np.ndarray") -> "pg.Mesh":
        """
        Built a 2D triangular cell mesh from nodes and cells table.
           - nodes_table columns: node id, x, y, z.
           - cells_table columns: cell id, node1_id, node2_id, node3_id.

        the retrived mesh contain nodes, boundaries and cells elements.
        All cells will have a marker = 1.
        Outer borders boundaries will have a marker = -2.
        Surface boundaries will have a marker = -1.
        other boundaries will have marker = None.

        Parameters
        ----------
        nodes_table : np.ndarray
            Nodes_table columns: node id, x, y, z.
        cells_table : np.ndarray
            Cells_table columns: cell id, node1_id, node2_id, node3_id.

        Returns
        -------
        mesh : "pg.Mesh"
            2D Triangular cell mesh for the domain.

        """
        # Create a 2D mesh object:
        mesh = pg.Mesh(2)

        # Create nodes with from [id, x, y, z]:
        nodes = [mesh.createNode((n[1], n[2], n[3])) for n in nodes_table]

        # Create boundaries table: [id, n1, n2]
        bounds_table = InputTools.get_boundaries_table(nodes_table,
                                                       cells_table)

        # Create boundaries from [id, n1, n2] with marker=None (for now):
        [mesh.createEdge(mesh.node(b[1]), mesh.node(b[2]))
         for b in bounds_table]

        # Create cells from [id, n1, n2, n3] with marker=1:
        [mesh.createTriangle(nodes[c[1]], nodes[c[2]], nodes[c[3]], marker=1)
         for c in cells_table]

        # Get ids of border boundaries:
        border_bounds_ids = InputTools.get_box_bounds_id(mesh)

        # Assign (-1) marker for all border bounds:
        [mesh.boundary(i).setMarker(-1) for i in border_bounds_ids]

        # Assign (-2) marker for right, bottom, left margin of the box:
        # Bottom:
        [mesh.boundary(bid).setMarker(-2) for bid in border_bounds_ids
         if (
            (mesh.boundary(bid).node(0).y() == mesh.yMin())
            and
            (mesh.boundary(bid).node(1).y() == mesh.yMin()))]

        # Left:
        [mesh.boundary(bid).setMarker(-2) for bid in border_bounds_ids
         if (
            (mesh.boundary(bid).node(0).x() == mesh.xMin())
            and
            (mesh.boundary(bid).node(1).x() == mesh.xMin()))]

        # Right:
        [mesh.boundary(bid).setMarker(-2) for bid in border_bounds_ids
         if (
            (mesh.boundary(bid).node(0).x() == mesh.xMax())
            and
            (mesh.boundary(bid).node(1).x() == mesh.xMax()))]

        return mesh

    def sort_by_first_col(array: "np.ndarray") -> "np.ndarray":
        """
        Sort an array by its first column in ascending order.

        Parameters
        ----------
        array : np.ndarray
            An array.

        Returns
        -------
        array: np.ndarray
            same array but ordered by ascending order of the first column.

        """
        return np.array(pd.DataFrame(array).sort_values(by=[0]))

    def convert_path(path: "str") -> str:
        """
        Convert a path containig "\" and/or "/" separator into a path
        with the right separator for the operating system used.

        Parameters
        -----------
        path : str
            A path to convert with right separators.

        Returns
        -------
        path : str
            Path converted with right separators.

        """
        # Split the chain in sub chains at "\" char:
        sub_anti = path.split("\\")

        # Split all sub chains at "/" char and save all parts into a list:
        sub_slas = [c.split("/") for c in sub_anti]

        # Get and unique list of all part of the chain:
        elts = []
        for e in sub_slas:
            if len(e) == 1:
                elts.append(e[0])
            elif len(e) > 1:
                for i in e:
                    elts.append(i)

        # Return the complete chain with right separator with os module:
        return os.path.join(*elts)

    def load_parameters(file_name: str) -> dict:
        """
        Prepare all the inputs objects for PyMERRY from a parameter file.

        Parameters
        ----------
        file_name : str
            The parameter.txt file.

        Returns
        -------
        input_dict : dict
            A dictionary containing the requested objects and parameters to
            instanciate MERRY object. keys are : "data", "mesh",  "model",
            "doi_W", "doi_DD","doi_WS", "alpha", "beta", "save_dir",
            "param_dict", "gamma", "cmin", "cmax", "logscale".

        """
        # Read parameter.txt file and get a dict:
        param_dict = InputTools.read_param_file(file_name)

        # Convert paths for use them whatever the OS used:
        cells_path = os.path.join(param_dict["cells_file"])
        nodes_path = os.path.join(param_dict["nodes_file"])
        model_path = os.path.join(param_dict["model_file"])
        data_path = os.path.join(param_dict["data_file"])
        cells_path = InputTools.convert_path(cells_path)
        nodes_path = InputTools.convert_path(nodes_path)
        model_path = InputTools.convert_path(model_path)
        data_path = InputTools.convert_path(data_path)

        # Load input files for mesh, model and data:
        cells = np.loadtxt(cells_path, delimiter=";", dtype=int)
        nodes = np.loadtxt(nodes_path, delimiter=";")
        model = np.loadtxt(model_path, delimiter=";")
        data = pg.load(data_path)

        # Ensure the right order for cells ids and nodes ids:
        cells = InputTools.sort_by_first_col(cells)
        nodes = InputTools.sort_by_first_col(nodes)
        model = InputTools.sort_by_first_col(model)

        # Check DOIs:
        doi = param_dict["DOI"]
        doi_custom = param_dict["DOI_custom"]
        try:
            doi_custom = doi_custom.lower()
        except:
            pass

        if doi_custom != "none":
            try:
                doi_W = float(doi_custom)
                doi_DD = float(doi_custom)
                doi_WS = float(doi_custom)
            except:
                pass
        else:
            if doi == "A":  # Roy and apparao, 1971 choice
                doi_W = 0.11
                doi_DD = 0.195
                doi_WS = 0.125
            elif doi == "B":  # Baker, 1989 choice
                doi_W = 0.17
                doi_DD = "adaptative"
                doi_WS = 0.19
            else:
                raise (Exception(f"DOI has to be 'A' or 'B', not {doi}."))

        # Check min / max for plot:
        cmin = param_dict["cmin"]
        if cmin == "from model":
            cmin = np.nanmin(model[:, 1])
        else:
            try:
                cmin = float(cmin)
            except:
                print("Color bar min value set failed. Set to model min.")
                cmin = np.nanmin(model[:, 1])

        cmax = param_dict["cmax"]
        if cmax == "from model":
            cmax = np.nanmax(model[:, 1])
        else:
            try:
                cmax = float(cmax)
            except:
                print("Color bar max value set failed. Set to model max.")
                cmax = np.nanmax(model[:, 1])

        # Convert logscale variable from str to bool:
        if param_dict["logscale"].lower() == "true":
            logscale = True
        elif param_dict["logscale"].lower() == "false":
            logscale = False

        # Prepare a input dict for run PyMERRY:
        input_dict = {}
        input_dict["data"] = data
        input_dict["mesh"] = InputTools.retrive_mesh(nodes, cells)
        input_dict["model"] = model[:, 1]
        input_dict["doi_W"] = doi_W
        input_dict["doi_DD"] = doi_DD
        input_dict["doi_WS"] = doi_WS
        input_dict["alpha"] = param_dict["alpha"] / 100
        input_dict["beta"] = param_dict["beta"] / 100
        input_dict["save_dir"] = param_dict["save_dir"]
        input_dict["param_dict"] = param_dict
        input_dict["gamma"] = param_dict["gamma"]
        input_dict["cmin"] = cmin
        input_dict["cmax"] = cmax
        input_dict["logscale"] = logscale

        return input_dict

    def save_parameters_file(inputs: dict) -> None:
        """
        Save the used parameters in a .txt file.

        Parameters
        ----------
        inputs : dict
            A dictionary containing the requested objects and parameters to
            instanciate MERRY object. keys are : "data", "mesh",  "model",
            "doi_W", "doi_DD","doi_WS", "alpha", "beta", "save_dir",
            "param_dict", "gamma", "cmin", "cmax", "logscale".

        Returns
        -------
        None.

        """
        string = ""
        string += "PyMERRY PARAMETER FILE\n"
        string += "\n"
        string += "WARNINGS !\n"
        string += "/!\ DO NOT ADD, DELETE OR MOVE LINES !\n"
        string += "/!\ ONLY MODIFY TEXTS AT THE RIGH OF ':' AT LINES MARKED WITH '->' !\n"
        string += "\n"
        string += "\n"
        string += "INPUT FILES PATHS\n"
        string += "Indicate here paths (without spaces) for mesh, model and data files.\n"
        string += f"-> mesh cells table  : {inputs['param_dict']['cells_file']}\n"
        string += f"-> mesh nodes table  : {inputs['param_dict']['nodes_file']}\n"
        string += f"-> resistivity model : {inputs['param_dict']['model_file']}\n"
        string += f"-> data file         : {inputs['param_dict']['data_file']}\n"
        string += "\n"
        string += "\n"
        string += "OUTPUT DIRECTORY\n"
        string += "Indicate here the name of the directory for save the results. If the directory\n"
        string += "does not exist, it will be created.\n"
        string += f"-> directory to save results : {inputs['param_dict']['save_dir']}\n"
        string += "\n"
        string += "\n"
        string += "PyMERRY PARAMETERS\n"
        string += "Device accuracies in percentage (default: 0.2 %):\n"
        string += f"-> resistivity-meter injection accuracy : {inputs['param_dict']['alpha']}\n"
        string += f"-> resistivity-meter potential accuracy : {inputs['param_dict']['beta']}\n"
        string += "\n"
        string += "Enter your choice for Depth Of Investigation (DOI) coefficient:\n"
        string += f"-> DOI Apparao (A), or Baker (B)                             :{inputs['param_dict']['DOI']}\n"
        string += f"-> DOI Custom* (set a value (example 0.3) or none if unused) : {inputs['param_dict']['DOI_custom']}\n"
        string += "\n"
        string += "\n"
        string += "PLOT PARAMETERS\n"
        string += "Enter parameters for display results.\n"
        string += "(default gamma = 0.25, cmin/cmax=from model):\n"
        string += f"-> gamma                       : {inputs['gamma']}\n"
        string += f"-> color bar min value (ohm-m) : {inputs['cmin']}\n"
        string += f"-> color bar max value (ohm-m) : {inputs['cmax']}\n"
        string += f"-> color bar in log scale      : {inputs['logscale']}\n"
        string += "\n"
        string += "\n"
        string += "--------------------------------------------------------------------------\n"
        string += "|INFORMATION ABOUT CHOICE OF DOI COEFFICIENT (DEPTH OF INVESTIGATION)    |\n"
        string += "|                                                                        |\n"
        string += "|    array type            Roy & Apparao (1971)       Baker (1989)       |\n"
        string += "|                                                                        |\n"
        string += "|     Wenner                   0.11                     0.17             |\n"
        string += "|  Dipole-Dipole               0.195                     **              |\n"
        string += "|Wenner-Schlumberger           0.125                    0.19             |\n"
        string += "|                                                                        |\n"
        string += "|*  If DOI Custom is used, all DOIs (W, WS, DD) will take this value.    |\n"
        string += "|** The Dipole-Dipole DOI value is adaptative to the b/L ratio           |\n"
        string += "|   (see Baker, 1989).                                                   |\n"
        string += "--------------------------------------------------------------------------\n"
        string += ""
        output_path = os.path.join(
            inputs['save_dir'], "parameters_used" + '.txt')
        with open(output_path, 'w', encoding='utf-8') as log_file:
            log_file.write(string)
        return


class Plots:
    """ Class containing plot functions."""

    def plot_error_bars(model_mask: "np.ndarray", profile_mask: "np.ndarray",
                        error_absolute: "np.ndarray", mesh: "pg.Mesh",
                        cmin: "float", cmax: "float", gamma: "float",
                        logscale: "bool", fig, ax1, ax2, ax3) -> tuple:
        """
        Create the error bar plot in 3 windows as :
            1) model - error in covered cells
            2) model in covered cells
            3) model + error in covered cells

        Parameters
        ----------
        model_mask : np.ndarray
            Resistivity model with np.nan in non covered cells.
            Shape: (nb cells, ).
        profile_mask : np.ndarray
            Binary mask at the profil scale.
            Shape: (nb cells, ).
        error_absolute : np.ndarray
            Absolute error in ohm-m.
            Shape: (nb cells, ).
        mesh : pg.Mesh
            pyGimli mesh object of the domain.
        cmin : float
            Minimal value for the color bar.
        cmax : float
            Maximal value for the color bar.
        gamma : float
            DESCRIPTION.
        logscale : bool
            Turn on (True) of off (False) the log scale color bar.
        fig : matplotlib.figure.Figure
            Figure.
        ax1 : matplotlib.axes._axes.Axes
            ax for plot model - error.
        ax2 : matplotlib.axes._axes.Axes
            ax for plot model.
        ax3 : matplotlib.axes._axes.Axes
            ax for plot model + error.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure.
        ax1 : matplotlib.axes._axes.Axes
            ax for plot model - error.
        ax2 : matplotlib.axes._axes.Axes
            ax for plot model.
        ax3 : matplotlib.axes._axes.Axes
            ax for plot model + error.

        """
        mean_model = np.nanmean(model_mask)
        err_neg = model_mask - error_absolute
        err_pos = model_mask + error_absolute
        delta_mean_limit = gamma * mean_model

        ids_lo = np.array(
            [c.id() for c in mesh.cells()
             if profile_mask[c.id()] > 0
             and model_mask[c.id()] <= mean_model + delta_mean_limit])

        # ids_lo_comp = np.setdiff1d(np.arange(0, len(model_mask), 1), ids_lo)

        ids_up = np.array(
            [c.id() for c in mesh.cells()
             if profile_mask[c.id()] > 0
             and model_mask[c.id()] >= mean_model - delta_mean_limit])

        # ids_up_comp = np.setdiff1d(np.arange(0, len(model_mask), 1), ids_up)

        model_lo = np.copy(model_mask)
        model_up = np.copy(model_mask)

        for i in ids_lo:
            model_lo[i] = err_neg[i]

        for i in ids_up:
            model_up[i] = err_pos[i]

        model_lo = ma.masked_array(model_lo, np.logical_not(profile_mask))
        model_up = ma.masked_array(model_up, np.logical_not(profile_mask))

        ax1, cb = pg.show(mesh, data=model_up, showMesh=False, ax=ax1,
                          cMap="Spectral_r", cMin=cmin, cMax=cmax,
                          logScale=logscale,
                          label="High resistivity (ohm-m)")

        ax2, cb = pg.show(mesh, data=model_mask, showMesh=False, ax=ax2,
                          cMap="Spectral_r", cMin=cmin, cMax=cmax,
                          logScale=logscale,
                          label="Resistivity model (ohm-m)")

        ax3, cb = pg.show(mesh, data=model_lo, showMesh=False, ax=ax3,
                          cMap="Spectral_r", cMin=cmin, cMax=cmax,
                          logScale=logscale,
                          label="Low resistivity (ohm-m)")

        return fig, ax1, ax2, ax3
