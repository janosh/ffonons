"""Locally run atomate2 PhononMaker on PhononDB, MP or GNoME supercells."""

# %%
import json
import os
import re
import shutil
from glob import glob
from time import perf_counter
from zipfile import BadZipFile

import atomate2.forcefields.jobs as ff_jobs
import pandas as pd
import plotly.graph_objects as go
import pymatviz as pmv
import torch
from atomate2.common.schemas.phonons import PhononBSDOSDoc as Atomate2PhononBSDOSDoc
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.utils import MLFF
from IPython.display import display
from jobflow import run_locally
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
from pymatviz.enums import Key
from tqdm import tqdm

from ffonons import DATA_DIR, PDF_FIGS, ROOT
from ffonons.dbs.phonondb import PhononDBDocParsed
from ffonons.enums import DB
from ffonons.plots import plotly_title

__author__ = "Janosh Riebesell"
__date__ = "2023-11-19"

# make go.Figure.show() a no-op
go.Figure.show = lambda *_args, **_kwargs: None


# %%
which_db = DB.phonon_db
RUNS_DIR = f"{ROOT}/tmp/runs"  # noqa: S108
PH_DOCS_DIR = f"{DATA_DIR}/{which_db}"
FIGS_DIR = f"{PDF_FIGS}/{which_db}"
shutil.rmtree(RUNS_DIR, ignore_errors=True)  # remove old runs to save space
for directory in (PH_DOCS_DIR, FIGS_DIR, RUNS_DIR):
    os.makedirs(directory, exist_ok=True)

common_relax_kwds = dict(fmax=0.00001)
mace_kwds = dict(model="medium")
chgnet_kwds = dict(optimizer_kwargs=dict(use_device="mps"))
s7net_kwds = dict(model="SevenNet-0")

all_model_kwargs = {
    MLFF.MACE: dict(model="medium", default_dtype="float64"),
    MLFF.M3GNet: dict(model="medium"),
    MLFF.CHGNet: dict(optimizer_kwargs=dict(use_device="mps")),
    MLFF.SevenNet: dict(model="SevenNet-0"),
}


do_mlff_relax = True  # whether to MLFF-relax the PBE structure
relax_cell = True  # whether to only relax atom position at fixed unit cell

with open(f"{DATA_DIR}/mp-ids-with-pbesol-phonons.yml") as file:
    mp_ids = file.read().splitlines()[2:]
bad_ids = [mp_id for mp_id in mp_ids if not mp_id.startswith("mp-")]
if len(bad_ids) != 0:
    raise RuntimeError(f"{bad_ids=}")


# %% check existing and missing DFT/ML phonon docs
dft_docs = glob(f"{DATA_DIR}/{which_db}/mp-*-pbe.json.xz")
pbe_ids = [re.search(r"mp-\d+", path).group() for path in dft_docs]

total_missing_ids, df_missing = set(), pd.DataFrame()
for model_name, _kwargs in all_model_kwargs.values():
    model_docs = glob(f"{DATA_DIR}/{which_db}/*-{model_name}.json.xz")
    model_ids = [re.search(r"mp-\d+", path).group() for path in model_docs]
    missing_ids = {*pbe_ids} - {*model_ids}
    total_missing_ids |= missing_ids
    df_missing[model_name.label] = {"missing": len(missing_ids), "have": len(model_ids)}

missing_paths = [
    path for path in dft_docs if any(mp_id in path for mp_id in total_missing_ids)
]

caption = (
    f"found {len(dft_docs):,} {which_db} DFT phonon docs<br>"
    f"total missing: {len(total_missing_ids):,}<br><br>"
)
display(df_missing.T.style.set_caption(caption))


# %% Main loop over materials and models
errors: list[tuple[str, str, str]] = []
skip_existing = True

for dft_doc_path in (pbar := tqdm(missing_paths)):  # PhononDB
    mat_id = "-".join(dft_doc_path.split("/")[-1].split("-")[:2])
    pbar.set_description(f"{mat_id=}")
    if not re.match(r"mp-\d+", mat_id):
        raise ValueError(f"Invalid {mat_id=}")

    with zopen(dft_doc_path, mode="rt") as file:
        phonondb_doc: PhononDBDocParsed = json.load(file, cls=MontyDecoder)

    struct = phonondb_doc.structure
    supercell = phonondb_doc.supercell
    pbe_dos = phonondb_doc.phonon_dos
    pbe_bands = phonondb_doc.phonon_bandstructure
    struct.properties[Key.mat_id] = mat_id
    formula = struct.formula.replace(" ", "")

    for pkg_name, (model_name, model_kwargs) in all_model_kwargs.items():
        os.makedirs(root_dir := f"{RUNS_DIR}/{model_name.value}", exist_ok=True)

        ml_doc_path = f"{PH_DOCS_DIR}/{mat_id}-{formula}-{model_name.value}.json.xz"

        if os.path.isfile(ml_doc_path) and skip_existing:
            # skip if ML doc exists, can easily generate bs_dos_fig from that without
            # rerunning workflow
            print(f"\nSkipping {model_name!s} for {mat_id}: phonon doc file exists")
            continue
        try:
            start = perf_counter()
            phonon_flow = PhononMaker(
                bulk_relax_maker=ff_jobs.ForceFieldRelaxMaker(
                    force_field_name=model_name,
                    relax_kwargs=common_relax_kwds,
                    calculator_kwargs=model_kwargs,
                    relax_cell=relax_cell,
                )
                if do_mlff_relax
                else None,
                phonon_displacement_maker=ff_jobs.ForceFieldStaticMaker(
                    force_field_name=pkg_name, calculator_kwargs=model_kwargs
                ),
                static_energy_maker=ff_jobs.ForceFieldStaticMaker(
                    force_field_name=pkg_name, calculator_kwargs=model_kwargs
                ),
                store_force_constants=False,
                # use "setyawan_curtarolo" when comparing to MP and "seekpath" else
                # since setyawan_curtarolo only compatible with primitive cell
                kpath_scheme="setyawan_curtarolo" if which_db == DB.mp else "seekpath",
                create_thermal_displacements=False,
                # use_symmetrized_structure="primitive",
                displacement=0.01,
            ).make(structure=struct, supercell_matrix=supercell)

            result = run_locally(
                phonon_flow, root_dir=root_dir, log=True, ensure_success=True
            )
            print(f"\n{model_name} took: {perf_counter() - start:.2f} s")

            last_job_id = phonon_flow[-1].uuid
            ml_phonon_doc: Atomate2PhononBSDOSDoc = result[last_job_id][1].output

            with zopen(ml_doc_path, mode="wt") as file:
                json.dump(ml_phonon_doc, file, cls=MontyEncoder)

            ml_bs, ml_dos = ml_phonon_doc.phonon_bandstructure, ml_phonon_doc.phonon_dos
            bands_dict = {model_name.label: ml_bs}
            dos_dict = {model_name.label: ml_dos}
            if "pbe_dos" in locals() and "pbe_bands" in locals():
                dos_dict[Key.pbe.label] = pbe_dos
                bands_dict[Key.pbe.label] = pbe_bands

            fig_bs_dos = pmv.phonon_bands_and_dos(bands_dict, dos_dict)
            fig_bs_dos.layout.title = dict(
                text=plotly_title(formula, mat_id), x=0.5, y=0.97
            )
            fig_bs_dos.layout.margin = dict(t=40, b=0, l=5, r=5)
            fig_bs_dos.layout.legend.update(x=1, y=1.07, xanchor="right")
            fig_bs_dos.show()

            img_name = f"{mat_id}-bs-dos-{Key.pbe}-vs-{model_name.value}"
            pmv.save_fig(fig_bs_dos, f"{FIGS_DIR}/{img_name}.pdf")
        except (ValueError, RuntimeError, BadZipFile, Exception) as exc:
            # known possible errors:
            # - the 2 band structures are not compatible, due to symmetry change during
            # MACE relaxation, try different PhononMaker symprec (default=1e-4). compare
            # PBE and MACE space groups to verify cause
            # - phonopy found imaginary dispersion > 1e-10 (fixed by disabling thermal
            # displacement matrices)
            # - phonopy-internal: RuntimeError: Creating primitive cell failed.
            # PRIMITIVE_AXIS may be incorrectly specified. For mp-754196 Ba2Sr1I6
            # faulty downloads of phonondb docs raise "BadZipFile: is not a zip file"
            # - mp-984055 raised: [1] 51628 segmentation fault
            # multiprocessing/resource_tracker.py:254: UserWarning: There appear to be 1
            # leaked semaphore objects to clean up at shutdown
            if "supercell" in locals():
                exc.add_note(f"{supercell=}")
            errors += [(mat_id, model_name, formula)]

        # MACE annoyingly changes the torch default dtype which breaks CHGNet
        # and M3GNet, so we reset it here
        torch.set_default_dtype(torch.float32)

if errors:
    print(f"\n{errors=}")
