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
import torch
from atomate2.common.schemas.phonons import PhononBSDOSDoc as Atomate2PhononBSDOSDoc
from atomate2.forcefields.flows.phonons import PhononMaker
from IPython.display import display
from jobflow import run_locally
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
from pymatviz import plot_phonon_bands_and_dos
from pymatviz.io import save_fig
from tqdm import tqdm

from ffonons import DATA_DIR, PDF_FIGS, ROOT
from ffonons.dbs.phonondb import PhononDBDocParsed
from ffonons.enums import DB, Key, Model
from ffonons.plots import plotly_title

__author__ = "Janosh Riebesell"
__date__ = "2023-11-19"


# %%
which_db = DB.phonon_db
RUNS_DIR = f"{ROOT}/tmp/runs"  # noqa: S108
PH_DOCS_DIR = f"{DATA_DIR}/{which_db}"
FIGS_DIR = f"{PDF_FIGS}/{which_db}"
shutil.rmtree(RUNS_DIR, ignore_errors=True)  # remove old runs to save space
for directory in (PH_DOCS_DIR, FIGS_DIR, RUNS_DIR):
    os.makedirs(directory, exist_ok=True)

common_relax_kwds = dict(fmax=0.00001)
mace_kwds = dict(model="https://tinyurl.com/y7uhwpje")
chgnet_kwds = dict(optimizer_kwargs=dict(use_device="mps"))

do_mlff_relax = True  # whether to MLFF-relax the PBE structure
models = {
    Model.mace_mp: dict(
        bulk_relax_maker=ff_jobs.MACERelaxMaker(
            relax_kwargs=common_relax_kwds,
            model_kwargs={"default_dtype": "float64"},
            **mace_kwds,
        )
        if do_mlff_relax
        else None,
        phonon_displacement_maker=ff_jobs.MACEStaticMaker(**mace_kwds),
        static_energy_maker=ff_jobs.MACEStaticMaker(**mace_kwds),
    ),
    Model.m3gnet_ms: dict(
        bulk_relax_maker=ff_jobs.M3GNetRelaxMaker(relax_kwargs=common_relax_kwds)
        if do_mlff_relax
        else None,
        phonon_displacement_maker=ff_jobs.M3GNetStaticMaker(),
        static_energy_maker=ff_jobs.M3GNetStaticMaker(),
    ),
    Model.chgnet_030: dict(
        bulk_relax_maker=ff_jobs.CHGNetRelaxMaker(
            relax_kwargs=common_relax_kwds | {"assign_magmoms": False}, **chgnet_kwds
        )
        if do_mlff_relax
        else None,
        phonon_displacement_maker=ff_jobs.CHGNetStaticMaker(
            **chgnet_kwds, model_kwargs={"assign_magmoms": False}
        ),
        static_energy_maker=ff_jobs.CHGNetStaticMaker(
            **chgnet_kwds, model_kwargs={"assign_magmoms": False}
        ),
    ),
}


with open(f"{DATA_DIR}/mp-ids-with-pbesol-phonons.yml") as file:
    mp_ids = file.read().splitlines()[2:]
bad_ids = [mp_id for mp_id in mp_ids if not mp_id.startswith("mp-")]
if len(bad_ids) != 0:
    raise RuntimeError(f"{bad_ids=}")


# %% check existing and missing DFT/ML phonon docs
dft_docs = glob(f"{DATA_DIR}/{which_db}/mp-*-pbe.json.lzma")
pbe_ids = [re.search(r"mp-\d+", path).group() for path in dft_docs]
print(f"found {len(dft_docs)} DFT phonon docs")

total_missing, df_missing = set(), pd.DataFrame()
for model in models:
    model_docs = glob(f"{DATA_DIR}/{which_db}/*-{model}.json.lzma")
    model_ids = [re.search(r"mp-\d+", path).group() for path in model_docs]
    missing_ids = {*pbe_ids} - {*model_ids}
    total_missing |= missing_ids
    df_missing[model.label] = {"missing": len(missing_ids), "have": len(model_ids)}

missing_paths = [
    path for path in dft_docs if any(mp_id in path for mp_id in total_missing)
]

caption = (
    f"matching DFT docs: {len(dft_docs)}<br>total missing: {len(total_missing)}<br><br>"
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

    with zopen(dft_doc_path, "rt") as file:
        phonondb_doc: PhononDBDocParsed = json.load(file, cls=MontyDecoder)

    struct = phonondb_doc.structure
    supercell = phonondb_doc.supercell
    pbe_dos = phonondb_doc.phonon_dos
    pbe_bands = phonondb_doc.phonon_bandstructure
    struct.properties[Key.mat_id] = mat_id
    formula = struct.formula.replace(" ", "")

    for model, mlff_makers in models.items():
        if model == Model.mace_mp:
            torch.set_default_dtype(torch.float64)

        model_key = model.lower().replace(" ", "-")
        os.makedirs(root_dir := f"{RUNS_DIR}/{model_key}", exist_ok=True)

        ml_doc_path = f"{PH_DOCS_DIR}/{mat_id}-{formula}-{model_key}.json.lzma"

        if os.path.isfile(ml_doc_path) and skip_existing:
            # skip if ML doc exists, can easily generate bs_dos_fig from that without
            # rerunning workflow
            print(f"\nSkipping {model!s} for {mat_id}: phonon doc file exists")
            continue
        try:
            start = perf_counter()
            phonon_flow = PhononMaker(
                **mlff_makers,
                store_force_constants=False,
                # use "setyawan_curtarolo" when comparing to MP and "seekpath" else
                # since setyawan_curtarolo only compatible with primitive cell
                kpath_scheme="setyawan_curtarolo" if which_db == "mp" else "seekpath",
                create_thermal_displacements=False,
                # use_symmetrized_structure="primitive",
            ).make(structure=struct, supercell_matrix=supercell)

            # phonon_flow.draw_graph().show()

            result = run_locally(
                phonon_flow, root_dir=root_dir, log=True, ensure_success=True
            )
            print(f"\n{model} took: {perf_counter() - start:.2f} s")

            last_job_id = phonon_flow[-1].uuid
            ml_phonon_doc: Atomate2PhononBSDOSDoc = result[last_job_id][1].output

            with zopen(ml_doc_path, "wt") as file:
                json.dump(ml_phonon_doc, file, cls=MontyEncoder)

            ml_bs, ml_dos = ml_phonon_doc.phonon_bandstructure, ml_phonon_doc.phonon_dos
            if "pbe_dos" in locals() and "pbe_bands" in locals():
                dos_dict = {Key.pbe.label: pbe_dos, model.label: ml_dos}
                bands_dict = {Key.pbe.label: pbe_bands, model.label: ml_bs}
            else:
                bands_dict = {model.label: ml_bs}
                dos_dict = {model.label: ml_dos}

            fig_bs_dos = plot_phonon_bands_and_dos(bands_dict, dos_dict)

            fig_bs_dos.layout.title = dict(
                text=plotly_title(formula, mat_id), x=0.5, y=0.97
            )
            fig_bs_dos.layout.margin = dict(t=40, b=0, l=5, r=5)
            fig_bs_dos.layout.legend.update(x=1, y=1.07, xanchor="right")
            fig_bs_dos.show()

            img_name = f"{mat_id}-bs-dos-{Key.pbe}-vs-{model_key}"
            save_fig(fig_bs_dos, f"{FIGS_DIR}/{img_name}.pdf")
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
            errors += [(mat_id, model, formula)]

        # MACE annoyingly changes the torch default dtype which breaks CHGNet
        # and M3GNet, so we reset it here
        torch.set_default_dtype(torch.float32)

if errors:
    print(f"\n{errors=}")
