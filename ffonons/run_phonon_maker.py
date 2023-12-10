# %%
import gzip
import json
import os
import re
import shutil
from glob import glob
from time import perf_counter
from typing import Literal

import atomate2.forcefields.jobs as ff_jobs
from atomate2.common.schemas.phonons import PhononBSDOSDoc as Atomate2PhononBSDOSDoc
from atomate2.forcefields.flows.phonons import PhononMaker
from jobflow import run_locally
from mp_api.client import MPRester
from pymatgen.phonon.plotter import PhononBSPlotter
from pymatviz.io import save_fig
from tqdm import tqdm

from ffonons import (
    DATA_DIR,
    FIGS_DIR,
    ROOT,
    bs_key,
    dos_key,
)
from ffonons.io import parse_phonondb_docs
from ffonons.plots import plot_phonon_bs, plot_phonon_dos

__author__ = "Janosh Riebesell"
__date__ = "2023-11-19"


# %%
runs_dir = f"{ROOT}/runs"
which_db: Literal["mp", "phonon_db"] = "phonon_db"
ph_docs_dir = f"{DATA_DIR}/{which_db}"
figs_out_dir = f"{FIGS_DIR}/{which_db}"
shutil.rmtree(runs_dir, ignore_errors=True)  # remove old runs to save space
os.makedirs(runs_dir, exist_ok=True)

common_relax_kwds = dict(fmax=0.00001)
mace_kwds = dict(
    model="https://tinyurl.com/y7uhwpje",
    model_kwargs={"device": "cpu", "default_dtype": "float64"},
)
chgnet_kwds = dict(optimizer_kwargs=dict(use_device="mps"))

models = {
    "MACE y7uhwpje": dict(
        bulk_relax_maker=ff_jobs.MACERelaxMaker(
            relax_kwargs=common_relax_kwds,
            **mace_kwds,
        ),
        phonon_displacement_maker=ff_jobs.MACEStaticMaker(**mace_kwds),
        static_energy_maker=ff_jobs.MACEStaticMaker(**mace_kwds),
    ),
    # "M3GNet":dict(
    #     bulk_relax_maker=ff_jobs.M3GNetRelaxMaker(relax_kwargs=common_relax_kwds),
    #     phonon_displacement_maker=ff_jobs.M3GNetStaticMaker(),
    #     static_energy_maker=ff_jobs.M3GNetStaticMaker(),
    # ),
    # "CHGNet v0.3.0": dict(
    #     bulk_relax_maker=ff_jobs.CHGNetRelaxMaker(
    #         relax_kwargs=common_relax_kwds, **chgnet_kwds
    #     ),
    #     phonon_displacement_maker=ff_jobs.CHGNetStaticMaker(**chgnet_kwds),
    #     static_energy_maker=ff_jobs.CHGNetStaticMaker(**chgnet_kwds),
    # ),
}


with open(f"{DATA_DIR}/mp-ids-with-pbesol-phonons.yml") as file:
    mp_ids = file.read().splitlines()[2:]
bad_ids = [mp_id for mp_id in mp_ids if not mp_id.startswith("mp-")]
assert len(bad_ids) == 0, f"{bad_ids=}"


# %%
mpr = MPRester(mute_progress_bars=True)


# %%
# for mp_id in mp_ids:
for yaml_filepath in tqdm(glob(f"{DATA_DIR}/phonon_db/mp-*-*.zip")):
    mp_id = "-".join(yaml_filepath.split("/")[-1].split("-")[:2])
    assert re.match(r"mp-\d+", mp_id), f"Invalid {mp_id=}"

    phonon_db_results = parse_phonondb_docs(
        yaml_filepath, out_dir=f"{FIGS_DIR}/phonon_db/{mp_id}"
    )

    struct, supercell_matrix, pbe_dos, pbe_bands = (
        phonon_db_results[key]
        for key in "unit_cell supercell_matrix phonon_dos phonon_bandstructure".split()
    )

    formula = struct.formula.replace(" ", "")
    struct.properties["id"] = mp_id
    id_formula = f"{mp_id}-{formula}"

    togo_doc_dict = {
        dos_key: phonon_db_results["phonon_dos"].as_dict(),
        bs_key: phonon_db_results["phonon_bandstructure"].as_dict(),
    }
    togo_doc_path = f"{ph_docs_dir}/{mp_id}-{formula}-pbe.json.gz"

    with gzip.open(togo_doc_path, "wt") as file:
        file.write(json.dumps(togo_doc_dict))

    for model, makers in models.items():
        model_key = model.lower().replace(" ", "-")
        os.makedirs(root_dir := f"{runs_dir}/{model_key}", exist_ok=True)

        dos_fig_path = f"{figs_out_dir}/{mp_id}-dos-pbe-vs-{model_key}.pdf"
        bands_fig_path = f"{figs_out_dir}/{mp_id}-bands-pbe-vs-{model_key}.pdf"
        ml_doc_path = f"{ph_docs_dir}/{id_formula}-{model_key}.json.gz"

        # skip workflow run if all output files already exist
        if all(map(os.path.isfile, (dos_fig_path, bands_fig_path, ml_doc_path))):
            print(f"Skipping {model!r} for {id_formula}")
            continue

        start = perf_counter()
        phonon_flow = PhononMaker(
            **makers,
            # use_symmetrized_structure="primitive",
            store_force_constants=False,
            # use kpath_scheme="setyawan_curtarolo" when comparing to MP
            # and 'seekpath' when comparing with PhonoDB!
            # note: setyawan_curtarolo only compatible with primitive cell
            # kpath_scheme="setyawan_curtarolo",
            kpath_scheme="seekpath",
        ).make(structure=struct, supercell_matrix=supercell_matrix)

        # phonon_flow.draw_graph().show()

        result = run_locally(
            phonon_flow, root_dir=root_dir, log=False, ensure_success=True
        )
        print(f"{model} took: {perf_counter() - start:.2f} s")

        last_job_id = phonon_flow[-1].uuid
        ml_phonon_doc: Atomate2PhononBSDOSDoc = result[last_job_id][1].output

        ml_doc_dict = {
            dos_key: ml_phonon_doc.phonon_dos.as_dict(),
            bs_key: ml_phonon_doc.phonon_bandstructure.as_dict(),
        }

        with gzip.open(ml_doc_path, "wt") as file:
            file.write(json.dumps(ml_doc_dict))

        ax_dos_compare = plot_phonon_dos(
            {model: ml_phonon_doc.phonon_dos, "PBE": pbe_dos},
            struct=struct,
        )
        save_fig(ax_dos_compare, dos_fig_path)

        ml_phonon_bands = ml_phonon_doc.phonon_bandstructure
        if which_db == "mp":  # TODO maybe remove, do we need single model bands plot?
            ax_bs = plot_phonon_bs(ml_phonon_bands, f"{model} - ", struct)
            save_fig(ax_bs, bands_fig_path)

        pbe_bs_plotter = PhononBSPlotter(pbe_bands, label="PBE")
        ml_bs_plotter = PhononBSPlotter(ml_phonon_bands, label=model)
        ax_bands_compare = pbe_bs_plotter.plot_compare(ml_bs_plotter, linewidth=2)
        ax_bands_compare.set_title(f"{formula} {mp_id}", fontsize=24)
        ax_bands_compare.figure.subplots_adjust(top=0.95)  # make room for title
        save_fig(ax_bands_compare, bands_fig_path)
