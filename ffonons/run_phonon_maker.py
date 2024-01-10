# %%
import json
import os
import re
import shutil
from glob import glob
from time import perf_counter
from zipfile import BadZipFile

import atomate2.forcefields.jobs as ff_jobs
from atomate2.common.schemas.phonons import PhononBSDOSDoc as Atomate2PhononBSDOSDoc
from atomate2.forcefields.flows.phonons import PhononMaker
from jobflow import run_locally
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
from mp_api.client import MPRester
from pymatviz import plot_phonon_bands_and_dos
from pymatviz.io import save_fig
from tqdm import tqdm

from ffonons import DATA_DIR, FIGS_DIR, ROOT, WhichDB, dft_key
from ffonons.dbs.phonondb import PhononDBDocParsed
from ffonons.plots import plotly_title, pretty_label_map

__author__ = "Janosh Riebesell"
__date__ = "2023-11-19"


# %%
runs_dir = f"{ROOT}/tmp/runs"  # noqa: S108
which_db: WhichDB = "phonon-db"
ph_docs_dir = f"{DATA_DIR}/{which_db}"
figs_out_dir = f"{FIGS_DIR}/{which_db}"
shutil.rmtree(runs_dir, ignore_errors=True)  # remove old runs to save space
for directory in (ph_docs_dir, figs_out_dir, runs_dir):
    os.makedirs(directory, exist_ok=True)

common_relax_kwds = dict(fmax=0.00001)
mace_kwds = dict(
    model="https://tinyurl.com/y7uhwpje",
    model_kwargs={"device": "cpu", "default_dtype": "float64"},
)
chgnet_kwds = dict(optimizer_kwargs=dict(use_device="mps"))

models = {
    "mace-y7uhwpje": dict(
        bulk_relax_maker=ff_jobs.MACERelaxMaker(
            relax_kwargs=common_relax_kwds,
            **mace_kwds,
        ),
        phonon_displacement_maker=ff_jobs.MACEStaticMaker(**mace_kwds),
        static_energy_maker=ff_jobs.MACEStaticMaker(**mace_kwds),
    ),
    # "m3gnet":dict(
    #     bulk_relax_maker=ff_jobs.M3GNetRelaxMaker(relax_kwargs=common_relax_kwds),
    #     phonon_displacement_maker=ff_jobs.M3GNetStaticMaker(),
    #     static_energy_maker=ff_jobs.M3GNetStaticMaker(),
    # ),
    "chgnet-v0.3.0": dict(
        bulk_relax_maker=ff_jobs.CHGNetRelaxMaker(
            relax_kwargs=common_relax_kwds, **chgnet_kwds
        ),
        phonon_displacement_maker=ff_jobs.CHGNetStaticMaker(**chgnet_kwds),
        static_energy_maker=ff_jobs.CHGNetStaticMaker(**chgnet_kwds),
    ),
}


with open(f"{DATA_DIR}/mp-ids-with-pbesol-phonons.yml") as file:
    mp_ids = file.read().splitlines()[2:]
bad_ids = [mp_id for mp_id in mp_ids if not mp_id.startswith("mp-")]
assert len(bad_ids) == 0, f"{bad_ids=}"


# %%
mpr = MPRester(mute_progress_bars=True)


# %% Main loop over materials and models
# for mat_id in mp_ids: # MP
# for mat_id, struct in get_gnome_pmg_structures().items():  # GNOME
errors: dict[str, Exception] = {}

for dft_doc_path in (
    pbar := tqdm(glob(f"{DATA_DIR}/{which_db}/mp-*-pbe.json.lzma"))
):  # PhononDB
    mat_id = "-".join(dft_doc_path.split("/")[-1].split("-")[:2])
    pbar.set_description(f"{mat_id=}")
    assert re.match(r"mp-\d+", mat_id), f"Invalid {mat_id=}"

    with zopen(dft_doc_path, "rt") as file:
        phonondb_doc: PhononDBDocParsed = json.load(file, cls=MontyDecoder)

    struct, supercell_matrix, pbe_dos, pbe_bands = (
        getattr(phonondb_doc, key)
        for key in "structure supercell_matrix phonon_dos phonon_bandstructure".split()
    )
    struct.properties["id"] = mat_id
    formula = struct.formula.replace(" ", "")
    id_formula = f"{mat_id}-{formula}"

    for model, makers in models.items():
        model_key = model.lower().replace(" ", "-")
        os.makedirs(root_dir := f"{runs_dir}/{model_key}", exist_ok=True)

        bs_dos_fig_path = f"{figs_out_dir}/{mat_id}-bs-dos-{dft_key}-vs-{model_key}.pdf"
        ml_doc_path = f"{ph_docs_dir}/{id_formula}-{model_key}.json.lzma"

        if os.path.isfile(ml_doc_path):  # skip if ML doc exists, can easily generate
            # bs_dos_fig_path from that without rerunning workflow
            print(f"Skipping {model!r} for {id_formula}: output files exist")
            continue
        try:
            start = perf_counter()
            phonon_flow = PhononMaker(
                **makers,
                store_force_constants=False,
                # use "setyawan_curtarolo" when comparing to MP and "seekpath" else
                # since setyawan_curtarolo only compatible with primitive cell
                kpath_scheme="setyawan_curtarolo" if which_db == "mp" else "seekpath",
                create_thermal_displacements=False,
                # use_symmetrized_structure="primitive",
            ).make(structure=struct, supercell_matrix=supercell_matrix)

            # phonon_flow.draw_graph().show()

            result = run_locally(
                phonon_flow, root_dir=root_dir, log=False, ensure_success=True
            )
            print(f"{model} took: {perf_counter() - start:.2f} s")

            last_job_id = phonon_flow[-1].uuid
            ml_phonon_doc: Atomate2PhononBSDOSDoc = result[last_job_id][1].output

            with zopen(ml_doc_path, "wt") as file:
                json.dump(ml_phonon_doc, file, cls=MontyEncoder)

            pbe_label, ml_label = pretty_label_map[dft_key], pretty_label_map[model]
            dos_dict = {ml_label: ml_phonon_doc.phonon_dos}
            bands_dict = {ml_label: ml_phonon_doc.phonon_bandstructure}
            if "pbe_dos" in locals() and "pbe_bands" in locals():
                dos_dict[pbe_label] = pbe_dos
                bands_dict[pbe_label] = pbe_bands
            fig_bs_dos = plot_phonon_bands_and_dos(bands_dict, dos_dict)

            fig_bs_dos.layout.title = dict(
                text=plotly_title(formula, mat_id), x=0.5, y=0.97
            )
            fig_bs_dos.layout.margin = dict(t=40, b=0, l=5, r=5)
            fig_bs_dos.layout.legend.update(x=1, y=1.07, xanchor="right")
            fig_bs_dos.show()
            save_fig(fig_bs_dos, bs_dos_fig_path)
        except (ValueError, RuntimeError, BadZipFile) as exc:
            # known possible errors:
            # - the 2 band structures are not compatible, due to symmetry change during
            # MACE relaxation, try different PhononMaker symprec (default=1e-4). compare
            # PBE and MACE space groups to verify cause
            # - phonopy found imaginary dispersion > 1e-10 (fixed by disabling thermal
            # displacement matrices)
            # - phonopy-internal: RuntimeError: Creating primitive cell failed.
            # PRIMITIVE_AXIS may be incorrectly specified. For mp-754196 Ba2Sr1I6
            # faulty downloads of phonondb docs raise "BadZipFile: is not a zip file"
            ml_spg = ml_phonon_doc.structure.get_space_group_info()[1]
            pbe_spg = struct.get_space_group_info()[1]
            errors[id_formula] = f"{model}: {exc} {ml_spg=} {pbe_spg=}"

if errors:
    print(f"{errors=}")
