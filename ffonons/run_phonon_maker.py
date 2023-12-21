# %%
import json
import os
import re
import shutil
from glob import glob
from time import perf_counter

import atomate2.forcefields.jobs as ff_jobs
from atomate2.common.schemas.phonons import PhononBSDOSDoc as Atomate2PhononBSDOSDoc
from atomate2.forcefields.flows.phonons import PhononMaker
from jobflow import run_locally
from monty.io import zopen
from mp_api.client import MPRester
from pymatgen.phonon.plotter import PhononBSPlotter
from pymatgen.util.string import latexify
from pymatviz.io import save_fig
from tqdm import tqdm

from ffonons import (
    DATA_DIR,
    FIGS_DIR,
    ROOT,
    WhichDB,
    bs_key,
    dft_key,
    dos_key,
    pretty_label_map,
)
from ffonons.dbs.phonondb import parse_phonondb_docs
from ffonons.plots import plot_phonon_bs, plot_phonon_dos_mpl

__author__ = "Janosh Riebesell"
__date__ = "2023-11-19"


# %%
runs_dir = f"{ROOT}/tmp/runs"  # noqa: S108
which_db: WhichDB = "phonon_db"
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
    # "chgnet-v0.3.0": dict(
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


# %% Main loop over materials and models
# for mat_id in mp_ids: # MP
# for mat_id, struct in get_gnome_pmg_structures().items():  # GNOME
errors: dict[str, Exception] = {}

for zip_path in tqdm(glob(f"{DATA_DIR}/{which_db}/mp-*-pbe.zip")):  # PhononDB
    mat_id = "-".join(zip_path.split("/")[-1].split("-")[:2])
    assert re.match(r"mp-\d+", mat_id), f"Invalid {mat_id=}"

    phonon_db_results = parse_phonondb_docs(zip_path, is_nac=False)

    struct, supercell_matrix, pbe_dos, pbe_bands = (
        phonon_db_results[key]
        for key in "unit_cell supercell_matrix phonon_dos phonon_bandstructure".split()
    )
    struct.properties["id"] = mat_id

    formula = struct.formula.replace(" ", "")

    id_formula = f"{mat_id}-{formula}"
    for model, makers in models.items():
        model_key = model.lower().replace(" ", "-")
        os.makedirs(root_dir := f"{runs_dir}/{model_key}", exist_ok=True)

        dos_fig_path = f"{figs_out_dir}/{mat_id}-dos-pbe-vs-{model_key}.pdf"
        bands_fig_path = f"{figs_out_dir}/{mat_id}-bands-pbe-vs-{model_key}.pdf"
        ml_doc_path = f"{ph_docs_dir}/{id_formula}-{model_key}.json.lzma"

        # skip workflow run if all output files already exist
        if all(map(os.path.isfile, (dos_fig_path, bands_fig_path, ml_doc_path))):
            print(f"Skipping {model!r} for {id_formula}")
            continue
        try:
            start = perf_counter()
            phonon_flow = PhononMaker(
                **makers,
                store_force_constants=False,
                # use "setyawan_curtarolo" when comparing to MP and "seekpath" else
                # note: setyawan_curtarolo only compatible with primitive cell
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

            ml_doc_dict = {
                dos_key: ml_phonon_doc.phonon_dos.as_dict(),
                bs_key: ml_phonon_doc.phonon_bandstructure.as_dict(),
            }

            with zopen(ml_doc_path, "wt") as file:
                file.write(json.dumps(ml_doc_dict))

            pbe_label, ml_label = pretty_label_map[dft_key], pretty_label_map[model]
            dos_dict = {ml_label: ml_phonon_doc.phonon_dos}
            if "pbe_dos" in locals():
                dos_dict[pbe_label] = pbe_dos
            ax_dos = plot_phonon_dos_mpl(dos_dict, struct=struct)
            save_fig(ax_dos, dos_fig_path)

            ml_phonon_bands = ml_phonon_doc.phonon_bandstructure
            if which_db in ("gnome",):
                ax_bs = plot_phonon_bs(ml_phonon_bands, f"{model} - ", struct)
                save_fig(ax_bs, bands_fig_path)

            pbe_bs_plotter = PhononBSPlotter(pbe_bands, label=pbe_label)
            ml_bs_plotter = PhononBSPlotter(ml_phonon_bands, label=ml_label)

            ax_bands = pbe_bs_plotter.plot_compare(ml_bs_plotter, linewidth=2)
            ax_bands.set_title(f"{latexify(formula)} {mat_id}", fontsize=24)
            ax_bands.figure.subplots_adjust(top=0.95)  # make room for title

            save_fig(ax_bands, bands_fig_path)
        except ValueError as exc:
            # known possible errors:
            # - the 2 band structures are not compatible, due to symmetry change during
            # MACE relaxation, try different PhononMaker symprec (default=1e-4). compare
            # PBE and MACE space groups to verify cause
            # - phonopy found imaginary dispersion > 1e-10 (fixed by disabling thermal
            # displacement matrices)
            ml_spg = ml_phonon_doc.structure.get_space_group_info()[1]
            pbe_spg = struct.get_space_group_info()[1]
            errors[id_formula] = f"{model}: {exc} {ml_spg=} {pbe_spg=}"

if errors:
    print(f"{errors=}")
