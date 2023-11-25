# %%
import gzip
import json
import os
import re
import shutil
from collections import defaultdict
from glob import glob
from time import perf_counter

import atomate2.forcefields.jobs as ff_jobs
from atomate2.common.schemas.phonons import PhononBSDOSDoc as AtomPhononBSDOSDoc
from atomate2.forcefields.flows.phonons import PhononMaker
from emmet.core.phonon import PhononBSDOSDoc
from jobflow import run_locally
from matbench_discovery import ROOT as MBD_ROOT
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatviz import plot_band_structure, plot_dos
from pymatviz.io import save_fig

from ffonons import FIGS_DIR, ROOT
from ffonons.plots import plot_phonon_bs, plot_phonon_dos

bs_key = "phonon_bandstructure"
dos_key = "phonon_dos"
id_key = "material_id"
formula_key = "formula"

# %%
runs_dir = f"{ROOT}/runs"
shutil.rmtree(runs_dir, ignore_errors=True)  # remove old runs
os.makedirs(runs_dir, exist_ok=True)

# common_relax_kwds = dict(fmax=0.00001, relax_cell=True)
common_relax_kwds = dict(fmax=0.00001)
mace_kwds = dict(
    model=f"{MBD_ROOT}/models/mace/checkpoints/"
    # "2023-11-15-mace-16M-pbenner-mptrj-200-epochs.model",
    "2023-10-29-mace-16M-pbenner-mptrj-no-conditional-loss.model",
    model_kwargs={"device": "cpu", "default_dtype": "float64"},
)
chgnet_kwds = dict(optimizer_kwargs=dict(use_device="mps"))

models = dict(
    MACE=dict(
        bulk_relax_maker=ff_jobs.MACERelaxMaker(
            relax_kwargs=common_relax_kwds,
            **mace_kwds,
        ),
        phonon_displacement_maker=ff_jobs.MACEStaticMaker(**mace_kwds),
        static_energy_maker=ff_jobs.MACEStaticMaker(**mace_kwds),
    ),
    # M3GNet=dict(
    #     bulk_relax_maker=ff_jobs.M3GNetRelaxMaker(relax_kwargs=common_relax_kwds),
    #     phonon_displacement_maker=ff_jobs.M3GNetStaticMaker(),
    #     static_energy_maker=ff_jobs.M3GNetStaticMaker(),
    # ),
    CHGNet=dict(
        bulk_relax_maker=ff_jobs.CHGNetRelaxMaker(
            relax_kwargs=common_relax_kwds, **chgnet_kwds
        ),
        phonon_displacement_maker=ff_jobs.CHGNetStaticMaker(**chgnet_kwds),
        static_energy_maker=ff_jobs.CHGNetStaticMaker(**chgnet_kwds),
    ),
)


# see https://legacy.materialsproject.org/#search/materials/{%22has_phonons%22%3Atrue}
# for all mp-ids with phonons data
mp_ids = (
    "mp-504729 mp-3486 mp-23137 mp-8799 mp-28591 mp-989639 mp-10182 mp-672 "
    "mp-5342 mp-27422 mp-29750 mp-9919 mp-2894 mp-406 mp-567776 mp-8192 mp-697144 "
    "mp-15794 mp-23268 mp-3163 mp-14069 mp-23405 mp-3056 mp-27207 mp-11175 mp-11718 "
    "mp-9566 mp-14092 mp-1960 mp-30459 mp-772290 mp-996943 mp-571093 mp-149 mp-504729 "
    "mp-3486 mp-23137 mp-8799 mp-28591 mp-989639 mp-10182 mp-672 mp-5342 mp-27422 "
    "mp-29750 mp-9919 mp-2894 mp-406 mp-567776 mp-8192 mp-697144 mp-15794 mp-23268 "
    "mp-3163 mp-14069 mp-23405 mp-3056 mp-27207 mp-11175 mp-11718 mp-9566 mp-14092 "
    "mp-1960 mp-30459 mp-772290 mp-996943 mp-571093 mp-149 mp-10418 mp-8235 mp-3666 "
    "mp-1020059 mp-27253 mp-23485 mp-20463 mp-7173 mp-14039 mp-7774 mp-961713 "
    "mp-20072 mp-19795 mp-27385 mp-10163 mp-36508 mp-8161 mp-7684 mp-27193 mp-866291 "
    "mp-14437 mp-10616 mp-867335 mp-7773 mp-28264 mp-638729 mp-9567 mp-22925 "
    "mp-989531 mp-549697 mp-15988 mp-7639 mp-8446 mp-13541 mp-561947 mp-553025 mp-4236 "
    "mp-989526 mp-9200 mp-755013 mp-30056 mp-862871 mp-752676 mp-553374 mp-3359 "
    "mp-2530 mp-3637 mp-29149 mp-13798 mp-2074 mp-11583 mp-8188 mp-9881 mp-753802 "
    "mp-8979 mp-643257 mp-10547 mp-551243 mp-8217 mp-6870 mp-989567 mp-4187 mp-1747 "
    "mp-23023 mp-996962 mp-8932".split()
)


# %%
mpr = MPRester(mute_progress_bars=True)

for mp_id in mp_ids:
    struct: Structure = mpr.get_structure_by_material_id(mp_id)
    struct.properties["id"] = mp_id

    out_dir = f"{FIGS_DIR}/{mp_id}-{struct.formula.replace(' ', '')}"
    mp_dos_fig_path = f"{out_dir}/dos-mp.pdf"
    mp_bands_fig_path = f"{out_dir}/bands-mp.pdf"
    # struct.to(filename=f"{out_dir}/struct.cif")

    if not os.path.isfile(mp_dos_fig_path) or not os.path.isfile(mp_bands_fig_path):
        mp_phonon_doc: PhononBSDOSDoc = mpr.materials.phonon.get_data_by_id(mp_id)

        mp_doc_dict = {
            dos_key: mp_phonon_doc.ph_dos.as_dict(),
            bs_key: mp_phonon_doc.ph_bs.as_dict(),
        }
        os.makedirs(out_dir, exist_ok=True)  # create dir only if MP has phonon data
        with gzip.open(f"{out_dir}/phonon-bs-dos-mp.json.gz", "wt") as file:
            file.write(json.dumps(mp_doc_dict))

        ax_bs_mp = plot_phonon_bs(mp_phonon_doc.ph_bs, "MP", struct)
        # always save figure right away, plotting another figure will clear the axis!
        save_fig(ax_bs_mp, mp_bands_fig_path)

        ax_dos_mp = plot_phonon_dos(mp_phonon_doc.ph_dos, "MP", struct)

        save_fig(ax_dos_mp, mp_dos_fig_path)

    for model, makers in models.items():
        try:
            dos_fig_path = f"{out_dir}/dos-{model.lower()}.pdf"
            bands_fig_path = f"{out_dir}/bands-{model.lower()}.pdf"
            ml_doc_path = f"{out_dir}/phonon-bs-dos-{model.lower()}.json.gz"
            if all(map(os.path.isfile, (dos_fig_path, bands_fig_path, ml_doc_path))):
                print(f"Skipping {model} for {struct.formula}")
                continue

            start = perf_counter()
            phonon_flow = PhononMaker(
                **makers, min_length=15, store_force_constants=False
            ).make(structure=struct)

            # phonon_flow.draw_graph().show()

            os.makedirs(root_dir := f"{runs_dir}/{model.lower()}", exist_ok=True)
            responses = run_locally(
                phonon_flow,
                root_dir=root_dir,
                log=False,
                ensure_success=True,
            )
            print(f"{model} took: {perf_counter() - start:.2f} s")

            ml_phonon_doc: AtomPhononBSDOSDoc = next(
                val
                for val in responses.values()
                if isinstance(val[1].output, AtomPhononBSDOSDoc)
            )[1].output
            labels = ml_phonon_doc.phonon_bandstructure.labels_dict
            labels["$\\Gamma$"] = labels.pop("GAMMA")
            ml_doc_dict = {
                dos_key: ml_phonon_doc.phonon_dos.as_dict(),
                bs_key: ml_phonon_doc.phonon_bandstructure.as_dict(),
            }

            with gzip.open(ml_doc_path, "wt") as file:
                file.write(json.dumps(ml_doc_dict))

            ax_dos = plot_phonon_dos(ml_phonon_doc.phonon_dos, model, struct)
            save_fig(ax_dos, dos_fig_path)

            phonon_bands = ml_phonon_doc.phonon_bandstructure
            ax_bs = plot_phonon_bs(phonon_bands, model, struct)
            save_fig(ax_bs, bands_fig_path)
        except (RuntimeError, ValueError) as exc:
            print(f"!!! {model} failed: {exc}")


# %% load all docs
all_docs = defaultdict(dict)
for phonon_doc in glob(f"{FIGS_DIR}/**/phonon-bs-dos-*.json.gz"):
    with gzip.open(phonon_doc, "rt") as file:
        doc_dict = json.load(file)

    doc_dict[dos_key] = PhononDos.from_dict(doc_dict[dos_key])
    doc_dict[bs_key] = PhononBandStructureSymmLine.from_dict(doc_dict[bs_key])

    mp_id, formula, model = re.search(
        r".*/(mp-.*)-(.*)/phonon-bs-dos-(.*).json.gz", phonon_doc
    ).groups()
    assert mp_id.startswith("mp-"), f"Invalid {mp_id=}"

    all_docs[mp_id][model] = doc_dict | {formula_key: formula, id_key: mp_id}


n_preds = sum(len(v) for v in all_docs.values())
print(f"got {len(all_docs)} materials and {n_preds} predictions")

materials_with_2 = [key for key, val in all_docs.items() if len(val) == 2]
print(f"{len(materials_with_2)=}")
materials_with_3 = [key for key, val in all_docs.items() if len(val) == 3]
print(f"{len(materials_with_3)=}")


# %%
for mp_id in materials_with_3:
    try:
        mace, chgnet = all_docs[mp_id]["mace"], all_docs[mp_id]["chgnet"]
        formula = mace[formula_key]

        band_structs = {"CHGnet": chgnet[bs_key], "MACE": mace[bs_key]}
        fig = plot_band_structure(band_structs)
        formula = chgnet[formula_key]
        # turn title into link to materials project page
        href = f"https://legacy.materialsproject.org/materials/{mp_id}"
        title = f"<a {href=}>{mp_id}</a> {formula}"
        fig.layout.title = dict(text=title, x=0.5)
        fig.show()

        # now the same for DOS
        doses = {
            "CHGnet": chgnet[dos_key],
            "MACE": mace[dos_key],
            "MP": all_docs[mp_id]["mp"][dos_key],
        }
        ax = plot_phonon_dos(doses)
        ax.set_title(f"{mp_id} {formula}", fontsize=22, fontweight="bold")
        save_fig(ax, f"{FIGS_DIR}/{mp_id}-{formula.replace(' ', '')}/dos-all.pdf")

        fig_dos = plot_dos(doses)
        fig_dos.layout.title = f"{mp_id} {formula}"
        fig_dos.show()

    except ValueError as exc:
        print(f"{mp_id} failed: {exc}")
