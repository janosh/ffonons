# %%
import os
import shutil
from time import perf_counter

import atomate2.forcefields.jobs as ff_jobs
from atomate2.common.schemas.phonons import PhononBSDOSDoc as AtomPhononBSDOSDoc
from atomate2.forcefields.flows.phonons import PhononMaker
from emmet.core.phonon import PhononBSDOSDoc
from jobflow import run_locally
from matbench_discovery import ROOT as MBD_ROOT
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatviz.io import save_fig

from uni_mace import FIGS_DIR, ROOT
from uni_mace.plots import plot_phonon_bs, plot_phonon_dos

# %%
os.makedirs(runs_dir := f"{ROOT}/runs", exist_ok=True)

# common_relax_kwds = dict(fmax=0.00001, relax_cell=True)
common_relax_kwds = dict(fmax=0.00001)
maker_kwds = dict(
    MACE=dict(
        model_path=f"{MBD_ROOT}/models/mace/checkpoints/"
        # "2023-11-15-mace-16M-pbenner-mptrj-200-epochs.model",
        "2023-10-29-mace-16M-pbenner-mptrj-no-conditional-loss.model",
        model_kwargs={"device": "cpu", "default_dtype": "float64"},
    ),
)

models = dict(
    MACE=dict(
        bulk_relax_maker=ff_jobs.MACERelaxMaker(
            relax_kwargs=common_relax_kwds,
            **maker_kwds["MACE"],
        ),
        phonon_displacement_maker=ff_jobs.MACEStaticMaker(**maker_kwds["MACE"]),
        static_energy_maker=ff_jobs.MACEStaticMaker(**maker_kwds["MACE"]),
    ),
    # M3GNet=dict(
    #     bulk_relax_maker=ff_jobs.M3GNetRelaxMaker(relax_kwargs=common_relax_kwds),
    #     phonon_displacement_maker=ff_jobs.M3GNetStaticMaker(),
    #     static_energy_maker=ff_jobs.M3GNetStaticMaker(),
    # ),
    CHGNet=dict(
        bulk_relax_maker=ff_jobs.CHGNetRelaxMaker(relax_kwargs=common_relax_kwds),
        phonon_displacement_maker=ff_jobs.CHGNetStaticMaker(),
        static_energy_maker=ff_jobs.CHGNetStaticMaker(),
    ),
)


# see https://legacy.materialsproject.org/#search/materials/{%22has_phonons%22%3Atrue}
# for all mp-ids with phonons data
mp_ids = (
    "mp-504729 mp-3486 mp-23137 mp-8799 mp-28591 mp-989639 mp-10182 mp-672 "
    "mp-5342 mp-27422 mp-29750 mp-9919 mp-2894 mp-406 mp-567776 mp-8192 mp-697144 "
    "mp-15794 mp-23268 mp-3163 mp-14069 mp-23405 mp-3056 mp-27207 mp-11175 mp-11718 "
    "mp-9566 mp-14092 mp-1960 mp-30459 mp-772290 mp-996943 mp-571093 mp-149".split()
)


# %%
mpr = MPRester(mute_progress_bars=True)

for mp_id in mp_ids:
    struct: Structure = mpr.get_structure_by_material_id(mp_id)
    struct.properties["id"] = mp_id

    out_dir = f"{FIGS_DIR}/phonon/{mp_id}-{struct.formula.replace(' ', '')}"
    mp_dos_fig_path = f"{out_dir}/dos-mp.pdf"
    mp_bands_fig_path = f"{out_dir}/bands-mp.pdf"
    # struct.to(filename=f"{out_dir}/struct.cif")

    if not os.path.isfile(mp_dos_fig_path) or not os.path.isfile(mp_bands_fig_path):
        mp_phonon_doc: PhononBSDOSDoc = mpr.materials.phonon.get_data_by_id(mp_id)
        os.makedirs(out_dir, exist_ok=True)  # create dir only if MP has phonon data

        ax_bs_mp = plot_phonon_bs(mp_phonon_doc.ph_bs, struct, "MP")
        # always save figure right away, plotting another figure will clear the axis!
        save_fig(ax_bs_mp, mp_bands_fig_path)

        ax_dos_mp = plot_phonon_dos(mp_phonon_doc.ph_dos, struct, "MP")

        save_fig(ax_dos_mp, mp_dos_fig_path)

    for model, makers in models.items():
        try:
            dos_fig_path = f"{out_dir}/dos-{model.lower()}.pdf"
            bands_fig_path = f"{out_dir}/bands-{model.lower()}.pdf"
            if os.path.isfile(dos_fig_path) and os.path.isfile(bands_fig_path):
                print(f"Skipping {model} for {struct.formula}")
                continue

            start = perf_counter()
            phonon_flow = PhononMaker(
                **makers, min_length=15, store_force_constants=False
            ).make(structure=struct)

            for job in phonon_flow:
                job.metadata["model"] = model

            # phonon_flow.draw_graph().show()

            os.makedirs(root_dir := f"{runs_dir}/{model.lower()}", exist_ok=True)
            responses = run_locally(
                phonon_flow,
                create_folders=True,
                root_dir=root_dir,
                log=True,
                ensure_success=True,
            )
            print(f"{model} took: {perf_counter() - start:.2f} s")

            ml_phonon_doc: AtomPhononBSDOSDoc = next(
                val
                for val in responses.values()
                if isinstance(val[1].output, AtomPhononBSDOSDoc)
            )[1].output

            ax_dos = plot_phonon_dos(ml_phonon_doc.phonon_dos, struct, model)
            save_fig(ax_dos, dos_fig_path)

            phonon_bands = ml_phonon_doc.phonon_bandstructure
            ax_bs = plot_phonon_bs(phonon_bands, struct, model)
            save_fig(ax_bs, bands_fig_path)

        except (RuntimeError, ValueError) as exc:
            print(f"!!! {model} failed: {exc}")


# %% remove all runs
shutil.rmtree(runs_dir)
