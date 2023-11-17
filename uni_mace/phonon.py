# %%
import os
import shutil
from time import perf_counter

import atomate2.forcefields.jobs as ff_jobs
from atomate2.forcefields.flows.phonons import PhononMaker
from jobflow import run_locally
from matbench_discovery import ROOT as MBD_ROOT
from pymatgen.ext.matproj import MPRester
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
    # CHGNet=dict(
    #     bulk_relax_maker=ff_jobs.CHGNetRelaxMaker(relax_kwargs=common_relax_kwds),
    #     phonon_displacement_maker=ff_jobs.CHGNetStaticMaker(),
    #     static_energy_maker=ff_jobs.CHGNetStaticMaker(),
    # ),
)

mp_ids = "mp-149 mp-169 mp-4511".split()
structs = list(map(MPRester().get_structure_by_material_id, mp_ids))
for idx, struct in enumerate(structs):
    struct.properties["id"] = mp_ids[idx]


# %%
for struct in structs:
    # try to urlretrieve the image at
    # https://legacy.materialsproject.org/phonons/{mp-id}/phonon_disp
    # and save to save folder as {mp-id}.png
    mat_id = struct.properties["id"]
    out_dir = f"{FIGS_DIR}/phonon/{mat_id}-{struct.formula}"
    mp_dos_fig_path = f"{out_dir}/dos-mp.pdf"
    mp_bands_fig_path = f"{out_dir}/bands-mp.pdf"
    if not os.path.isfile(mp_dos_fig_path) or not os.path.isfile(mp_bands_fig_path):
        try:
            mp_phonon = MPRester().materials.phonon.get_data_by_id(
                mat_id, fields=["ph_bs", "ph_dos"]
            )
            mp_phonon_dos = mp_phonon.ph_dos
            mp_phonon_bs = mp_phonon.ph_bs
            ax_bs_mp = plot_phonon_bs(struct, mp_phonon_bs, "MP")
            save_fig(ax_bs_mp, mp_bands_fig_path)
            ax_dos_mp = plot_phonon_dos(struct, mp_phonon_dos, "MP")
            save_fig(ax_dos_mp, mp_dos_fig_path)

        except Exception as exc:
            print(f"MP doesn't have phonon data for {mat_id!r}:\n{exc}")
            continue

    for model, makers in models.items():
        try:
            dos_fig_path = f"{out_dir}/dos-{model.lower()}.pdf"
            bands_fig_path = f"{out_dir}/bands-{model.lower()}.pdf"
            if os.path.isfile(dos_fig_path) and os.path.isfile(bands_fig_path):
                print(f"Skipping {model} for {struct.formula}")
                continue
            os.makedirs(out_dir, exist_ok=True)

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

            phonon_bs_dos_doc = next(reversed(responses.values()))[1].output
            phonon_dos = phonon_bs_dos_doc.phonon_dos
            ax_dos = plot_phonon_dos(struct, phonon_dos, model)
            ax_dos.figure.subplots_adjust(top=0.95)

            save_fig(ax_dos, dos_fig_path)

            phonon_bands = phonon_bs_dos_doc.phonon_bandstructure
            ax_bs = plot_phonon_bs(struct, phonon_bands, model)
            save_fig(ax_bs, bands_fig_path)

        except (RuntimeError, ValueError) as exc:
            print(f"!!! {model} failed: {exc}")


# %% remove all runs
shutil.rmtree(runs_dir)


# %% plot all DOS into one figure
