# %%
import os
import shutil
from time import perf_counter

import pandas as pd
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import (
    CHGNetRelaxMaker,
    CHGNetStaticMaker,
    M3GNetRelaxMaker,
    M3GNetStaticMaker,
    MACERelaxMaker,
    MACEStaticMaker,
)
from jobflow import SETTINGS, run_locally
from matbench_discovery import ROOT as MBD_ROOT
from pymatgen.ext.matproj import MPRester
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatviz.io import save_fig

from uni_mace import FIGS_DIR, ROOT

# %%
store = SETTINGS.JOB_STORE
store.connect()
assert store.count() == 0
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

makers = dict(
    MACE=dict(
        bulk_relax_maker=MACERelaxMaker(
            relax_kwargs=common_relax_kwds,
            **maker_kwds["MACE"],
        ),
        phonon_displacement_maker=MACEStaticMaker(**maker_kwds["MACE"]),
        static_energy_maker=MACEStaticMaker(**maker_kwds["MACE"]),
    ),
    M3GNet=dict(
        bulk_relax_maker=M3GNetRelaxMaker(relax_kwargs=common_relax_kwds),
        phonon_displacement_maker=M3GNetStaticMaker(),
        static_energy_maker=M3GNetStaticMaker(),
    ),
    CHGNet=dict(
        bulk_relax_maker=CHGNetRelaxMaker(relax_kwargs=common_relax_kwds),
        phonon_displacement_maker=CHGNetStaticMaker(),
        static_energy_maker=CHGNetStaticMaker(),
    ),
)

mp_ids = "mp-149 mp-169".split()
structs = list(map(MPRester().get_structure_by_material_id, mp_ids))
for idx, struct in enumerate(structs):
    struct.properties["id"] = mp_ids[idx]


# %%
for struct in structs:
    for model in models:
        try:
            mat_id = struct.properties["id"]
            out_dir = f"{FIGS_DIR}/phonon/{mat_id}-{struct.formula}"
            os.makedirs(out_dir, exist_ok=True)

            start = perf_counter()
            phonon_flow = PhononMaker(
                **makers[model], min_length=15, store_force_constants=False
            ).make(structure=struct)

            for job in phonon_flow:
                job.metadata["model"] = model

            # phonon_flow.draw_graph().show()

            os.makedirs(root_dir := f"{runs_dir}/{model.lower()}", exist_ok=True)
            responses = run_locally(
                phonon_flow,
                create_folders=True,
                store=SETTINGS.JOB_STORE,
                root_dir=root_dir,
                log=False,
                ensure_success=True,
            )
            print(f"{model} took: {perf_counter() - start:.2f} s")

            result = store.query_one(
                {"name": "generate_frequencies_eigenvectors", "metadata.model": model},
                # {"name": "generate_frequencies_eigenvectors", "output.forcefield_name": model},
                properties=[
                    "output.phonon_dos",
                    "output.phonon_bandstructure",
                    "metadata.model",
                ],
                load=True,
                sort={"completed_at": -1},  # to get the latest computation
            )
            if result is None:
                raise ValueError(f"No results for {model=}")

            phonon_bs = PhononBandStructureSymmLine.from_dict(
                result["output"]["phonon_bandstructure"],
            )
            phonon_dos = PhononDos.from_dict(result["output"]["phonon_dos"])

            dos_plot = PhononDosPlotter()
            dos_plot.add_dos(label=model, dos=phonon_dos)
            ax_dos = dos_plot.get_plot()
            title_prefix = f"{model} {struct.formula} ({mat_id}) phonon"
            ax_dos.set_title(f"{title_prefix} DOS", fontsize=22)

            save_fig(ax_dos, f"{out_dir}/dos-{model.lower()}.pdf")

            bs_plot = PhononBSPlotter(bs=phonon_bs)
            ax_bs = bs_plot.get_plot()
            ax_bs.set_title(
                f"{title_prefix} band structure", fontsize=22, fontweight="bold"
            )
            # increase top margin to make room for title
            ax_bs.figure.subplots_adjust(top=0.95)
            save_fig(ax_bs, f"{out_dir}/bands-{model.lower()}.pdf")

        except (RuntimeError, ValueError) as exc:
            print(f"!!! {model} failed: {exc}")


# %% remove all runs
shutil.rmtree(runs_dir)
