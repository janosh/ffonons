# %%
import os
import shutil

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
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatviz.io import save_fig

from uni_mace import FIGS_DIR, ROOT

# %%
store = SETTINGS.JOB_STORE
store.connect()
os.makedirs(runs_dir := f"{ROOT}/runs", exist_ok=True)


common_relax_kwds = dict(fmax=0.00001)
maker_kwds = dict(
    MACE=dict(
        potential_param_file_name=f"{MBD_ROOT}/models/mace/checkpoints/"
        # "2023-11-15-mace-16M-pbenner-mptrj-200-epochs.model",
        "2023-10-29-mace-16M-pbenner-mptrj-no-conditional-loss.model",
        potential_kwargs={"device": "cpu"},
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


struct = Structure(
    lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
    species=["Si", "Si"],
    coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
)
# %%
for model in ("MACE", "M3GNet", "CHGNet"):
    phonon_flow = PhononMaker(
        **makers[model], min_length=15, store_force_constants=False
    ).make(structure=struct)

    # phonon_flow.draw_graph().show()

    run_locally(
        phonon_flow,
        create_folders=True,
        store=SETTINGS.JOB_STORE,
        root_dir=f"runs/{model.lower()}",
    )

    result = store.query_one(
        {"name": "generate_frequencies_eigenvectors"},
        properties=[
            "output.phonon_dos",
            "output.phonon_bandstructure",
        ],
        load=True,
        sort={"completed_at": -1},  # to get the latest computation
    )

    phonon_bs = PhononBandStructureSymmLine.from_dict(
        result["output"]["phonon_bandstructure"],
    )
    phonon_dos = PhononDos.from_dict(result["output"]["phonon_dos"])

    dos_plot = PhononDosPlotter()
    dos_plot.add_dos(label="a", dos=phonon_dos)
    ax_dos = dos_plot.get_plot()

    save_fig(ax_dos, f"{FIGS_DIR}/si-phonon-dos-{model}.pdf")

    bs_plot = PhononBSPlotter(bs=phonon_bs)
    ax_bs = bs_plot.get_plot()
    title = f"{model} Si2 mp-149 phonon band structure"
    ax_bs.set_title(title, fontsize=22, fontweight="bold")
    # increase top margin to make room for title
    ax_bs.figure.subplots_adjust(top=0.95)
    save_fig(ax_bs, f"{FIGS_DIR}/si-phonon-bands-{model}.pdf")


# %% remove all runs
shutil.rmtree(runs_dir)
