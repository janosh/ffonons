# %%
import gzip
import json
import os
import shutil
from time import perf_counter

import atomate2.forcefields.jobs as ff_jobs
from atomate2.common.schemas.phonons import PhononBSDOSDoc as Atomate2PhononBSDOSDoc
from atomate2.forcefields.flows.phonons import PhononMaker
from jobflow import run_locally
from mp_api.client import MPRester
from pymatviz.io import save_fig

from ffonons import DATA_DIR, ROOT, bs_key, dos_key, fetch_db_data, find_last_dos_peak
from ffonons.plots import plot_phonon_bs, plot_phonon_dos

__author__ = "Janosh Riebesell"
__date__ = "2023-11-19"


# %%
runs_dir = f"{ROOT}/runs"
docs_dir = f"{DATA_DIR}/phonon-bs-dos"
shutil.rmtree(runs_dir, ignore_errors=True)  # remove old runs
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
    "CHGNet v0.3.0": dict(
        bulk_relax_maker=ff_jobs.CHGNetRelaxMaker(
            relax_kwargs=common_relax_kwds, **chgnet_kwds
        ),
        phonon_displacement_maker=ff_jobs.CHGNetStaticMaker(**chgnet_kwds),
        static_energy_maker=ff_jobs.CHGNetStaticMaker(**chgnet_kwds),
    ),
}


with open(f"{DATA_DIR}/mp-ids-with-phonons.yml") as file:
    mp_ids = file.read().splitlines()[2:]
bad_ids = [mp_id for mp_id in mp_ids if not mp_id.startswith("mp-")]
assert len(bad_ids) == 0, f"{bad_ids=}"


# %%
mpr = MPRester(mute_progress_bars=True)

for mp_id in mp_ids:
    struct, mp_doc_dict, figs_out_dir = fetch_db_data(mp_id, docs_dir)
    id_formula = f"{mp_id}-{struct.formula.replace(' ', '')}"

    for model, makers in models.items():
        key = model.lower().replace(" ", "-")
        try:
            dos_fig_path = f"{figs_out_dir}/dos-{key}.pdf"
            bands_fig_path = f"{figs_out_dir}/bands-{key}.pdf"
            ml_doc_path = f"{docs_dir}/{id_formula}-{key}.json.gz"
            if all(map(os.path.isfile, (dos_fig_path, bands_fig_path, ml_doc_path))):
                print(f"Skipping {model} for {struct.formula}")
                continue

            start = perf_counter()
            phonon_flow = PhononMaker(
                **makers,
                use_symmetrized_structure="primitive",
                min_length=15,
                store_force_constants=False,
                # use kpath_scheme="setyawan_curtarolo" when comparing to MP
                # and 'seek_path' when comparing with PhonoDB!
                # note: setyawan_curtarolo only compatible with primitive cell
                kpath_scheme="setyawan_curtarolo",
            ).make(structure=struct)

            # phonon_flow.draw_graph().show()

            os.makedirs(root_dir := f"{runs_dir}/{key}", exist_ok=True)
            result = run_locally(
                phonon_flow,
                root_dir=root_dir,
                log=False,
                ensure_success=True,
            )
            print(f"{model} took: {perf_counter() - start:.2f} s")

            ml_phonon_doc: Atomate2PhononBSDOSDoc = result[phonon_flow[-1].uuid].output

            ml_doc_dict = {
                dos_key: ml_phonon_doc.phonon_dos.as_dict(),
                bs_key: ml_phonon_doc.phonon_bandstructure.as_dict(),
            }

            with gzip.open(ml_doc_path, "wt") as file:
                file.write(json.dumps(ml_doc_dict))

            ax_dos = plot_phonon_dos(
                {model: ml_phonon_doc.phonon_dos}, f"{model} - ", struct
            )

            last_peak = find_last_dos_peak(ml_phonon_doc.phonon_dos)
            save_fig(ax_dos, dos_fig_path)

            phonon_bands = ml_phonon_doc.phonon_bandstructure
            ax_bs = plot_phonon_bs(phonon_bands, f"{model} - ", struct)
            save_fig(ax_bs, bands_fig_path)
        except (RuntimeError, ValueError) as exc:
            print(f"!!! {model} failed: {exc}")
