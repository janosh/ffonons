"""Plot heat capacities vs temperature for random phonon docs."""

# %%
import os
import random
import re
from collections import defaultdict
from glob import glob

import pymatviz as pmv

from ffonons import DATA_DIR, PAPER_DIR
from ffonons.enums import Model
from ffonons.io import load_pymatgen_phonon_docs
from ffonons.plots import plot_thermo_props

# Get list of unique material IDs and their available models
mat_id_models = defaultdict(set)
for path in glob(f"{DATA_DIR}/phonon-db/*.json.xz"):
    mat_id = re.search(r"(mp-\d+)", path).group(1)
    model = re.search(rf"{mat_id}-.*?-(.*?)\.json\.xz", path).group(1)
    mat_id_models[mat_id].add(model)

# Filter for materials with at least 3 models including PBE
min_models_needed = 3
eligible_mat_ids = [
    mat_id
    for mat_id, models in mat_id_models.items()
    if len(models) >= min_models_needed and "pbe" in models
]

if not eligible_mat_ids:
    raise ValueError("No materials found with 3+ models including PBE")

print(f"Found {len(eligible_mat_ids)} materials with 3+ models including PBE")

# Randomly select materials
random.seed(42)  # For reproducibility
selected_mat_ids = random.sample(eligible_mat_ids, min(8, len(eligible_mat_ids)))

# For each material ID, find all model files
selected_paths = []
for mat_id in selected_mat_ids:
    model_paths = glob(f"{DATA_DIR}/phonon-db/{mat_id}-*.json.xz")
    selected_paths.extend(model_paths)

print(f"Loading {len(selected_paths)} docs for {len(selected_mat_ids)} materials")

# Load the phonon docs
ph_docs = load_pymatgen_phonon_docs(selected_paths)

# Create directory for saving figures
os.makedirs(f"{PAPER_DIR}/heat-capacities", exist_ok=True)

# %% Plot thermodynamic properties for each structure
for mat_id, models in ph_docs.items():
    if Model.pbe not in models:
        continue

    fig = plot_thermo_props(models)
    formula = models[Model.pbe].structure.formula
    title = f"{formula} (<a href='https://materialsproject.org/materials/{mat_id}'>{mat_id}</a>)"
    fig.layout.title = dict(text=title, x=0.5, y=0.99)
    fig.show()
    img_path = f"{PAPER_DIR}/heat-capacities/{mat_id}.pdf"

    pmv.save_fig(fig, img_path, scale=2)
