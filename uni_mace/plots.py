from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter


def plot_phonon_bs(
    phonon_bs: PhononBandStructureSymmLine, struct: Structure, source: str
) -> plt.Axes:
    """Plot phonon band structure."""
    mat_id = struct.properties.get("id", "")
    ax_bs = PhononBSPlotter(phonon_bs).get_plot()
    title_prefix = f"{source} {struct.formula} ({mat_id}) Phonon"
    ax_bs.set_title(f"{title_prefix} Band Structure", fontsize=22)
    ax_bs.figure.subplots_adjust(top=0.95)
    return ax_bs


def plot_phonon_dos(phonon_dos: PhononDos, struct: Structure, source: str) -> plt.Axes:
    """Plot phonon DOS."""
    mat_id = struct.properties.get("id", "")
    ph_dos_plot = PhononDosPlotter()
    ph_dos_plot.add_dos(dos=phonon_dos, label=source)
    ax_dos = ph_dos_plot.get_plot(legend=dict(fontsize=22))
    title_prefix = f"{source} {struct.formula} ({mat_id}) Phonon"
    ax_dos.set_title(f"{title_prefix} DOS", fontsize=22)
    ax_dos.figure.subplots_adjust(top=0.95)
    return ax_dos
