from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter


def plot_phonon_bs(
    struct: Structure, phonon_bs: PhononBandStructureSymmLine, source: str
) -> plt.Axes:
    """Plot phonon band structure."""
    mat_id = struct.properties["id"]
    bs_plot = PhononBSPlotter(phonon_bs)
    ax_bs = bs_plot.get_plot()
    title_prefix = f"{source} {struct.formula} ({mat_id}) Phonon"
    ax_bs.set_title(f"{title_prefix} Band Structure", fontsize=22)
    ax_bs.figure.subplots_adjust(top=0.95)
    return ax_bs


def plot_phonon_dos(struct: Structure, phonon_dos: PhononDos, source: str) -> plt.Axes:
    """Plot phonon DOS."""
    mat_id = struct.properties["id"]
    dos_plot = PhononDosPlotter(phonon_dos)
    ax_dos = dos_plot.get_plot()
    title_prefix = f"{source} {struct.formula} ({mat_id}) Phonon"
    ax_dos.set_title(f"{title_prefix} DOS", fontsize=22)
    ax_dos.figure.subplots_adjust(top=0.95)
    return ax_dos
