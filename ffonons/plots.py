from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter


def plot_phonon_bs(
    phonon_bs: PhononBandStructureSymmLine,
    title: str = "",
    struct: Structure | None = None,
) -> plt.Axes:
    """Plot phonon band structure."""
    ax_bs = PhononBSPlotter(phonon_bs).get_plot()
    if struct:
        mat_id = struct.properties.get("id", "")
        title += f" {struct.formula} {mat_id}"
    ax_bs.set_title(title, fontsize=22, fontweight="bold")
    ax_bs.figure.subplots_adjust(top=0.95)
    return ax_bs


def plot_phonon_dos(
    phonon_dos: PhononDos | dict[str, PhononDos],
    title: str = "",
    struct: Structure | None = None,
) -> plt.Axes:
    """Plot phonon DOS."""
    ph_dos_plot = PhononDosPlotter()

    for key, dos in (
        phonon_dos if isinstance(phonon_dos, dict) else {"": phonon_dos}
    ).items():
        ph_dos_plot.add_dos(dos=dos, label=key)

    ax_dos = ph_dos_plot.get_plot(legend=dict(fontsize=22))
    if struct:
        mat_id = struct.properties.get("id", "")
        title += f" {struct.formula} {mat_id}"
    ax_dos.set_title(title, fontsize=22, fontweight="bold")
    ax_dos.figure.subplots_adjust(top=0.95)
    return ax_dos
