from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.phonon import PhononBSPlotter, PhononDos, PhononDosPlotter
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.util.string import htmlify, latexify

from ffonons import find_last_dos_peak

pretty_label_map = {
    "mp": "MP",
    "mace": "MACE-MP0",
    "chgnet": "CHGNet",
    "pbe": "PBE",
    "pbesol": "PBEsol",
    "mace-y7uhwpje": "MACE-MP0",
    "chgnet-v0.3.0": "CHGNet",
    "phonon-db": "PhononDB",
    "gnome": "GNoMe",
}


def plot_phonon_bs_mpl(
    phonon_bs: PhononBandStructureSymmLine,
    title: str = "",
    struct: Structure | None = None,
) -> plt.Axes:
    """Plot phonon band structure.

    Args:
        phonon_bs: PhononBandStructureSymmLine
            Phonon band structure.
        title: str = ""
            Title of the plot.
        struct: Structure | None = None
            Structure to add to the title.

    Returns:
        Matplotlib axes
    """
    ax_bs = PhononBSPlotter(phonon_bs).get_plot()
    if struct:
        mat_id = struct.properties.get("id", "")
        title += f" {latexify(struct.formula)} {mat_id}"
    ax_bs.set_title(title, fontsize=22, fontweight="bold")
    ax_bs.figure.subplots_adjust(top=0.95)
    return ax_bs


def plot_phonon_dos_mpl(
    phonon_dos: PhononDos | dict[str, PhononDos],
    title: str = "",
    struct: Structure | None = None,
    ax: plt.Axes | None = None,
    last_peak_anno: str | None = r"$\omega_\text{{max}}^{{{key}}}={last_peak:.1f}$",
) -> plt.Axes:
    r"""Plot phonon DOS.

    Args:
        phonon_dos: PhononDos | dict[str, PhononDos]
            Phonon DOS.
        title: str = ""
            Title of the plot.
        struct: Structure | None = None
            Structure to add to the title.
        ax: plt.Axes | None = None
            Matplotlib axes to plot on. If None, a new figure is created.
        last_peak_anno: str = r"$\\omega_\text{{max}}^{{{key}}}={last_peak:.1f}$ THz"
            Annotation for last DOS peak with f-string placeholders for key (of DOS in
            passed phonon_dos dict) and last_peak (frequency in THz). Set to None or
            empty string to disable.

    Returns:
        Matplotlib axes
    """
    ph_dos_plot = PhononDosPlotter()
    phonon_dos = phonon_dos if isinstance(phonon_dos, dict) else {"": phonon_dos}

    for key, kwargs in phonon_dos.items():
        if isinstance(kwargs, PhononDos):
            kwargs = dict(dos=kwargs)
        ph_dos_plot.add_dos(label=key, **kwargs)

    ax = ph_dos_plot.get_plot(legend=dict(fontsize=22), ax=ax)
    if struct:
        mat_id = struct.properties.get("id", "")
        title += f" {latexify(struct.formula)} {mat_id}"

    if last_peak_anno:
        _, dos_x_max, _, dos_y_max = ax.axis()

        for idx, (key, dos) in enumerate(phonon_dos.items()):
            last_peak = find_last_dos_peak(dos)
            line_dict = {line.get_label(): line for line in ax.lines}
            # use line with same index if no label matches
            line = line_dict.get(key, ax.lines[idx])
            color = line.get_color()

            single_anchor = dict(
                va="bottom", ha="center" if last_peak > 0.7 * dos_x_max else "left"
            )
            # multi_anchor = dict(rotation=90, va="top", ha="right")  # vertical mode
            multi_anchor = dict(
                va="top", ha="right", xytext=(-5, -5)
            )  # horizontal mode
            ax.axvline(last_peak, color=color, linestyle="--", label="last peak")
            ax.annotate(
                last_peak_anno.format(key=key.split(" ")[0], last_peak=last_peak),
                (last_peak, 1 - idx * 0.05),
                # textcoords=ax.transAxes,
                # use data/axes coordinates for x/y respectively
                xycoords=("data", "axes fraction"),
                textcoords="offset points",
                fontsize=16,
                color=color,
                **(single_anchor if len(phonon_dos) == 1 else multi_anchor),
            )

    ax.set_title(title, fontsize=22)
    ax.figure.subplots_adjust(top=0.95)
    return ax


def plotly_title(formula: str, mat_id: str = "") -> str:
    """Make plotly figure title from HTML-ified formula and and link to MP details page
    (legacy since only legacy has phonons) or other URL.
    """
    if mat_id.startswith(("mp-", "mvc-")):
        href = f"https://legacy.materialsproject.org/materials/{mat_id}"
    if mat_id.startswith("https://"):
        href = mat_id
        mat_id = mat_id.split("/")[-1]
    title = f"{htmlify(formula)}"
    if mat_id:
        title += f'  <a href="{href}">{mat_id}</a>'
    return title
