import plotly.express as px
from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.phonon import PhononDos, PhononDosPlotter
from pymatgen.util.string import htmlify, latexify

from ffonons.enums import DB, Key, Model

pretty_labels = Key.val_label_dict() | Model.val_label_dict() | DB.val_label_dict()

px.defaults.labels |= pretty_labels


def plot_phonon_dos_mpl(
    phonon_dos: PhononDos | dict[str, PhononDos],
    title: str = "",
    struct: Structure | None = None,
    ax: plt.Axes | None = None,
    last_peak_anno: str | None = r"$\omega_\text{{max}}^{{{key}}}={last_peak:.1f}$",
) -> plt.Axes:
    r"""Plot phonon DOS.

    Args:
        phonon_dos (PhononDos | dict[str, PhononDos]): Single phonon DOS or dict of
            DOSes with plot labels as keys.
        title (str = ""): Plot title. Will be appended with formula and material ID if
            struct is passed.
        struct (Structure | None = None): Structure whose formula and material ID to add
            to the title.
        ax (plt.Axes | None = None): Matplotlib axes to plot on. If None, a new figure
            is created.
        last_peak_anno (str = r"$\\omega_\text{{max}}^{{{key}}}={last_peak:.1f}$ THz"):
            Annotation for last DOS peak with f-string placeholders for key (of DOS in
            passed phonon_dos dict) and last_peak (frequency in THz). Set to None or
            empty string to disable.

    Returns:
        plt.Axes: Matplotlib axes
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
            last_peak = dos.get_last_peak()
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
    if mat_id.lower().strip().startswith(("mp-", "mvc-")):
        href = f"https://legacy.materialsproject.org/materials/{mat_id}"
    elif mat_id.startswith("https://"):
        href = mat_id
        mat_id = mat_id.split("/")[-1]
    else:
        raise ValueError(f"unrecognized {mat_id=}")
    title = f"{htmlify(formula)}"
    if mat_id:
        title += f'  <a href="{href}">{mat_id}</a>'
    return title
