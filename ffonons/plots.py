"""Plotting utilities for phonon DOS using matplotlib. No longer used. Prefer
interactive and more customizable band structure and DOS plotting functions in pymatviz.
"""

import re
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from pymatgen.core import Structure
from pymatgen.phonon import PhononDos, PhononDosPlotter
from pymatgen.util.string import htmlify, latexify

from ffonons.enums import DB, Model, PhKey

pretty_labels = PhKey.val_label_dict() | Model.val_label_dict() | DB.val_label_dict()

px.defaults.labels |= pretty_labels


def plot_phonon_dos_mpl(
    phonon_dos: PhononDos | dict[str, PhononDos],
    title: str = "",
    struct: Structure | None = None,
    ax: plt.Axes | None = None,
    last_peak_anno: str | None = r"$\omega_\text{{max}}^{{{key}}}={last_peak:.1f}$",
) -> plt.Axes:
    r"""Plot a phonon DOS with matplotlib.

    Deprecated in favor of pymatviz.plot_phonon_dos powered by plotly.

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
    print(
        "plot_phonon_dos_mpl is deprecated, use the interactive plotly version in "
        "pymatviz.plot_phonon_dos instead",
        file=sys.stderr,
    )
    ph_dos_plot = PhononDosPlotter()
    phonon_dos = phonon_dos if isinstance(phonon_dos, dict) else {"": phonon_dos}

    for key, kwargs in phonon_dos.items():
        if isinstance(kwargs, PhononDos):
            kwargs = dict(dos=kwargs)  # noqa: PLW2901
        ph_dos_plot.add_dos(label=key, **kwargs)

    ax = ph_dos_plot.get_plot(legend=dict(fontsize=22), ax=ax)
    if struct:
        mat_id = struct.properties.get("id", "")
        title += f" {latexify(struct.formula)} {mat_id}"

    if last_peak_anno:
        _dos_x_min, dos_x_max, _dos_y_min, _dos_y_max = ax.axis()

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


def plotly_title(formula: str, href: str = "") -> str:
    """Make plotly figure title from HTML-ified formula and link to MP details page
    (legacy since only legacy has phonons) or other URL.
    """
    title = f"{htmlify(formula)}"

    if href:
        if href.lower().strip().startswith(("mp-", "mvc-")):
            link_text = href
            href = f"https://legacy.materialsproject.org/materials/{href}"
        elif "materialsproject.org" in href and re.match(r"/mp-\d+$", href):
            link_text = re.search(r"/(mp-\d+)$", href).group()
        else:
            link_text = href.split("://")[1].replace("www.", "")
        title += f'  <a href="{href}">{link_text}</a>'

    return title


def plot_thermo_props(
    models: dict[str, Any],
    subplot_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Plot thermodynamic properties (heat capacity, free energy, internal energy,
    entropy) for different models compared to PBE reference.

    Args:
        models: Dictionary mapping model names to phonon docs
        subplot_kwargs: Keyword arguments for make_subplots

    Returns:
        Plotly figure with 4 subplots showing thermodynamic properties
    """
    title_attr_map = {
        "Heat Capacity": ("heat_capacities", "C<sub>V</sub>"),
        "Free Energy": ("free_energies", "F"),
        "Internal Energy": ("internal_energies", "U"),
        "Entropy": ("entropies", "S"),
    }
    subplot_defaults = dict(
        rows=2,
        cols=2,
        subplot_titles=list(title_attr_map),
        vertical_spacing=0.09,
        horizontal_spacing=0.065,
    )
    fig = make_subplots(**subplot_defaults | (subplot_kwargs or {}))

    # Get PBE data first as reference
    pbe_doc = models[Model.pbe]

    # Store traces and metrics for each property
    traces = defaultdict(list)
    property_metrics = defaultdict(dict)

    for model_name, doc in models.items():
        temps = getattr(doc, "temps", getattr(doc, "temperatures", None))
        if temps is None:
            print(f"Warning: no temperatures found for {model_name}")
            continue

        # Calculate error metrics against PBE (only for ML models)
        if model_name != Model.pbe:
            # Calculate metrics for each property
            for title, (attr, _symbol) in title_attr_map.items():
                prop = np.array(getattr(doc, attr))
                pbe_prop = np.array(getattr(pbe_doc, attr))

                # Calculate normalized error
                pbe_range = np.ptp(pbe_prop)  # peak-to-peak (max - min)
                # Normalize by the range of the reference data
                nrmse = np.sqrt(np.mean((prop - pbe_prop) ** 2)) / pbe_range * 100
                property_metrics[title][model_name] = nrmse

            model_label = f"<b>{Model(model_name).label}</b>"
        else:
            model_label = Model(model_name).label

        # Store traces for each property
        for attr, _symbol in title_attr_map.values():
            line_kwargs = {
                "color": Model(model_name).color if model_name in Model else None,
                "dash": "dash" if model_name == Model.pbe else "solid",
            }
            trace_kwargs = {
                "x": temps,
                "y": getattr(doc, attr),
                "name": model_label,
                "line": line_kwargs,
                "model_name": model_name,
            }
            traces[attr].append(trace_kwargs)

    # Add traces to figure
    for idx, (_prop, prop_traces) in enumerate(traces.items()):
        for trace in prop_traces:
            fig.add_scatter(
                x=trace["x"],
                y=trace["y"],
                name=trace["name"],
                line=trace["line"],
                legendgroup=trace["model_name"],
                row=idx // 2 + 1,
                col=idx % 2 + 1,
                showlegend=idx == 0,
            )

    # Add annotations for model rankings in each subplot
    for idx, (title, (_attr, symbol)) in enumerate(title_attr_map.items()):
        if property_metrics[title]:
            sorted_models = sorted(property_metrics[title].items(), key=lambda x: x[1])

            # Create ranking text with colored model names
            ranking_text = ""
            for rank, (model_name, nrmse) in enumerate(sorted_models, 1):
                model_color = (
                    Model(model_name).color if model_name in Model else "black"
                )
                frac = (
                    (rank - 1) / (len(sorted_models) - 1)
                    if len(sorted_models) > 1
                    else 0
                )
                nrmse_color = f"rgb({int(255 * frac)},0,{int(255 * (1-frac))})"

                ranking_text += (
                    f"{rank}. <span style='color:{model_color}'>"
                    f"{Model(model_name).label}</span> "
                    f"(<span style='color:{nrmse_color}'>{nrmse:.1f}%</span>)<br>"
                )

            fig.add_annotation(
                text=ranking_text,
                xref=f"x{idx + 1 if idx > 0 else ''} domain",
                yref=f"y{idx + 1 if idx > 0 else ''} domain",
                x={"U": 0.02, "F": 0.02, "S": 0.02, "C<sub>V</sub>": 0.98}[symbol],
                y={"U": 0.98, "F": 0.02, "S": 0.98, "C<sub>V</sub>": 0.02}[symbol],
                showarrow=False,
                align="left",
            )

    fig.layout.legend = dict(
        y=0, xanchor="right", x=1, tracegroupgap=0, bgcolor="rgba(0,0,0,0)"
    )
    fig.layout.margin = dict(l=10, r=10, t=50, b=10)
    fig.layout.update(width=1000, height=800)

    # Update axes labels
    fig.update_xaxes(title_text="Temperature (K)", title_standoff=0)
    fig.update_yaxes(title_standoff=0)
    y_axis_titles = ["C<sub>V</sub> (J/mol/K)", "F (eV)", "U (eV)", "S (J/mol/K)"]
    for idx, title in enumerate(y_axis_titles, start=1):
        fig.layout[f"yaxis{idx}"].title.update(text=title)

    return fig
