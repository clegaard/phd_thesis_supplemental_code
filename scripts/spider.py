import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == "__main__":
    score = (0.0, 0.0, 0.0, 0.0)
    NONE = 0.0
    MINOR = 0.33
    SIGNIFICANT = 0.66
    MAJOR = 1.00

    publications = {
        "Constructing Neural Network based Models for Simulating Dynamical Systems": (
            MAJOR,
            MAJOR,
            NONE,
            SIGNIFICANT,
        ),
        "A Universal Mechanism for Implementing Functional Mock-up Units ": (
            NONE,
            NONE,
            MAJOR,
            NONE,
        ),
        "NeuroMANCER Framework": score,
        "Identification and Control of Networked Dynamical Systems: A case study in HVAC": score,
        "Portable runtime environments for Python-based FMUs: Adding Docker support to UniFMU": score,
        "Coupling physical and machine learning models: case study of a single-family house": score,
        "Energy Prediction under Changed Demand Conditions: Robust Machine Learning Models and Input Feature Combinations": score,
        "Rapid Prototyping of Self-Adaptive-Systems using Python Functional Mock-up Units": score,
    }
    publications = {f"P{i+1}": s for i, s in enumerate(publications.values())}
    criteria = [
        "RQ1: How to pick the right model?",
        "RQ2: How to integrate knowledge?",
        "RQ3: How to integrate in simulation tools?",
        "RQ4: Novel applications of SciML methods?",
    ]

    criteria_to_pub_score = {
        c: [s[idx] for s in publications.values()] for idx, c in enumerate(criteria)
    }

    n_citeria = len(criteria)
    # data = [
    #     publications,
    #     *[(cri, [np.random.uniform(size=len(publications))]) for cri in criteria],
    # ]

    theta = radar_factory(len(publications), frame="polygon")

    # data = example_data()

    fig, axs = plt.subplots(
        figsize=(9, 9),
        nrows=2,
        ncols=2,
        subplot_kw=dict(projection="radar"),
    )
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ["b", "r", "g"]  # ['b', 'r', 'g', 'm', 'y']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, criteria_to_pub_score.items()):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(
            title,
            weight="bold",
            size="medium",
            position=(0.5, 1.1),
            horizontalalignment="center",
            verticalalignment="center",
        )
        for d, color in zip(case_data, colors):
            ax.plot(theta, case_data, color=color)
            ax.fill(theta, case_data, facecolor=color, alpha=0.25, label="_nolegend_")
        ax.set_varlabels(publications)

    # add legend relative to top-left plot
    # labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    # legend = axs[0, 0].legend(labels, loc=(0.9, .95),
    #                           labelspacing=0.1, fontsize='small')

    # fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')
    plt.tight_layout()
    plt.savefig("figures/spider_plot.pdf", bbox_inches="tight")
    plt.savefig("figures/spider_plot.svg", bbox_inches="tight")

    plt.show()
