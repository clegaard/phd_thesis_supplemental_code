import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator

if __name__ == "__main__":
    offset = 0.2
    contributions = {
        "C1 Self-Adaptive": (2, 2),
        "C2: SciML Taxonomy": (
            3,
            0,
        ),
        "C3: SciML model implementation": (
            0,
            2,
        ),
        "C4: Optimal Control HVAC": (3, 0 + offset),
        "C5: PythonFMU": (
            0,
            2 - offset,
        ),
        "C6: UniFMU": (
            0,
            3,
        ),
        "C7: Building Energy Modeling": (
            2,
            3,
        ),
    }

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for idx, (contribution_name, score) in enumerate(contributions.items()):
        ax.scatter(*score)
        if score[0] == 3:
            ax.annotate(contribution_name, score, rotation=60, rotation_mode="anchor")
        else:
            ax.annotate(contribution_name, score)

    ax.set_xlabel(rf"$\mathbf{{SciML}}$")
    ax.set_ylabel(rf"$\mathbf{{Co-Simulation}}$")
    ax.set_aspect(1 / 1.41)
    fig.suptitle("Significance of Contribution")

    # plt.savefig("figures/spider_plot.pdf", bbox_inches="tight")
    # plt.savefig("figures/spider_plot.svg", bbox_inches="tight")

    ax.set_xticks([0, 1, 2, 3], ["None", "Minor", "Significant", "Major"])
    ax.set_yticks([0, 1, 2, 3], ["None", "Minor", "Significant", "Major"])
    plt.tight_layout()
    plt.savefig("figures/introduction/assessment.pdf", bbox_inches="tight")

    plt.show()
