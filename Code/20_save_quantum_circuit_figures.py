
r"""
Generate clean publication-grade AQUA-SLA quantum circuit figures.

Corrections:
- Smaller control dots
- Thinner circuit lines
- Better gate spacing
- No overlapping labels
- Explicit initialization and measurement
- Additional quantum circuits:
    1. State preparation
    2. Hybrid feature map
    3. Quantum state evolution
    4. Variational ansatz
    5. Fidelity quantum-kernel circuit
    6. SWAP-test kernel circuit
    7. QAOA cost layer
    8. QAOA mixer layer
    9. Full QAOA p=1
   10. Full QAOA p=2
   11. Full QAOA p=3

All figures are schematic paper diagrams. Replace symbolic blocks with the
exact implemented circuit if the production code differs.

Output:
D:\other\AQUA-SLA\results\aqua_sla_final\sample_size_study\
saved_models\quantum_circuit_figures_final

Format:
- JPEG only
- 1920 x 1080 pixels
- JPEG quality 100
- no chroma subsampling
- 900-DPI metadata
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc


OUTPUT_DIR = Path(
    r"D:\other\AQUA-SLA\results\aqua_sla_final"
    r"\sample_size_study\saved_models\quantum_circuit_figures_final"
)

WIDTH_PX = 1920
HEIGHT_PX = 1080
RENDER_DPI = 160
JPEG_DPI = 900

LINE_WIDTH = 1.45
GATE_EDGE_WIDTH = 1.45
CONTROL_RADIUS = 0.045
TARGET_RADIUS = 0.115

QUBIT_Y = [3.55, 2.65, 1.75, 0.85]


def add_gate(
    ax,
    x,
    y,
    label,
    width=0.72,
    height=0.46,
    fontsize=13,
    facecolor="white",
):
    rect = Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        linewidth=GATE_EDGE_WIDTH,
        edgecolor="black",
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=4,
    )


def add_small_control(ax, x, y):
    ax.add_patch(
        Circle(
            (x, y),
            CONTROL_RADIUS,
            facecolor="black",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
    )


def add_cnot(ax, x, control_y, target_y):
    ax.plot(
        [x, x],
        [control_y, target_y],
        color="black",
        linewidth=LINE_WIDTH,
        zorder=2,
    )
    add_small_control(ax, x, control_y)
    ax.add_patch(
        Circle(
            (x, target_y),
            TARGET_RADIUS,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
    )
    ax.plot(
        [x - TARGET_RADIUS * 0.7, x + TARGET_RADIUS * 0.7],
        [target_y, target_y],
        color="black",
        linewidth=1.1,
        zorder=6,
    )
    ax.plot(
        [x, x],
        [target_y - TARGET_RADIUS * 0.7, target_y + TARGET_RADIUS * 0.7],
        color="black",
        linewidth=1.1,
        zorder=6,
    )


def add_controlled_rotation(
    ax,
    x,
    control_y,
    target_y,
    label,
    fontsize=10.5,
    width=0.92,
    facecolor="#eef8ee",
):
    ax.plot(
        [x, x],
        [control_y, target_y],
        linewidth=LINE_WIDTH,
        color="black",
        zorder=2,
    )
    add_small_control(ax, x, control_y)
    add_gate(
        ax,
        x,
        target_y,
        label,
        width=width,
        height=0.44,
        fontsize=fontsize,
        facecolor=facecolor,
    )


def add_measurement(ax, x, y, c_index):
    add_gate(
        ax,
        x,
        y,
        "M",
        width=0.58,
        height=0.46,
        fontsize=12,
        facecolor="#f3f3f3",
    )
    ax.plot(
        [x + 0.29, x + 0.67],
        [y, y],
        color="black",
        linewidth=1.25,
    )
    ax.text(
        x + 0.77,
        y,
        rf"$c_{{{c_index}}}$",
        fontsize=12.5,
        va="center",
        ha="left",
    )


def add_measurement_meter(ax, x, y):
    add_gate(
        ax,
        x,
        y,
        "",
        width=0.62,
        height=0.48,
        facecolor="#f3f3f3",
    )
    ax.add_patch(
        Arc(
            (x, y - 0.015),
            0.34,
            0.27,
            theta1=0,
            theta2=180,
            linewidth=1.0,
        )
    )
    ax.plot(
        [x, x + 0.105],
        [y - 0.015, y + 0.09],
        linewidth=1.0,
        color="black",
    )


def setup_canvas(title, qubits=4, x_max=13.8):
    fig, ax = plt.subplots(
        figsize=(WIDTH_PX / RENDER_DPI, HEIGHT_PX / RENDER_DPI),
        dpi=RENDER_DPI,
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(-1.0, 5.0)
    ax.axis("off")

    ax.text(
        x_max / 2,
        4.63,
        title,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )

    ys = QUBIT_Y[:qubits]

    for index, y in enumerate(ys):
        ax.plot(
            [1.55, x_max - 1.15],
            [y, y],
            linewidth=LINE_WIDTH,
            color="black",
        )
        ax.text(
            1.18,
            y,
            rf"$q_{{{index}}}:|0\rangle$",
            ha="right",
            va="center",
            fontsize=15,
        )

    return fig, ax, ys


def save_jpeg(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temp_png = OUTPUT_DIR / f"_{filename}.png"
    final_jpeg = OUTPUT_DIR / f"{filename}.jpeg"

    fig.savefig(
        temp_png,
        dpi=RENDER_DPI,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)

    image = Image.open(temp_png).convert("RGB")
    image = image.resize(
        (WIDTH_PX, HEIGHT_PX),
        Image.Resampling.LANCZOS,
    )
    image.save(
        final_jpeg,
        "JPEG",
        quality=100,
        subsampling=0,
        optimize=True,
        dpi=(JPEG_DPI, JPEG_DPI),
    )
    temp_png.unlink(missing_ok=True)

    print(f"Saved: {final_jpeg}")


def draw_state_preparation():
    fig, ax, ys = setup_canvas(
        "Quantum State Preparation and Measurement"
    )

    for i, y in enumerate(ys):
        add_gate(ax, 2.35, y, "H", facecolor="#f2f2f2")
        add_gate(
            ax,
            4.05,
            y,
            rf"$R_Y(\theta_{i})$",
            width=1.18,
            facecolor="#fff4e6",
        )
        add_gate(
            ax,
            5.65,
            y,
            rf"$R_Z(\phi_{i})$",
            width=1.18,
            facecolor="#eaf3ff",
        )

    add_cnot(ax, 7.0, ys[0], ys[1])
    add_cnot(ax, 7.75, ys[1], ys[2])
    add_cnot(ax, 8.5, ys[2], ys[3])

    ax.text(
        9.55,
        2.17,
        r"$|\psi(\boldsymbol{\theta},\boldsymbol{\phi})\rangle$",
        fontsize=17,
        ha="center",
        va="center",
    )

    for i, y in enumerate(ys):
        add_measurement(ax, 11.25, y, i)

    save_jpeg(fig, "01_State_Preparation_to_Measurement")


def draw_hybrid_feature_map():
    fig, ax, ys = setup_canvas(
        "Full Hybrid Quantum Feature Map"
    )

    for i, y in enumerate(ys):
        add_gate(ax, 2.1, y, "H", facecolor="#f2f2f2")
        add_gate(
            ax,
            3.45,
            y,
            rf"$R_Z(2x_{i})$",
            width=1.16,
            fontsize=12,
            facecolor="#eaf3ff",
        )
        add_gate(
            ax,
            4.8,
            y,
            rf"$R_Y(\theta_{i})$",
            width=1.16,
            fontsize=12,
            facecolor="#fff4e6",
        )

    for j, x in enumerate([6.15, 7.4, 8.65]):
        add_controlled_rotation(
            ax,
            x,
            ys[j],
            ys[j + 1],
            rf"$R_{{ZZ}}(\varphi_{{{j},{j+1}}})$",
            width=0.98,
            fontsize=9.7,
        )

    ax.text(
        9.95,
        2.15,
        r"$|\phi(\mathbf{x},\boldsymbol{\theta})\rangle$",
        fontsize=17,
        ha="center",
        va="center",
    )

    for i, y in enumerate(ys):
        add_measurement(ax, 11.5, y, i)

    save_jpeg(fig, "02_Full_Hybrid_Feature_Map")


def draw_state_evolution():
    fig, ax = plt.subplots(
        figsize=(WIDTH_PX / RENDER_DPI, HEIGHT_PX / RENDER_DPI),
        dpi=RENDER_DPI,
    )
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        7,
        5.55,
        "Quantum State Evolution in AQUA-SLA",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )

    stages = [
        (1.25, r"$|0\rangle^{\otimes n}$", "Initialization"),
        (3.55, r"$|+\rangle^{\otimes n}$", "Superposition"),
        (5.95, r"$U_{\phi}(\mathbf{x})|+\rangle$", "Feature encoding"),
        (8.55, r"$U_C(\gamma)U_B(\beta)|\phi\rangle$", "QAOA evolution"),
        (11.15, r"$|\psi^\ast\rangle$", "Optimized state"),
        (13.0, r"$z\in\{0,1\}^{n}$", "Measurement"),
    ]

    y = 3.15

    for index, (x, state, label) in enumerate(stages):
        width = 1.65
        rect = Rectangle(
            (x - width / 2, y - 0.60),
            width,
            1.20,
            linewidth=GATE_EDGE_WIDTH,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y + 0.10,
            state,
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.text(
            x,
            y - 0.34,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
        )

        if index < len(stages) - 1:
            next_x = stages[index + 1][0]
            ax.add_patch(
                FancyArrowPatch(
                    (x + width / 2 + 0.04, y),
                    (next_x - width / 2 - 0.04, y),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.4,
                    color="black",
                )
            )

    ax.text(
        7,
        1.35,
        (
            "Initialization  →  feature encoding  →  variational evolution  "
            "→  computational-basis measurement"
        ),
        ha="center",
        va="center",
        fontsize=13,
    )

    save_jpeg(fig, "03_Quantum_State_Evolution")


def draw_variational_ansatz():
    fig, ax, ys = setup_canvas(
        "Hardware-Efficient Variational Ansatz"
    )

    for i, y in enumerate(ys):
        add_gate(
            ax,
            2.45,
            y,
            rf"$R_Y(\theta_{{{i},1}})$",
            width=1.25,
            fontsize=11.5,
            facecolor="#fff4e6",
        )
        add_gate(
            ax,
            4.0,
            y,
            rf"$R_Z(\phi_{{{i},1}})$",
            width=1.25,
            fontsize=11.5,
            facecolor="#eaf3ff",
        )

    add_cnot(ax, 5.55, ys[0], ys[1])
    add_cnot(ax, 6.2, ys[1], ys[2])
    add_cnot(ax, 6.85, ys[2], ys[3])

    for i, y in enumerate(ys):
        add_gate(
            ax,
            8.15,
            y,
            rf"$R_Y(\theta_{{{i},2}})$",
            width=1.25,
            fontsize=11.5,
            facecolor="#fff4e6",
        )
        add_gate(
            ax,
            9.7,
            y,
            rf"$R_Z(\phi_{{{i},2}})$",
            width=1.25,
            fontsize=11.5,
            facecolor="#eaf3ff",
        )

    for i, y in enumerate(ys):
        add_measurement(ax, 11.45, y, i)

    save_jpeg(fig, "04_Hardware_Efficient_Variational_Ansatz")


def draw_fidelity_kernel():
    fig, ax, ys = setup_canvas(
        "Fidelity Quantum Kernel Circuit"
    )

    for i, y in enumerate(ys):
        add_gate(
            ax,
            2.75,
            y,
            r"$U_{\phi}(\mathbf{x})$",
            width=1.6,
            fontsize=12.5,
            facecolor="#eaf3ff",
        )
        add_gate(
            ax,
            5.25,
            y,
            r"$U_{\phi}^{\dagger}(\mathbf{x}')$",
            width=1.9,
            fontsize=12.5,
            facecolor="#fff4e6",
        )

    ax.text(
        7.55,
        2.15,
        r"$K(\mathbf{x},\mathbf{x}')="
        r"|\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$",
        fontsize=16,
        ha="center",
        va="center",
    )

    for i, y in enumerate(ys):
        add_measurement(ax, 10.7, y, i)

    ax.text(
        12.15,
        2.15,
        r"$P(0^{\otimes n})$",
        fontsize=16,
        ha="center",
        va="center",
    )

    save_jpeg(fig, "05_Fidelity_Quantum_Kernel_Circuit")


def draw_swap_test():
    fig, ax = plt.subplots(
        figsize=(WIDTH_PX / RENDER_DPI, HEIGHT_PX / RENDER_DPI),
        dpi=RENDER_DPI,
    )
    ax.set_xlim(0, 13.8)
    ax.set_ylim(-0.8, 5.2)
    ax.axis("off")

    ax.text(
        6.9,
        4.78,
        "SWAP-Test Quantum Kernel Circuit",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )

    ys = [3.75, 2.65, 1.55]
    labels = [
        r"$a:|0\rangle$",
        r"$|\phi(\mathbf{x})\rangle$",
        r"$|\phi(\mathbf{x}')\rangle$",
    ]

    for label, y in zip(labels, ys):
        ax.plot(
            [1.8, 12.3],
            [y, y],
            linewidth=LINE_WIDTH,
            color="black",
        )
        ax.text(
            1.45,
            y,
            label,
            ha="right",
            va="center",
            fontsize=14,
        )

    add_gate(ax, 2.6, ys[0], "H", facecolor="#f2f2f2")

    x_swap = 6.25
    add_small_control(ax, x_swap, ys[0])
    ax.plot(
        [x_swap, x_swap],
        [ys[0], ys[2]],
        color="black",
        linewidth=LINE_WIDTH,
    )

    for y in [ys[1], ys[2]]:
        ax.text(
            x_swap,
            y,
            "×",
            fontsize=18,
            ha="center",
            va="center",
        )

    add_gate(ax, 9.0, ys[0], "H", facecolor="#f2f2f2")
    add_measurement_meter(ax, 10.55, ys[0])

    ax.text(
        11.95,
        3.75,
        r"$P(a=0)=\frac{1+|\langle\phi(\mathbf{x})|"
        r"\phi(\mathbf{x}')\rangle|^2}{2}$",
        fontsize=13,
        ha="center",
        va="center",
    )

    save_jpeg(fig, "06_SWAP_Test_Quantum_Kernel")


def draw_qaoa_cost_layer():
    fig, ax, ys = setup_canvas(
        "QAOA Cost Hamiltonian Layer"
    )

    for j, x in enumerate([3.0, 5.1, 7.2]):
        add_controlled_rotation(
            ax,
            x,
            ys[j],
            ys[j + 1],
            r"$R_{ZZ}(2\gamma w_{ij})$",
            width=1.45,
            fontsize=10.5,
            facecolor="#eef8ee",
        )

    for i, y in enumerate(ys):
        add_gate(
            ax,
            9.4,
            y,
            rf"$R_Z(2\gamma h_{i})$",
            width=1.45,
            fontsize=10.5,
            facecolor="#eaf3ff",
        )

    ax.text(
        11.55,
        2.15,
        r"$U_C(\gamma)=e^{-i\gamma H_C}$",
        fontsize=17,
        ha="center",
        va="center",
    )

    save_jpeg(fig, "07_QAOA_Cost_Hamiltonian_Layer")


def draw_qaoa_mixer_layer():
    fig, ax, ys = setup_canvas(
        "QAOA Mixer Layer"
    )

    for i, y in enumerate(ys):
        add_gate(
            ax,
            4.3,
            y,
            rf"$R_X(2\beta)$",
            width=1.30,
            fontsize=12,
            facecolor="#fff4e6",
        )

    ax.text(
        8.2,
        2.15,
        r"$U_B(\beta)=\prod_{i=1}^{n}e^{-i\beta X_i}$",
        fontsize=18,
        ha="center",
        va="center",
    )

    save_jpeg(fig, "08_QAOA_Mixer_Layer")


def draw_full_qaoa(depth):
    fig, ax, ys = setup_canvas(
        rf"Full QAOA Scheduling Circuit ($p={depth}$)"
    )

    for y in ys:
        add_gate(ax, 2.0, y, "H", facecolor="#f2f2f2")

    if depth == 1:
        layer_centers = [4.6]
    elif depth == 2:
        layer_centers = [4.0, 7.4]
    else:
        layer_centers = [3.55, 6.45, 9.35]

    for layer, center in enumerate(layer_centers, start=1):
        x_cost = center - 0.5
        x_mix = center + 0.65

        for j in range(3):
            add_controlled_rotation(
                ax,
                x_cost + j * 0.11,
                ys[j],
                ys[j + 1],
                rf"$ZZ(\gamma_{layer})$",
                width=0.85,
                fontsize=9.0,
            )

        for y in ys:
            add_gate(
                ax,
                x_mix,
                y,
                rf"$R_X(2\beta_{layer})$",
                width=1.0,
                fontsize=10.0,
                facecolor="#fff4e6",
            )

        ax.text(
            center,
            4.10,
            rf"Layer {layer}",
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
        )

    measure_x = 11.45

    for i, y in enumerate(ys):
        add_measurement(ax, measure_x, y, i)

    save_jpeg(
        fig,
        f"{8+depth:02d}_Full_QAOA_Circuit_p_{depth}",
    )


def main():
    draw_state_preparation()
    draw_hybrid_feature_map()
    draw_state_evolution()
    draw_variational_ansatz()
    draw_fidelity_kernel()
    draw_swap_test()
    draw_qaoa_cost_layer()
    draw_qaoa_mixer_layer()

    for depth in [1, 2, 3]:
        draw_full_qaoa(depth)

    print("\nCompleted.")
    print(f"Output directory:\n{OUTPUT_DIR}")
    print("All figures are 1920 x 1080 JPEG with 900-DPI metadata.")


if __name__ == "__main__":
    main()
