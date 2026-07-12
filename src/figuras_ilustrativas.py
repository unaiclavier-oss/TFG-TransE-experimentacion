"""
Figuras ilustrativas del marco teórico (Figuras 1 y 2 del TFG).

Genera las dos figuras de ejemplo de la Sección 1 con una plantilla visual
unificada:

  - Figura 1 (einstein_graph.png): porción del KG en torno a "Albert Einstein"
    (Introducción / Sección 1.2.1).
  - Figura 2 (knowledge_graph_toy.png): grafo de ejemplo de "Alan Turing"
    (Sección 1.5), consistente con la matriz de adyacencia dirigida M.

Ambas se dibujan como grafos DIRIGIDOS, con puntas de flecha que reflejan la
orientación de cada triplete (imprescindible para reconstruir M desde la figura),
mismo color de nodo, mismo color de etiqueta de relación, sin título incrustado
(esa función la cumple el caption numerado) y con las etiquetas de nodo desplazadas
fuera del círculo para evitar solapamientos.

Ejecución:
    python src/figuras_ilustrativas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "latex" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)

# ---- plantilla visual común a las dos figuras ilustrativas ----
NODE_COLOR   = "#a9d3ec"   # azul claro (mismo en ambas figuras)
NODE_EDGE    = "black"
NODE_SIZE    = 2600
REL_COLOR    = "#c1121f"   # rojo oscuro para las etiquetas de relación
EDGE_COLOR   = "#555555"
LABEL_FONT   = 12
REL_FONT     = 11


def draw_kg(edges, pos, node_labels, label_offsets, filename, figsize):
    """Dibuja un KG dirigido con la plantilla común y lo guarda en OUT/filename."""
    G = nx.DiGraph()
    G.add_nodes_from(pos)
    for h, r, t in edges:
        G.add_edge(h, t, relation=r)

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_color=NODE_COLOR, node_size=NODE_SIZE,
                           edgecolors=NODE_EDGE, linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, node_size=NODE_SIZE, edge_color=EDGE_COLOR,
                           width=1.8, arrows=True, arrowstyle="-|>",
                           arrowsize=22, min_source_margin=2, min_target_margin=2,
                           ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(h, t): r for h, r, t in edges},
        font_color=REL_COLOR, font_size=REL_FONT, rotate=True,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        ax=ax)

    # Etiquetas de nodo desplazadas fuera del círculo (evita el solapamiento).
    for n, (x, y) in pos.items():
        dx, dy = label_offsets[n]
        ax.text(x + dx, y + dy, node_labels[n], fontsize=LABEL_FONT,
                fontweight="bold", ha="center", va="center")

    ax.margins(0.18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {filename}")


def figura_einstein():
    # Albert Einstein en el centro; tres tripletes salientes (star).
    # Rótulos con guion bajo, idénticos a los identificadores del texto (§1.2.1).
    edges = [
        ("Albert_Einstein", "lugar_de_nacimiento", "Ulm"),
        ("Albert_Einstein", "cónyuge",            "Mileva_Marić"),
        ("Albert_Einstein", "campo_de_estudio",   "Física"),
    ]
    pos = {
        "Albert_Einstein": (0.0, 0.0),
        "Ulm":             (1.7, 0.15),
        "Física":          (-1.4, 1.05),
        "Mileva_Marić":    (-0.25, -1.35),
    }
    node_labels = {n: n for n in pos}
    label_offsets = {
        "Albert_Einstein": (0.0, -0.42),
        "Ulm":             (0.0,  0.40),
        "Física":          (0.0,  0.40),
        "Mileva_Marić":    (0.0, -0.42),
    }
    draw_kg(edges, pos, node_labels, label_offsets,
            "einstein_graph.png", figsize=(11, 7))


def figura_turing():
    # Grafo toy de Alan Turing (Sección 1.5), consistente con la matriz M.
    # Rótulos con guion bajo, idénticos a los identificadores del texto (§1.5).
    edges = [
        ("Alan_Turing",    "nació_en",   "Londres"),
        ("Londres",        "ciudad_de",  "Reino_Unido"),
        ("Alan_Turing",    "trabajó_en", "Bletchley_Park"),
        ("Bletchley_Park", "ubicado_en", "Reino_Unido"),
        ("Alan_Turing",    "estudió_en", "Univ_Cambridge"),
        ("Alan_Turing",    "descifró",   "Máquina_Enigma"),
    ]
    pos = {
        "Alan_Turing":     (0.0, 0.0),
        "Univ_Cambridge":  (-1.55, 1.15),
        "Londres":         (1.35, 1.35),
        "Reino_Unido":     (2.75, 0.0),
        "Bletchley_Park":  (1.35, -1.35),
        "Máquina_Enigma":  (-1.55, -1.15),
    }
    node_labels = {n: n for n in pos}
    label_offsets = {
        "Alan_Turing":     (0.0, -0.42),
        "Univ_Cambridge":  (0.0,  0.42),
        "Londres":         (0.0,  0.42),
        "Reino_Unido":     (0.0,  0.42),
        "Bletchley_Park":  (0.0, -0.42),
        "Máquina_Enigma":  (0.0, -0.42),
    }
    draw_kg(edges, pos, node_labels, label_offsets,
            "knowledge_graph_toy.png", figsize=(12, 8))


def main():
    figura_einstein()
    figura_turing()


if __name__ == "__main__":
    main()
