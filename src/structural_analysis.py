"""
Análisis estructural del grafo (§3.1 del TFG).

Calcula y representa las propiedades topológicas conservadas en el marco teórico
(Sección 1.2): distribución de tipos de relación, distribución de grados, diámetro
y componentes conexas. Genera dos figuras de soporte en transe_run_v2/.

Nota: los grados se calculan sobre un MultiDiGraph, de modo que dos relaciones
distintas entre el mismo par de entidades cuentan como dos aristas (el grado de
una entidad es su número de tripletas). El diámetro y las componentes se calculan
sobre el esqueleto no dirigido simple (Sección 1.2.1 del TFG).

Ejecución:
    python src/structural_analysis.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
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
DATA = ROOT / "dataset" / "kg_geografia_historia_v2.csv"
OUT  = ROOT / "transe_run_v2"
OUT.mkdir(exist_ok=True)


def main():
    with open(DATA, encoding="utf-8") as f:
        triples = [(r["head"], r["relation"], r["tail"]) for r in csv.DictReader(f)]

    rel_counts = Counter(r for _, r, _ in triples)

    # ---- multigrafo dirigido: el grado cuenta tripletas, no pares de nodos ----
    G = nx.MultiDiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
    deg = dict(G.degree())                       # grado total (in+out), con multiaristas

    # ---- esqueleto no dirigido simple para conectividad y diámetro ----
    Gu = nx.Graph()
    Gu.add_edges_from((h, t) for h, _, t in triples)
    components = list(nx.connected_components(Gu))
    largest = max(components, key=len)
    diameter = nx.diameter(Gu.subgraph(largest).copy())

    print(f"entidades                      : {Gu.number_of_nodes()}")
    print(f"componentes conexas (esqueleto): {len(components)}")
    print(f"diámetro (mayor componente)    : {diameter}")
    print("top-8 grado (hubs):")
    for n, d in sorted(deg.items(), key=lambda x: -x[1])[:8]:
        print(f"    {n:20s} {d}")
    print("\ndistribución de relaciones:")
    for r, c in rel_counts.most_common():
        print(f"    {r:20s} {c}")

    # ---- figura: histograma de relaciones ----
    items = sorted(rel_counts.items(), key=lambda x: -x[1])
    rels, counts = zip(*items)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(rels, counts, color="steelblue", edgecolor="black")
    plt.bar_label(bars, padding=2, fontsize=8)   # valor exacto sobre cada barra
    plt.ylim(0, max(counts) * 1.08)              # holgura para las anotaciones
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Número de tripletes")
    plt.tight_layout()
    plt.savefig(OUT / "histograma_relaciones.png", dpi=200)
    plt.close()
    print("\n  histograma_relaciones.png")

    # ---- figura: distribución de grados ----
    values = sorted(deg.values(), reverse=True)
    plt.figure(figsize=(8, 5))
    plt.plot(values, marker="o", markersize=3, linestyle="-", color="darkorange")
    plt.xlabel("Ranking de entidad (orden descendente)")
    plt.ylabel("Grado total (entrada + salida)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "distribucion_grados.png", dpi=200)
    plt.close()
    print("  distribucion_grados.png")


if __name__ == "__main__":
    main()
