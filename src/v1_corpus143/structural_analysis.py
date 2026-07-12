"""
Análisis estructural del grafo (paso 4.1.C / §3.1 del TFG).

Calcula y representa las propiedades topológicas conservadas en el marco teórico
(Sección 1.2): distribución de tipos de relación, distribución de grados, diámetro
y componentes conexas. Genera dos figuras de soporte en transe_run/.

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
DATA = ROOT / "dataset" / "kg_geografia_historia.csv"
OUT  = ROOT / "transe_run"
OUT.mkdir(exist_ok=True)


def main():
    with open(DATA, encoding="utf-8") as f:
        triples = [(r["head"], r["relation"], r["tail"]) for r in csv.DictReader(f)]

    rel_counts = Counter(r for _, r, _ in triples)

    # ---- grafo dirigido ----
    G = nx.DiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
    deg = dict(G.degree())                       # grado total (in+out)
    Gu  = G.to_undirected()

    wcc = list(nx.weakly_connected_components(G))
    largest = max(nx.connected_components(Gu), key=len)
    diameter = nx.diameter(Gu.subgraph(largest).copy())

    print(f"componentes débilmente conexas : {len(wcc)}")
    print(f"diámetro (mayor componente)    : {diameter}")
    print("top-6 grado (hubs):")
    for n, d in sorted(deg.items(), key=lambda x: -x[1])[:6]:
        print(f"    {n:16s} {d}")
    print("\ndistribución de relaciones:")
    for r, c in rel_counts.most_common():
        print(f"    {r:16s} {c}")

    # ---- figura 1: histograma de relaciones ----
    items = sorted(rel_counts.items(), key=lambda x: -x[1])
    rels, counts = zip(*items)
    plt.figure(figsize=(10, 5))
    plt.bar(rels, counts, color="steelblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Número de tripletas")
    plt.title("Distribución de tipos de relación en el corpus")
    plt.tight_layout()
    plt.savefig(OUT / "histograma_relaciones.png", dpi=200)
    plt.close()
    print("\n  histograma_relaciones.png")

    # ---- figura 2: distribución de grados ----
    values = sorted(deg.values(), reverse=True)
    plt.figure(figsize=(8, 5))
    plt.plot(values, marker="o", linestyle="-", color="darkorange")
    plt.xlabel("Ranking de entidad (orden descendente)")
    plt.ylabel("Grado total (entrada + salida)")
    plt.title("Distribución de grados del grafo de conocimiento")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "distribucion_grados.png", dpi=200)
    plt.close()
    print("  distribucion_grados.png")


if __name__ == "__main__":
    main()
