"""
Paso ⑦: evaluación cualitativa para §3.3. Produce la curva de pérdida,
la proyección PCA, la aritmética vectorial sobre test y la confrontación
con las Proposiciones 2.2.1 / 2.2.3 / 2.2.5 (+ inversiones):

  - Prop. 2.2.1 sobre las dos relaciones simétricas (limitrofe_con, contemporaneo_de).
  - Prop. 2.2.3 como gradiente: ratio de colapso frente a fan-out medio para
    goberno / tiene_ciudad / incluye_pais, con un control de tres niveles
    (global vs. mismo tipo vs. misma cabeza) que separa el colapso 1-a-N de
    la mera similitud semántica.
  - Prop. 2.2.5 sobre las dos reglas composicionales.
  - Inversión sobre los tres pares inversos.

Resultados numéricos en transe_run_v2/proposition_results.json (+ CSVs).

Ejecución:
    python src/evaluate_transe.py
"""

import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from sklearn.decomposition import PCA

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RUN  = ROOT / "transe_run_v2"
DATA = ROOT / "dataset" / "kg_geografia_historia_v2.csv"

SYMMETRIC    = ["limitrofe_con", "contemporaneo_de"]
ONE_TO_N     = ["goberno", "tiene_ciudad", "incluye_pais"]
COMPOSITIONS = [("nacionalidad", "nacio_en", "ubicada_en"),
                ("situada_en_region", "ubicada_en", "parte_de")]
INVERSES     = [("capital_de", "tiene_capital"),
                ("ubicada_en", "tiene_ciudad"),
                ("incluye_pais", "parte_de")]


def load():
    ent_emb = np.load(RUN / "entity_embeddings.npy")
    rel_emb = np.load(RUN / "relation_embeddings.npy")
    entity2id   = json.loads((RUN / "entity2id.json").read_text(encoding="utf-8"))
    relation2id = json.loads((RUN / "relation2id.json").read_text(encoding="utf-8"))
    id2entity = {i: e for e, i in entity2id.items()}

    def read_csv(path):
        with open(path, encoding="utf-8") as f:
            return [(r["head"], r["relation"], r["tail"]) for r in csv.DictReader(f)]

    return (ent_emb, rel_emb, entity2id, relation2id, id2entity,
            read_csv(DATA), read_csv(RUN / "test.csv"))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mean_pairwise(E):
    if len(E) < 2:
        return None
    return float(np.mean([np.linalg.norm(E[a] - E[b])
                          for a, b in combinations(range(len(E)), 2)]))


def plot_loss():
    ep, loss = [], []
    with open(RUN / "loss_history.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); loss.append(float(r["loss"]))
    plt.figure(figsize=(8, 5)); plt.plot(ep, loss, color="darkred", lw=1.4)
    plt.xscale("log")   # la caída ocurre en las primeras ~25 epochs: el eje log la expande
    plt.xlabel("Epoch (escala logarítmica)"); plt.ylabel("Pérdida por margen (promedio)")
    plt.grid(alpha=0.3, which="both"); plt.tight_layout()
    plt.savefig(RUN / "loss_curve.png", dpi=200); plt.close()
    return loss


def infer_groups(all_triples, entities):
    by_rel = defaultdict(list)
    for h, r, t in all_triples:
        by_rel[r].append((h, t))
    cities    = {h for h, _ in by_rel["ubicada_en"]}
    countries = {t for _, t in by_rel["ubicada_en"]} | {h for h, _ in by_rel["tiene_capital"]}
    regions   = ({t for _, t in by_rel["parte_de"]} |
                 {t for _, t in by_rel["situada_en_region"]}) - countries
    monarcas  = ({h for h, _ in by_rel["goberno"]} | {h for h, _ in by_rel["nacio_en"]} |
                 {h for h, _ in by_rel["contemporaneo_de"]})
    batallas  = {h for h, _ in by_rel["tuvo_lugar_en"]}
    groups = defaultdict(list)
    for e in entities:
        if   e in cities:    groups["Ciudades"].append(e)
        elif e in countries: groups["Países"].append(e)
        elif e in regions:   groups["Regiones"].append(e)
        elif e in monarcas:  groups["Monarcas"].append(e)
        elif e in batallas:  groups["Batallas"].append(e)
        else:                groups["Otros"].append(e)
    return groups


def _declutter_y(xy, xwin, gap, iters=400):
    """Separa verticalmente etiquetas próximas (anti-solape determinista, sin
    dependencias externas): empuja en y los pares cuyas x distan menos de xwin
    y cuyas y distan menos de gap. Devuelve las posiciones ajustadas."""
    p = xy.astype(float).copy()
    n = len(p)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                if abs(p[i, 0] - p[j, 0]) >= xwin:
                    continue
                d = p[i, 1] - p[j, 1]
                if abs(d) < gap:
                    shift = (gap - abs(d)) / 2 + 1e-4
                    s = 1.0 if d >= 0 else -1.0
                    p[i, 1] += s * shift
                    p[j, 1] -= s * shift
                    moved = True
        if not moved:
            break
    return p


def plot_pca(ent_emb, id2entity, groups):
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(ent_emb)
    var = pca.explained_variance_ratio_ * 100.0   # % de varianza por componente
    entity2id = {e: i for i, e in id2entity.items()}
    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(13, 10.5))
    for k, (gname, members) in enumerate(groups.items()):
        idx = [entity2id[e] for e in members]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=55, alpha=0.85,
                   color=palette[k % len(palette)], label=gname,
                   edgecolors="black", linewidths=0.5)

    # Con ~150 entidades, etiquetar todas es ilegible: solo países y regiones.
    labelled = sorted(set(groups.get("Países", [])) | set(groups.get("Regiones", [])))
    lab_idx = np.array([entity2id[e] for e in labelled])
    anchors = Z[lab_idx]
    xr = Z[:, 0].max() - Z[:, 0].min()
    yr = Z[:, 1].max() - Z[:, 1].min()
    # margen derecho con holgura: las etiquetas se extienden a la derecha del punto
    ax.set_xlim(Z[:, 0].min() - 0.03 * xr, Z[:, 0].max() + 0.16 * xr)
    placed = _declutter_y(anchors, xwin=0.17 * xr, gap=0.036 * yr)
    for e, (ax0, ay0), (px, py) in zip(labelled, anchors, placed):
        # línea guía tenue si la etiqueta se desplazó apreciablemente del punto
        if abs(py - ay0) > 0.02 * yr:
            ax.plot([ax0, px + 0.012 * xr], [ay0, py], color="0.6", lw=0.4, zorder=1)
        ax.text(px + 0.015 * xr, py, e, fontsize=7, va="center", zorder=3,
                path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")])

    ax.set_xlabel(f"Componente principal 1 ({var[0]:.1f}% de la varianza)")
    ax.set_ylabel(f"Componente principal 2 ({var[1]:.1f}% de la varianza)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(RUN / "embeddings_pca.png", dpi=200); plt.close(fig)
    print(f"  PCA varianza explicada: PC1={var[0]:.1f}%  PC2={var[1]:.1f}%  "
          f"(2D total={var[0] + var[1]:.1f}%)")
    return {"pc1": round(float(var[0]), 1), "pc2": round(float(var[1]), 1),
            "pc1_pc2": round(float(var[0] + var[1]), 1)}


def vector_arithmetic(ent_emb, rel_emb, entity2id, relation2id, id2entity,
                      all_triples, test, top_k=5):
    """Ranking filtrado de colas: ¿está t entre las entidades más próximas a h+r?"""
    filt = defaultdict(set)
    for h, r, t in all_triples:
        filt[(entity2id[h], relation2id[r])].add(entity2id[t])
    rows, ranks = [], []
    for h, r, t in test:
        hi, ri, ti = entity2id[h], relation2id[r], entity2id[t]
        v = ent_emb[hi] + rel_emb[ri]
        d = np.linalg.norm(ent_emb - v, axis=1).copy()
        for tt in filt[(hi, ri)]:
            if tt != ti:
                d[tt] = np.inf
        order = np.argsort(d)
        rank = int((d < d[ti]).sum()) + 1
        ranks.append(rank)
        rows.append({"head": h, "relation": r, "tail_real": t, "rank_real": rank,
                     "top_k": " | ".join(id2entity[i] for i in order[:top_k])})
    with open(RUN / "vector_arithmetic_test.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    ranks = np.array(ranks, dtype=float)
    summary = {"n_test": int(len(ranks)),
               "hits@1": float((ranks <= 1).mean()),
               "hits@3": float((ranks <= 3).mean()),
               "hits@10": float((ranks <= 10).mean())}
    print(f"  aritmética vectorial (test, n={summary['n_test']}): "
          f"Hits@1={summary['hits@1']:.2f}  Hits@3={summary['hits@3']:.2f}")
    return summary


def proposition_analysis(ent_emb, rel_emb, entity2id, relation2id, all_triples, groups):
    rid = relation2id
    results = {}

    # ---- Prop. 2.2.1: normas de relación (simétricas vs. resto) ----
    norms = {rel: float(np.linalg.norm(rel_emb[i])) for rel, i in rid.items()}
    with open(RUN / "relation_norms.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["relation", "norm"])
        w.writerows(sorted(norms.items(), key=lambda x: x[1]))
    results["norms"] = dict(sorted(norms.items(), key=lambda x: x[1]))
    results["min_nonsym_norm"] = min(v for k, v in norms.items() if k not in SYMMETRIC)
    print("  normas ||r|| (menor a mayor):")
    for rel, n in sorted(norms.items(), key=lambda x: x[1]):
        marca = "  <- simétrica" if rel in SYMMETRIC else ""
        print(f"    {rel:20s} {n:6.3f}{marca}")

    # ---- Prop. 2.2.3: gradiente fan-out vs. ratio de colapso ----
    glob = mean_pairwise(ent_emb)
    results["global_mean_dist"] = round(glob, 3)
    by_rel_heads = defaultdict(lambda: defaultdict(list))
    for h, rel, t in all_triples:
        by_rel_heads[rel][h].append(entity2id[t])
    gradient = []
    for rel in ONE_TO_N:
        heads = by_rel_heads[rel]
        fanout = sum(len(ts) for ts in heads.values()) / len(heads)
        dists = [mean_pairwise(ent_emb[ts]) for ts in heads.values() if len(ts) >= 2]
        dists = [d for d in dists if d is not None]
        ratio = float(np.mean(dists)) / glob if dists else None
        gradient.append({"relation": rel, "mean_fanout": round(fanout, 2),
                         "tail_dist": round(float(np.mean(dists)), 3) if dists else None,
                         "ratio_vs_global": round(ratio, 2) if ratio else None})
        print(f"  {rel:14s} fan-out medio={fanout:4.2f}  ratio colapso={ratio:.2f}")
    results["one_to_n_gradient"] = gradient

    # ---- Control de tres niveles (separar colapso 1-a-N de similitud de tipo) ----
    city_ids = np.array([entity2id[e] for e in groups["Ciudades"]])
    within_type = mean_pairwise(ent_emb[city_ids])
    same_head = [mean_pairwise(ent_emb[ts])
                 for ts in by_rel_heads["tiene_ciudad"].values() if len(ts) >= 2]
    same_head = [d for d in same_head if d is not None]
    results["control_cities"] = {
        "global": round(glob, 3),
        "within_type_cities": round(within_type, 3),
        "same_country_cities": round(float(np.mean(same_head)), 3),
    }
    print(f"  control: global={glob:.3f}  ciudades(todas)={within_type:.3f}  "
          f"ciudades(mismo país)={np.mean(same_head):.3f}")

    # ---- Prop. 2.2.5: composiciones aditivas ----
    results["compositions"] = []
    for r3, r1, r2 in COMPOSITIONS:
        v = rel_emb[rid[r1]] + rel_emb[rid[r2]]
        target = rel_emb[rid[r3]]
        c = cos(target, v)
        rel_err = float(np.linalg.norm(target - v) / (np.linalg.norm(target) + 1e-12))
        results["compositions"].append({"rule": f"{r1} + {r2} -> {r3}",
                                        "cos": round(c, 3), "rel_err": round(rel_err, 2)})
        print(f"  cos(r_{r3}, r_{r1}+r_{r2}) = {c:.3f}  (err. rel. {rel_err:.2f})")

    # ---- Inversión: pares inversos ----
    results["inversions"] = []
    for r2, r1 in INVERSES:
        c = cos(rel_emb[rid[r2]], -rel_emb[rid[r1]])
        results["inversions"].append({"pair": f"{r2} = -{r1}", "cos": round(c, 3)})
        print(f"  cos(r_{r2}, -r_{r1}) = {c:.3f}")

    return results


def main():
    ent_emb, rel_emb, entity2id, relation2id, id2entity, all_triples, test = load()
    entities = list(entity2id)
    loss = plot_loss()
    print(f"  loss: inicial={loss[0]:.3f}  epoch100={loss[99]:.3f}  final={loss[-1]:.4f}")
    groups = infer_groups(all_triples, entities)
    print("  grupos:", {k: len(v) for k, v in groups.items()})
    pca_var = plot_pca(ent_emb, id2entity, groups)
    summary = vector_arithmetic(ent_emb, rel_emb, entity2id, relation2id,
                                id2entity, all_triples, test)
    results = proposition_analysis(ent_emb, rel_emb, entity2id, relation2id,
                                   all_triples, groups)
    results["test_ranking"] = summary
    results["pca_variance"] = pca_var
    results["loss"] = {"first": round(loss[0], 3), "epoch100": round(loss[99], 3),
                       "final": round(loss[-1], 4)}
    (RUN / "proposition_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  resultados -> {RUN / 'proposition_results.json'}")


if __name__ == "__main__":
    main()
