"""
Paso ⑦ del montaje experimental: evaluación cualitativa para §3.4.

Carga el modelo entrenado (transe_run/) y produce:

  1. Curva de pérdida                       -> loss_curve.png
  2. Reducción de dimensionalidad t-SNE/PCA -> embeddings_tsne.png / _pca.png
  3. Aritmética vectorial sobre test        -> vector_arithmetic_test.csv
  4. Confrontación con las Proposiciones 1-3 (+ inversión):
       · Prop 1 (simetría)     : norma de cada vector de relación
       · Prop 2 (1-a-N)        : colapso de las colas que comparten cabeza
       · Prop 3 (composición)  : r_nacionalidad ≈ r_nacio_en + r_ubicada_en
       · Inversión             : r_capital_de ≈ - r_tiene_capital

Ejecución:
    python src/evaluate_transe.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RUN  = ROOT / "transe_run"
DATA = ROOT / "dataset" / "kg_geografia_historia.csv"


# =====================================================================
# Carga
# =====================================================================
def load():
    ent_emb = np.load(RUN / "entity_embeddings.npy")
    rel_emb = np.load(RUN / "relation_embeddings.npy")
    entity2id   = json.loads((RUN / "entity2id.json").read_text(encoding="utf-8"))
    relation2id = json.loads((RUN / "relation2id.json").read_text(encoding="utf-8"))
    id2entity = {i: e for e, i in entity2id.items()}

    def read_csv(path):
        with open(path, encoding="utf-8") as f:
            return [(r["head"], r["relation"], r["tail"]) for r in csv.DictReader(f)]

    all_triples = read_csv(DATA)
    test  = read_csv(RUN / "test.csv")
    return ent_emb, rel_emb, entity2id, relation2id, id2entity, all_triples, test


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# =====================================================================
# 1. Curva de pérdida
# =====================================================================
def plot_loss():
    ep, loss = [], []
    with open(RUN / "loss_history.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); loss.append(float(r["loss"]))
    plt.figure(figsize=(8, 5))
    plt.plot(ep, loss, color="darkred", lw=1.4)
    plt.xlabel("Época"); plt.ylabel("Margin Ranking Loss promedio")
    plt.title("Convergencia del entrenamiento de TransE")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RUN / "loss_curve.png", dpi=200); plt.close()
    print(f"  loss_curve.png  (loss final = {loss[-1]:.4f})")


# =====================================================================
# 2. t-SNE / PCA con grupos semánticos (inferidos de las relaciones)
# =====================================================================
def infer_groups(all_triples, entities):
    by_rel = defaultdict(list)
    for h, r, t in all_triples:
        by_rel[r].append((h, t))
    cities    = {h for h, _ in by_rel["ubicada_en"]}
    countries = {t for _, t in by_rel["ubicada_en"]} | {h for h, _ in by_rel["tiene_capital"]}
    regions   = {t for _, t in by_rel["parte_de"]} - countries
    monarcas  = {h for h, _ in by_rel["goberno"]} | {h for h, _ in by_rel["nacio_en"]}
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


def plot_2d(ent_emb, id2entity, groups, method, path):
    if method == "tsne":
        Z = TSNE(n_components=2, perplexity=12, random_state=SEED,
                 init="pca").fit_transform(ent_emb)
        title = "Embeddings de entidades — t-SNE"
    else:
        Z = PCA(n_components=2, random_state=SEED).fit_transform(ent_emb)
        title = "Embeddings de entidades — PCA"

    entity2id = {e: i for i, e in id2entity.items()}
    palette = plt.cm.tab10.colors
    plt.figure(figsize=(12, 9))
    for k, (gname, members) in enumerate(groups.items()):
        idx = [entity2id[e] for e in members]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=60, alpha=0.85,
                    color=palette[k % len(palette)], label=gname,
                    edgecolors="black", linewidths=0.5)
    for i, name in id2entity.items():
        plt.text(Z[i, 0] + 0.02, Z[i, 1] + 0.02, name, fontsize=7)
    plt.legend(loc="best", fontsize=9); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    print(f"  {path.name}")


# =====================================================================
# 3. Aritmética vectorial sobre test (ranking filtrado)
# =====================================================================
def vector_arithmetic(ent_emb, rel_emb, entity2id, relation2id, id2entity,
                      all_triples, test, top_k=5):
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
        rows.append({"head": h, "relation": r, "tail_real": t,
                     "rank_real": rank,
                     "top_k": " | ".join(id2entity[i] for i in order[:top_k])})

    with open(RUN / "vector_arithmetic_test.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    ranks = np.array(ranks, dtype=float)
    print(f"  vector_arithmetic_test.csv  "
          f"(Hits@1={np.mean(ranks <= 1):.2f}, Hits@3={np.mean(ranks <= 3):.2f}, "
          f"MRR={np.mean(1/ranks):.3f}, n={len(ranks)})")
    return rows


# =====================================================================
# 4. Confrontación con las Proposiciones
# =====================================================================
def proposition_analysis(ent_emb, rel_emb, entity2id, relation2id, all_triples):
    r = relation2id
    rn = {rel: float(np.linalg.norm(rel_emb[i])) for rel, i in relation2id.items()}

    # ---- Prop 1: simetría -> ||r|| ~ 0 ----
    print("\n  [Prop 1] Norma de los vectores de relación (simétrica -> 0):")
    for rel, n in sorted(rn.items(), key=lambda x: x[1]):
        flag = "  <- SIMÉTRICA" if rel == "limitrofe_con" else ""
        print(f"      {rel:16s} ||r|| = {n:5.3f}{flag}")
    with open(RUN / "relation_norms.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["relation", "norm"])
        w.writerows(sorted(rn.items(), key=lambda x: x[1]))

    # ---- Prop 2: 1-a-N -> colapso de colas ----
    print("\n  [Prop 2] Colapso de colas en relaciones 1-a-N:")
    by_rel_heads = defaultdict(lambda: defaultdict(list))
    for h, rel, t in all_triples:
        by_rel_heads[rel][h].append(entity2id[t])

    def mean_pairwise(ids):
        if len(ids) < 2:
            return None
        E = ent_emb[ids]
        D = [np.linalg.norm(E[a] - E[b])
             for a in range(len(E)) for b in range(a + 1, len(E))]
        return float(np.mean(D))

    # distancia media global entre entidades (referencia)
    allE = ent_emb
    glob = float(np.mean([np.linalg.norm(allE[a] - allE[b])
                          for a in range(len(allE)) for b in range(a + 1, len(allE))]))
    print(f"      distancia media global entre entidades = {glob:.3f}")
    for rel in ("tiene_ciudad", "goberno"):
        co = [mean_pairwise(ts) for ts in by_rel_heads[rel].values() if len(ts) >= 2]
        co = [c for c in co if c is not None]
        if co:
            print(f"      {rel:14s}: dist. media entre colas que comparten cabeza "
                  f"= {np.mean(co):.3f}  (ratio vs global = {np.mean(co)/glob:.2f})")

    # ---- Prop 3: composición aditiva ----
    print("\n  [Prop 3] Composición  r_nacionalidad ≈ r_nacio_en + r_ubicada_en:")
    comp = rel_emb[r["nacio_en"]] + rel_emb[r["ubicada_en"]]
    target = rel_emb[r["nacionalidad"]]
    err = np.linalg.norm(target - comp) / (np.linalg.norm(target) + 1e-12)
    print(f"      cos(r_nac, r_nacio+r_ubic) = {cos(target, comp):.3f}")
    print(f"      error relativo ||r_nac-(r_nacio+r_ubic)||/||r_nac|| = {err:.3f}")

    # ---- Inversión ----
    print("\n  [Inversión]  r_capital_de ≈ - r_tiene_capital:")
    inv = cos(rel_emb[r["capital_de"]], -rel_emb[r["tiene_capital"]])
    print(f"      cos(r_capital_de, -r_tiene_capital) = {inv:.3f}")


# =====================================================================
def main():
    print("=" * 64)
    print("Evaluación cualitativa de TransE — paso ⑦ (§3.4)")
    print("=" * 64)

    ent_emb, rel_emb, entity2id, relation2id, id2entity, all_triples, test = load()
    entities = list(entity2id)

    print("\n[1] Curva de pérdida")
    plot_loss()

    print("\n[2] Reducción de dimensionalidad")
    groups = infer_groups(all_triples, entities)
    plot_2d(ent_emb, id2entity, groups, "tsne", RUN / "embeddings_tsne.png")
    plot_2d(ent_emb, id2entity, groups, "pca",  RUN / "embeddings_pca.png")

    print("\n[3] Aritmética vectorial sobre test")
    vector_arithmetic(ent_emb, rel_emb, entity2id, relation2id, id2entity,
                      all_triples, test)

    print("\n[4] Confrontación con las Proposiciones")
    proposition_analysis(ent_emb, rel_emb, entity2id, relation2id, all_triples)

    print("\n" + "=" * 64)
    print(f"Figuras y tablas guardadas en: {RUN}")
    print("=" * 64)


if __name__ == "__main__":
    main()
