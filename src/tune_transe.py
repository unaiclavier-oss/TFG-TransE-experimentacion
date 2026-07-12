"""
Paso ⑥ del montaje experimental: validación / selección de hiperparámetros.

Realiza una búsqueda en rejilla, entrena un TransE por combinación, y la evalúa
con el **Hits@1 filtrado** sobre el conjunto de validación (criterio de §3.2 del
TFG; empates resueltos por Hits@3 y después MRR). Elige la mejor configuración,
reentrena con ella y sobrescribe los artefactos de transe_run_v2/.

Importante: las métricas se usan aquí SOLO como criterio interno de selección de
hiperparámetros. NO son el resultado central de §3.3 (cualitativo).

Ejecución:
    python src/tune_transe.py
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from train_transe import (OUTPUT_DIR, DATA_CSV, SEED, P_NORM, BATCH_SIZE, DEVICE,
                          load_triples, build_vocab, to_indices, split_triples,
                          TransE, train, save_split)


# ---- Rejilla de búsqueda (pequeña: grafo de ~150 entidades, CPU) ----
GRID = {
    "dim":    [30, 50],
    "margin": [1.0, 2.0, 4.0],
    "lr":     [0.01],
    "epochs": [1000],
}


def make_filters(idx_triples):
    """Conjuntos de verdad para el ranking filtrado (train+val+test)."""
    ft, fh = defaultdict(set), defaultdict(set)
    for h, r, t in idx_triples:
        ft[(h, r)].add(t)
        fh[(r, t)].add(h)
    return ft, fh


def rank_eval(model, eval_idx, ft, fh):
    """MRR / Hits@k filtrados, promediando predicción de cabeza y de cola."""
    ent = model.ent.weight.detach()
    rel = model.rel.weight.detach()
    ranks = []
    for h, r, t in eval_idx:
        # --- predicción de cola: ||h + r - t'|| sobre todo t' ---
        d = torch.norm((ent[h] + rel[r]).unsqueeze(0) - ent, p=model.p, dim=1).clone()
        for tt in ft[(h, r)]:
            if tt != t:
                d[tt] = float("inf")
        ranks.append(int((d < d[t]).sum().item()) + 1)
        # --- predicción de cabeza: ||h' + r - t|| sobre todo h' ---
        d2 = torch.norm(ent + (rel[r] - ent[t]).unsqueeze(0), p=model.p, dim=1).clone()
        for hh in fh[(r, t)]:
            if hh != h:
                d2[hh] = float("inf")
        ranks.append(int((d2 < d2[h]).sum().item()) + 1)
    ranks = np.array(ranks, dtype=float)
    return {
        "MRR":     float((1.0 / ranks).mean()),
        "Hits@1":  float((ranks <= 1).mean()),
        "Hits@3":  float((ranks <= 3).mean()),
        "Hits@10": float((ranks <= 10).mean()),
        "MR":      float(ranks.mean()),
    }


def train_one(train_idx, Ne, Nr, dim, margin, lr, epochs):
    """Entrena una configuración de forma reproducible (misma semilla de init)."""
    torch.manual_seed(SEED)
    model = TransE(Ne, Nr, dim=dim, p=P_NORM).to(DEVICE)
    history = train(model, train_idx, Ne, epochs=epochs, batch_size=BATCH_SIZE,
                    lr=lr, margin=margin, verbose=False)
    return model, history


def main():
    print("=" * 64)
    print("Validación de hiperparámetros de TransE — paso ⑥")
    print("=" * 64)

    triples = load_triples(DATA_CSV)
    entity2id, relation2id = build_vocab(triples)
    Ne, Nr = len(entity2id), len(relation2id)

    train_t, val_t, test_t = split_triples(triples)
    all_idx   = to_indices(triples, entity2id, relation2id)
    train_idx = to_indices(train_t, entity2id, relation2id)
    val_idx   = to_indices(val_t,   entity2id, relation2id)
    ft, fh    = make_filters(all_idx)

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"\n{len(combos)} configuraciones · selección por Hits@1(val) filtrado\n")
    print(f"  {'dim':>3} {'margin':>6} {'lr':>5} {'epochs':>6} "
          f"{'MRR':>6} {'H@1':>5} {'H@3':>5} {'H@10':>5}")

    results, best = [], None
    for c in combos:
        model, _ = train_one(train_idx, Ne, Nr,
                             c["dim"], c["margin"], c["lr"], c["epochs"])
        m = rank_eval(model, val_idx, ft, fh)
        row = {**c, **m}
        results.append(row)
        print(f"  {c['dim']:>3} {c['margin']:>6.1f} {c['lr']:>5.3f} "
              f"{c['epochs']:>6} {m['MRR']:>6.3f} {m['Hits@1']:>5.2f} "
              f"{m['Hits@3']:>5.2f} {m['Hits@10']:>5.2f}")
        key = (m["Hits@1"], m["Hits@3"], m["MRR"])
        if best is None or key > (best["Hits@1"], best["Hits@3"], best["MRR"]):
            best = row

    print(f"\nMejor configuración (Hits@1 val = {best['Hits@1']:.3f}, "
          f"Hits@3 val = {best['Hits@3']:.3f}):")
    print(f"  dim={best['dim']}  margin={best['margin']}  "
          f"lr={best['lr']}  epochs={best['epochs']}")

    # ---- Reentrenar la mejor configuración y sobrescribir artefactos ----
    print("\nReentrenando la mejor configuración y guardando artefactos...")
    model, history = train_one(train_idx, Ne, Nr,
                              best["dim"], best["margin"], best["lr"], best["epochs"])

    (OUTPUT_DIR / "entity2id.json").write_text(
        json.dumps(entity2id, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "relation2id.json").write_text(
        json.dumps(relation2id, ensure_ascii=False, indent=2), encoding="utf-8")
    save_split(train_t, OUTPUT_DIR / "train.csv")
    save_split(val_t,   OUTPUT_DIR / "valid.csv")
    save_split(test_t,  OUTPUT_DIR / "test.csv")
    np.save(OUTPUT_DIR / "entity_embeddings.npy",
            model.ent.weight.detach().cpu().numpy())
    np.save(OUTPUT_DIR / "relation_embeddings.npy",
            model.rel.weight.detach().cpu().numpy())
    torch.save(model.state_dict(), OUTPUT_DIR / "transe_model.pt")
    with open(OUTPUT_DIR / "loss_history.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "loss"]); w.writerows(enumerate(history, 1))

    chosen = {k: best[k] for k in ("dim", "margin", "lr", "epochs")}
    chosen["p_norm"], chosen["batch_size"] = P_NORM, BATCH_SIZE
    (OUTPUT_DIR / "chosen_hyperparams.json").write_text(
        json.dumps(chosen, indent=2), encoding="utf-8")
    with open(OUTPUT_DIR / "tuning_results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    print(f"loss inicial = {history[0]:.4f}   loss final = {history[-1]:.4f}")
    print(f"\nArtefactos (mejor modelo) en: {OUTPUT_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()
