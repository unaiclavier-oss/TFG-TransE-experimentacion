"""
Pipeline de entrenamiento de TransE para el Capítulo 3 del TFG.
Pasos ②–⑤ del montaje experimental:

    ②  Mapeo entidad/relación -> ID  (entity2id, relation2id)
    ③  Partición train / val / test  (80/10/10, transductiva)
    ④  Definición del modelo TransE  (dos tablas de embeddings, distancia L_p)
    ⑤  Bucle de entrenamiento        (muestreo negativo + pérdida por margen + Adam)

Entrada : dataset/kg_geografia_historia_v2.csv   (head,relation,tail)
Salidas : transe_run_v2/  (mapeos, splits, embeddings entrenados, histórico de pérdida)

La evaluación cualitativa (curva de pérdida, PCA, aritmética vectorial) es el
paso ⑦ y se implementa aparte. Este script deja el modelo entrenado y todos los
artefactos persistidos para esa fase.

Ejecución:
    python src/train_transe.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# La consola de Windows usa cp1252 por defecto y no imprime símbolos como ②, ρ.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =====================================================================
# 0. Configuración global y reproducibilidad
# =====================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# La semilla cubre la inicialización de embeddings, el barajado de minibatches y
# la corrupción negativa; el vocabulario se construye con sorted(), de modo que
# ningún resultado numérico depende del orden de iteración de set/dict (no hace
# falta fijar PYTHONHASHSEED).

ROOT       = Path(__file__).resolve().parent.parent
DATA_CSV   = ROOT / "dataset" / "kg_geografia_historia_v2.csv"
OUTPUT_DIR = ROOT / "transe_run_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Hiperparámetros (tabla de §3.2; se afinan por validación en el paso ⑥) ----
EMB_DIM    = 30      # d  : dimensión del espacio vectorial (valor seleccionado por validación, §3.2)
MARGIN     = 1.0     # γ  : margen de la pérdida por margen (valor seleccionado por validación, §3.2)
LR         = 0.01    # η  : tasa de aprendizaje (Adam)
EPOCHS     = 1000
BATCH_SIZE = 32
P_NORM     = 2       # norma L2 para la distancia

VAL_RATIO  = 0.10
TEST_RATIO = 0.10

DEVICE = torch.device("cpu")  # fijado a CPU: garantiza resultados deterministas y reproducibles (el RNG de CUDA no es contractual entre GPUs)


# =====================================================================
# ②  Carga del corpus y mapeo entidad/relación -> ID
# =====================================================================
def load_triples(path: Path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(r["head"], r["relation"], r["tail"]) for r in reader]


def build_vocab(triples):
    entities  = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    entity2id   = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    return entity2id, relation2id


def to_indices(triples, entity2id, relation2id):
    return [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]


# =====================================================================
# ③  Partición train / validation / test (transductiva)
# =====================================================================
def split_triples(triples, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED):
    """Garantiza que toda entidad y toda relación de val/test aparezca también
    en train (régimen transductivo: TransE no asigna embedding a entidades no
    vistas, Sección 2.2.1 del TFG)."""
    rng = random.Random(seed)
    idx = list(range(len(triples)))
    rng.shuffle(idx)

    ents_seen, rels_seen = set(), set()
    train_idx, hold_idx  = [], []

    # 1ª pasada: cubrir todas las entidades y relaciones con train
    for i in idx:
        h, r, t = triples[i]
        if h not in ents_seen or t not in ents_seen or r not in rels_seen:
            train_idx.append(i)
            ents_seen.update([h, t])
            rels_seen.add(r)
        else:
            hold_idx.append(i)

    # 2ª pasada: ajustar tamaños al 80 / 10 / 10
    n      = len(triples)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))
    target_train = n - n_val - n_test

    rng.shuffle(hold_idx)
    if len(train_idx) < target_train:
        need = target_train - len(train_idx)
        train_idx.extend(hold_idx[:need])
        hold_idx = hold_idx[need:]

    val_idx  = hold_idx[:n_val]
    test_idx = hold_idx[n_val:n_val + n_test]
    train_idx.extend(hold_idx[n_val + n_test:])

    pick = lambda ids: [triples[i] for i in ids]
    return pick(train_idx), pick(val_idx), pick(test_idx)


# =====================================================================
# ④  Modelo TransE
# =====================================================================
class TransE(nn.Module):
    """Bordes et al. (2013):  f_r(h,t) = -|| h + r - t ||_p.

    - Inicialización uniforme en (-6/√d, 6/√d) (Algoritmo 1, §2.2.2).
    - Relaciones normalizadas a la esfera unidad al inicio.
    - Entidades renormalizadas al principio de cada epoch (evita ||e||->∞).
    """

    def __init__(self, n_entities, n_relations, dim=EMB_DIM, p=P_NORM):
        super().__init__()
        self.p = p
        self.ent = nn.Embedding(n_entities,  dim)
        self.rel = nn.Embedding(n_relations, dim)

        bound = 6.0 / np.sqrt(dim)
        nn.init.uniform_(self.ent.weight, -bound, bound)
        nn.init.uniform_(self.rel.weight, -bound, bound)
        with torch.no_grad():
            self.rel.weight.copy_(F.normalize(self.rel.weight, p=2, dim=1))

    def normalize_entities(self):
        with torch.no_grad():
            self.ent.weight.copy_(F.normalize(self.ent.weight, p=2, dim=1))

    def distance(self, h, r, t):
        return torch.norm(self.ent(h) + self.rel(r) - self.ent(t), p=self.p, dim=1)

    def forward(self, pos, neg):
        return (self.distance(pos[:, 0], pos[:, 1], pos[:, 2]),
                self.distance(neg[:, 0], neg[:, 1], neg[:, 2]))


# =====================================================================
# ⑤  Muestreo negativo, pérdida y bucle de entrenamiento
# =====================================================================
def corrupt(batch, n_entities, train_set, rng):
    """S': reemplaza cabeza o cola (prob. 1/2) por una entidad uniforme,
    evitando colisiones con el conjunto de entrenamiento (filtrado LCWA)."""
    corrupted = batch.clone()
    for i in range(corrupted.size(0)):
        h, r, t = corrupted[i].tolist()
        flip = rng.randint(0, 1)
        new_e = rng.randint(0, n_entities - 1)
        for _ in range(50):  # cota anti-bucle
            trip = (new_e, r, t) if flip == 0 else (h, r, new_e)
            if trip not in train_set:
                break
            new_e = rng.randint(0, n_entities - 1)
        corrupted[i, 0 if flip == 0 else 2] = new_e
    return corrupted


def margin_ranking_loss(d_pos, d_neg, margin):
    """L = mean( [ γ + d(h+r,t) - d(h'+r,t') ]_+ )."""
    return F.relu(margin + d_pos - d_neg).mean()


def train(model, train_idx, n_entities,
          epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, margin=MARGIN,
          verbose=True):
    data      = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    train_set = set(map(tuple, train_idx))
    rng       = random.Random(SEED)
    optim     = torch.optim.Adam(model.parameters(), lr=lr)
    n         = data.size(0)

    history = []
    for ep in range(1, epochs + 1):
        model.normalize_entities()
        perm = torch.randperm(n)
        ep_loss, n_batches = 0.0, 0
        for s in range(0, n, batch_size):
            pos = data[perm[s:s + batch_size]]
            neg = corrupt(pos.cpu(), n_entities, train_set, rng).to(DEVICE)
            d_pos, d_neg = model(pos, neg)
            loss = margin_ranking_loss(d_pos, d_neg, margin)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss   += loss.item()
            n_batches += 1
        history.append(ep_loss / max(n_batches, 1))
        if verbose and (ep == 1 or ep % 100 == 0):
            print(f"  epoch {ep:4d}   loss = {history[-1]:.4f}")
    return history


# =====================================================================
# Persistencia de artefactos
# =====================================================================
def save_split(triples, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "relation", "tail"])
        w.writerows(triples)


def main():
    print("=" * 64)
    print("Entrenamiento de TransE — pasos ②–⑤")
    print("=" * 64)

    # ② carga + vocabulario
    triples = load_triples(DATA_CSV)
    entity2id, relation2id = build_vocab(triples)
    Ne, Nr, Nt = len(entity2id), len(relation2id), len(triples)
    print(f"\n[②] |V|={Ne}  |R|={Nr}  |F|={Nt}  "
          f"rho={Nt/(Ne*Ne*Nr):.6f}")

    # ③ split
    train_t, val_t, test_t = split_triples(triples)
    print(f"[③] split train/val/test = "
          f"{len(train_t)} / {len(val_t)} / {len(test_t)}")
    # comprobación transductiva
    ents_tr = {e for h, _, t in train_t for e in (h, t)}
    rels_tr = {r for _, r, _ in train_t}
    leak_e  = {e for h, _, t in (val_t + test_t) for e in (h, t)} - ents_tr
    leak_r  = {r for _, r, _ in (val_t + test_t)} - rels_tr
    print(f"     entidades/relaciones de val+test ausentes en train: "
          f"{len(leak_e)} / {len(leak_r)}  (deben ser 0)")

    train_idx = to_indices(train_t, entity2id, relation2id)

    # ④ modelo
    model = TransE(Ne, Nr, dim=EMB_DIM, p=P_NORM).to(DEVICE)
    print(f"[④] TransE(d={EMB_DIM}, p={P_NORM})  "
          f"#parámetros = {(Ne + Nr) * EMB_DIM}")

    # ⑤ entrenamiento
    print(f"[⑤] entrenando  (γ={MARGIN}, η={LR}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}, optim=Adam)")
    history = train(model, train_idx, Ne)
    print(f"     loss inicial = {history[0]:.4f}   loss final = {history[-1]:.4f}")

    # --- persistencia ---
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
        w = csv.writer(f)
        w.writerow(["epoch", "loss"])
        w.writerows(enumerate(history, start=1))

    print(f"\nArtefactos guardados en: {OUTPUT_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()
