# Inferencia de Grafos de Conocimiento — Experimento con TransE

Código y datos del experimento de mi Trabajo de Fin de Grado en Matemáticas
(*Inferencia de Grafos de Conocimiento a partir de Texto*).

El objetivo del experimento no es competir en un benchmark, sino **validar
empíricamente tres proposiciones teóricas** sobre las limitaciones del modelo
TransE, demostradas en la memoria:

1. **Relaciones simétricas** → el vector de relación colapsa al vector nulo (r ≈ 0).
2. **Relaciones 1-a-N** → los embeddings de las colas colapsan entre sí.
3. **Relaciones compuestas** → la composición se traduce en suma de vectores
   (r_comp ≈ r_1 + r_2).

## Contenido

- `kg_geografia_historia.csv` — grafo de conocimiento sintético y
  procesado (153 entidades, 15 relaciones, 613 tripletes) sobre geografía e
  historia europea, diseñado a propósito para contener relaciones simétricas,
  1-a-N, compuestas e inversas.
- `src/generate_dataset.py` — generación y procesado del corpus.
- `src/train_transe.py` — implementación de TransE en PyTorch (margin ranking
  loss, muestreo negativo por corrupción) y entrenamiento.
- `src/tune_transe.py` — búsqueda en rejilla de hiperparámetros sobre validación.
- `src/evaluate_transe.py` — evaluación (Hits@k con protocolo filtrado) y
  contraste de las tres proposiciones.
- `src/structural_analysis.py` — análisis estructural del grafo.

