"""
Generador del corpus sintético v2 (geografía e historia europea) para la Sección 3
del TFG. Estrategia de curación en dos niveles:

  1. DATOS SEMILLA curados a mano (este fichero): países con capital, ciudades y
     región; fronteras; monarcas con años de reinado, país gobernado y ciudad de
     nacimiento; batallas con participantes y país.
  2. CIERRES AUTOMÁTICOS por construcción, que garantizan los patrones que las
     Proposiciones 2.2.1 / 2.2.3 / 2.2.5 del TFG necesitan sin posibilidad de error manual:
       - simétrico   : limitrofe_con (ambos sentidos), contemporaneo_de (calculada
                       a partir del solapamiento de reinados)
       - inverso     : tiene_capital/capital_de, tiene_ciudad/ubicada_en,
                       parte_de (nivel país)/incluye_pais
       - composición : nacio_en ∘ ubicada_en  => nacionalidad
                       ubicada_en ∘ parte_de  => situada_en_region

Tras generar, el script VERIFICA todos los cierres y emite un informe
(dataset/generation_report_v2.md). Salida: dataset/kg_geografia_historia_v2.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "dataset" / "kg_geografia_historia_v2.csv"
OUT_REPORT = ROOT / "dataset" / "generation_report_v2.md"

# =====================================================================
# 1. DATOS SEMILLA (curados a mano; única fuente de verdad)
# =====================================================================

# país -> (capital, [ciudades incluida la capital], región)
# región == "Europa" significa adscripción directa al continente (sin subregión
# inequívoca en el corpus).
COUNTRIES = {
    "Espana":       ("Madrid",    ["Madrid", "Barcelona", "Sevilla", "Valencia", "Bilbao", "Valladolid"], "Peninsula_Iberica"),
    "Portugal":     ("Lisboa",    ["Lisboa", "Oporto", "Braga"], "Peninsula_Iberica"),
    "Francia":      ("Paris",     ["Paris", "Lyon", "Marsella", "Burdeos", "Toulouse", "Ajaccio", "Versalles"], "Europa"),
    "Italia":       ("Roma",      ["Roma", "Milan", "Napoles", "Venecia", "Florencia", "Turin"], "Europa"),
    "Alemania":     ("Berlin",    ["Berlin", "Munich", "Hamburgo", "Colonia", "Frankfurt"], "Europa_Central"),
    "Reino_Unido":  ("Londres",   ["Londres", "Manchester", "Edimburgo", "Glasgow", "Birmingham"], "Islas_Britanicas"),
    "Irlanda":      ("Dublin",    ["Dublin", "Cork"], "Islas_Britanicas"),
    "Belgica":      ("Bruselas",  ["Bruselas", "Amberes", "Gante"], "Benelux"),
    "Paises_Bajos": ("Amsterdam", ["Amsterdam", "Roterdam", "La_Haya", "Utrecht"], "Benelux"),
    "Luxemburgo":   ("Ciudad_de_Luxemburgo", ["Ciudad_de_Luxemburgo"], "Benelux"),
    "Suiza":        ("Berna",     ["Berna", "Zurich", "Ginebra"], "Europa_Central"),
    "Austria":      ("Viena",     ["Viena", "Salzburgo", "Graz"], "Europa_Central"),
    "Polonia":      ("Varsovia",  ["Varsovia", "Cracovia", "Gdansk", "Breslavia"], "Europa_Central"),
    "Chequia":      ("Praga",     ["Praga", "Brno"], "Europa_Central"),
    "Eslovaquia":   ("Bratislava", ["Bratislava", "Kosice"], "Europa_Central"),
    "Hungria":      ("Budapest",  ["Budapest", "Debrecen"], "Europa_Central"),
    "Rumania":      ("Bucarest",  ["Bucarest", "Cluj_Napoca"], "Europa"),
    "Bulgaria":     ("Sofia",     ["Sofia", "Plovdiv"], "Balcanes"),
    "Grecia":       ("Atenas",    ["Atenas", "Salonica"], "Balcanes"),
    "Serbia":       ("Belgrado",  ["Belgrado", "Novi_Sad"], "Balcanes"),
    "Croacia":      ("Zagreb",    ["Zagreb", "Split"], "Balcanes"),
    "Eslovenia":    ("Liubliana", ["Liubliana", "Maribor"], "Europa"),
    "Dinamarca":    ("Copenhague", ["Copenhague", "Aarhus"], "Escandinavia"),
    "Suecia":       ("Estocolmo", ["Estocolmo", "Gotemburgo", "Malmo"], "Escandinavia"),
    "Noruega":      ("Oslo",      ["Oslo", "Bergen"], "Escandinavia"),
    "Finlandia":    ("Helsinki",  ["Helsinki", "Tampere"], "Europa"),
    "Rusia":        ("Moscu",     ["Moscu", "San_Petersburgo", "Kazan"], "Europa"),
    "Ucrania":      ("Kiev",      ["Kiev", "Leopolis", "Odesa"], "Europa"),
    "Lituania":     ("Vilna",     ["Vilna", "Kaunas"], "Paises_Balticos"),
    "Letonia":      ("Riga",      ["Riga"], "Paises_Balticos"),
    "Estonia":      ("Tallin",    ["Tallin"], "Paises_Balticos"),
}

SUBREGIONS = ["Peninsula_Iberica", "Islas_Britanicas", "Benelux", "Europa_Central",
              "Balcanes", "Escandinavia", "Paises_Balticos"]

# Pares fronterizos (cada par UNA vez; el cierre simétrico añade el sentido inverso)
BORDERS = [
    ("Espana", "Portugal"), ("Espana", "Francia"),
    ("Francia", "Belgica"), ("Francia", "Luxemburgo"), ("Francia", "Alemania"),
    ("Francia", "Suiza"), ("Francia", "Italia"),
    ("Belgica", "Paises_Bajos"), ("Belgica", "Luxemburgo"), ("Belgica", "Alemania"),
    ("Paises_Bajos", "Alemania"), ("Luxemburgo", "Alemania"),
    ("Alemania", "Suiza"), ("Alemania", "Austria"), ("Alemania", "Chequia"),
    ("Alemania", "Polonia"), ("Alemania", "Dinamarca"),
    ("Suiza", "Italia"), ("Suiza", "Austria"),
    ("Austria", "Italia"), ("Austria", "Chequia"), ("Austria", "Eslovaquia"),
    ("Austria", "Hungria"), ("Austria", "Eslovenia"),
    ("Italia", "Eslovenia"),
    ("Chequia", "Polonia"), ("Chequia", "Eslovaquia"),
    ("Eslovaquia", "Polonia"), ("Eslovaquia", "Hungria"), ("Eslovaquia", "Ucrania"),
    ("Hungria", "Rumania"), ("Hungria", "Serbia"), ("Hungria", "Croacia"),
    ("Hungria", "Eslovenia"), ("Hungria", "Ucrania"),
    ("Polonia", "Lituania"), ("Polonia", "Ucrania"), ("Polonia", "Rusia"),
    ("Lituania", "Letonia"), ("Lituania", "Rusia"),
    ("Letonia", "Estonia"), ("Letonia", "Rusia"),
    ("Estonia", "Rusia"),
    ("Rusia", "Ucrania"), ("Rusia", "Finlandia"), ("Rusia", "Noruega"),
    ("Finlandia", "Suecia"), ("Finlandia", "Noruega"),
    ("Suecia", "Noruega"),
    ("Rumania", "Serbia"), ("Rumania", "Bulgaria"), ("Rumania", "Ucrania"),
    ("Bulgaria", "Grecia"), ("Bulgaria", "Serbia"),
    ("Serbia", "Croacia"), ("Croacia", "Eslovenia"),
    ("Reino_Unido", "Irlanda"),
]

# monarca -> (año inicio reinado, año fin, [países gobernados], ciudad de nacimiento o None)
# La ciudad de nacimiento es None cuando no pertenece al inventario del corpus
# (p. ej. Carlos I nació en Gante bajo otra entidad política; Catalina la Grande
# en Stettin): en esos casos NO se emite nacio_en y, por la regla composicional,
# tampoco nacionalidad.
MONARCHS = {
    "Carlos_I_de_Espana":     (1516, 1556, ["Espana"], None),
    "Felipe_II":              (1556, 1598, ["Espana", "Portugal"], "Valladolid"),
    "Carlos_II_de_Espana":    (1665, 1700, ["Espana"], "Madrid"),
    "Enrique_VIII":           (1509, 1547, [], "Londres"),
    "Isabel_I_de_Inglaterra": (1558, 1603, [], "Londres"),
    "Victoria":               (1837, 1901, ["Reino_Unido"], "Londres"),
    "Luis_XIV":               (1643, 1715, ["Francia"], None),
    "Luis_XVI":               (1774, 1792, ["Francia"], "Versalles"),
    "Napoleon":               (1804, 1815, ["Francia"], "Ajaccio"),
    "Pedro_el_Grande":        (1682, 1725, ["Rusia"], "Moscu"),
    "Catalina_la_Grande":     (1762, 1796, ["Rusia"], None),
    "Alejandro_I_de_Rusia":   (1801, 1825, ["Rusia"], "San_Petersburgo"),
    "Maria_Teresa_de_Austria": (1740, 1780, ["Austria", "Hungria"], "Viena"),
    "Gustavo_II_Adolfo":      (1611, 1632, ["Suecia"], "Estocolmo"),
    "Juan_III_Sobieski":      (1674, 1696, ["Polonia", "Lituania"], None),
    "Guillermo_I_de_Alemania": (1871, 1888, ["Alemania"], "Berlin"),
}

# sucedio_a(x, y): x ocupó con posterioridad el trono del mismo país que y
# (no necesariamente de forma inmediata; véase README).
SUCCESSIONS = [
    ("Felipe_II", "Carlos_I_de_Espana"),
    ("Carlos_II_de_Espana", "Felipe_II"),
    ("Luis_XVI", "Luis_XIV"),
    ("Napoleon", "Luis_XVI"),
    ("Catalina_la_Grande", "Pedro_el_Grande"),
    ("Alejandro_I_de_Rusia", "Catalina_la_Grande"),
]

# batalla -> ([participantes: monarcas presentes o países beligerantes], país donde tuvo lugar
#             según las fronteras actuales)
BATTLES = {
    "Batalla_de_Waterloo":   (["Napoleon", "Reino_Unido"], "Belgica"),
    "Batalla_de_Trafalgar":  (["Francia", "Reino_Unido", "Espana"], "Espana"),
    "Batalla_de_Austerlitz": (["Napoleon", "Alejandro_I_de_Rusia", "Austria"], "Chequia"),
    "Batalla_de_Borodino":   (["Napoleon", "Rusia"], "Rusia"),
    "Batalla_de_Leipzig":    (["Napoleon", "Rusia", "Austria"], "Alemania"),
    "Batalla_de_Lepanto":    (["Espana"], "Grecia"),
    "Batalla_de_Lutzen":     (["Gustavo_II_Adolfo"], "Alemania"),
    "Batalla_de_Viena":      (["Juan_III_Sobieski", "Austria"], "Austria"),
    "Batalla_de_Poltava":    (["Pedro_el_Grande", "Suecia"], "Ucrania"),
}


# =====================================================================
# 2. GENERACIÓN CON CIERRES AUTOMÁTICOS
# =====================================================================
def generate():
    triples = []
    add = lambda h, r, t: triples.append((h, r, t))

    # --- geografía: ciudades, capitales, regiones (cierres inversos) ---
    for country, (capital, cities, region) in COUNTRIES.items():
        assert capital in cities, f"capital {capital} no está en las ciudades de {country}"
        for city in cities:
            add(city, "ubicada_en", country)        # N-a-1
            add(country, "tiene_ciudad", city)      # 1-a-N  (inversa de la anterior)
        add(country, "tiene_capital", capital)      # funcional
        add(capital, "capital_de", country)         # inversa
        add(country, "parte_de", region)            # país -> región
        add(region, "incluye_pais", country)        # inversa (1-a-N)
    for sub in SUBREGIONS:
        add(sub, "parte_de", "Europa")              # jerarquía región -> continente

    # --- cierre simétrico: fronteras ---
    for a, b in BORDERS:
        add(a, "limitrofe_con", b)
        add(b, "limitrofe_con", a)

    # --- cierre simétrico calculado: reinados solapados ---
    names = list(MONARCHS)
    for i, m1 in enumerate(names):
        for m2 in names[i + 1:]:
            s1, e1 = MONARCHS[m1][0], MONARCHS[m1][1]
            s2, e2 = MONARCHS[m2][0], MONARCHS[m2][1]
            if max(s1, s2) < min(e1, e2):           # solapamiento estricto
                add(m1, "contemporaneo_de", m2)
                add(m2, "contemporaneo_de", m1)

    # --- monarcas: gobierno, nacimiento y composición 1 ---
    for monarch, (_, _, governed, birth_city) in MONARCHS.items():
        for country in governed:
            add(monarch, "goberno", country)        # 1-a-N de fan-out bajo
        if birth_city is not None:
            add(monarch, "nacio_en", birth_city)
            birth_country = next(c for c, (_, cities, _) in
                                 ((c, (cap, cities, reg)) for c, (cap, cities, reg) in COUNTRIES.items())
                                 if birth_city in COUNTRIES[c][1])
            add(monarch, "nacionalidad", birth_country)   # cierre composición 1

    # --- composición 2: ubicada_en ∘ parte_de => situada_en_region ---
    for country, (_, cities, region) in COUNTRIES.items():
        for city in cities:
            add(city, "situada_en_region", region)

    # --- sucesiones y batallas ---
    for x, y in SUCCESSIONS:
        add(x, "sucedio_a", y)
    for battle, (participants, place) in BATTLES.items():
        for p in participants:
            add(p, "participo_en", battle)
        add(battle, "tuvo_lugar_en", place)

    return triples


# =====================================================================
# 3. VERIFICACIÓN DE CIERRES (la parte automatizada de la curación)
# =====================================================================
def verify(triples):
    errors = []
    tset = set(triples)
    if len(tset) != len(triples):
        dup = [t for t, c in Counter(triples).items() if c > 1]
        errors.append(f"tripletas duplicadas: {dup}")

    by_rel = defaultdict(set)
    for h, r, t in triples:
        by_rel[r].add((h, t))

    # simetría exacta
    for r in ("limitrofe_con", "contemporaneo_de"):
        for h, t in by_rel[r]:
            if (t, h) not in by_rel[r]:
                errors.append(f"simetría rota en {r}: falta ({t},{h})")

    # inversas exactas
    for r1, r2 in (("tiene_capital", "capital_de"),
                   ("tiene_ciudad", "ubicada_en")):
        if {(t, h) for h, t in by_rel[r1]} != by_rel[r2]:
            errors.append(f"inversión rota: {r1} / {r2}")
    pais_parte = {(h, t) for h, t in by_rel["parte_de"] if h in COUNTRIES}
    if {(t, h) for h, t in pais_parte} != by_rel["incluye_pais"]:
        errors.append("inversión rota: parte_de (nivel país) / incluye_pais")

    # composición 1: nacio_en ∘ ubicada_en => nacionalidad (cierre exacto)
    comp1 = {(p, country)
             for p, city in by_rel["nacio_en"]
             for c2, country in by_rel["ubicada_en"] if c2 == city}
    if comp1 != by_rel["nacionalidad"]:
        errors.append("cierre composicional 1 (nacionalidad) no exacto")

    # composición 2: ubicada_en ∘ parte_de => situada_en_region (cierre exacto)
    comp2 = {(city, region)
             for city, country in by_rel["ubicada_en"]
             for c2, region in pais_parte if c2 == country}
    if comp2 != by_rel["situada_en_region"]:
        errors.append("cierre composicional 2 (situada_en_region) no exacto")

    # antisimetría de sucedio_a
    for h, t in by_rel["sucedio_a"]:
        if (t, h) in by_rel["sucedio_a"]:
            errors.append(f"sucedio_a no antisimétrica: ({h},{t})")

    return errors


def report(triples):
    entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    rel_counts = Counter(r for _, r, _ in triples)

    cities = {c for _, (_, cs, _) in COUNTRIES.items() for c in cs}
    types = {"Países": set(COUNTRIES), "Ciudades": cities,
             "Regiones": set(SUBREGIONS) | {"Europa"},
             "Monarcas": set(MONARCHS), "Batallas": set(BATTLES)}

    # fan-out medio de las relaciones 1-a-N (nº medio de colas por cabeza)
    fanouts = {}
    for r in ("goberno", "tiene_ciudad", "incluye_pais"):
        heads = defaultdict(int)
        for h, rel, t in triples:
            if rel == r:
                heads[h] += 1
        fanouts[r] = sum(heads.values()) / len(heads)

    ne, nr, nt = len(entities), len(relations), len(triples)
    lines = [
        "# Informe de generación del corpus v2", "",
        f"- Entidades: **{ne}**  |  Relaciones: **{nr}**  |  Tripletas: **{nt}**",
        f"- Densidad multirrelacional: {nt}/({ne}^2*{nr}) = {nt/(ne*ne*nr):.2e}", "",
        "## Tipos de entidad",
        *[f"- {k}: {len(v)}" for k, v in types.items()], "",
        "## Distribución de relaciones",
        *[f"- {r}: {c}" for r, c in rel_counts.most_common()], "",
        "## Fan-out medio (relaciones 1-a-N, Prop. 2.2.3)",
        *[f"- {r}: {f:.2f}" for r, f in sorted(fanouts.items(), key=lambda x: x[1])], "",
        "## Verificación de cierres",
    ]
    errors = verify(triples)
    lines += [f"- ERROR: {e}" for e in errors] if errors else ["- Todos los cierres verificados sin errores."]
    return "\n".join(lines) + "\n", errors


def main():
    triples = generate()
    rep, errors = report(triples)
    OUT_REPORT.write_text(rep, encoding="utf-8")
    print(rep)
    if errors:
        sys.exit("Generación abortada: hay errores de verificación.")
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "relation", "tail"])
        w.writerows(triples)
    print(f"OK -> {OUT_CSV}")


if __name__ == "__main__":
    main()
