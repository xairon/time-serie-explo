"""Garde-fou anti-divergence cross-repo du calcul d'indice IPS/SSFI.

Le mapping z -> classe est dupliqué entre ce repo (dashboard/utils/drought.py)
et le repo hubeau (src/hubeau_pipeline/ml/indices.py). Cette table golden fige
les seuils (±0.84/±1.28/±1.75, borne basse inclusive). Le MÊME bloc existe dans
hubeau (tests/test_indices.py::GOLDEN_Z_TO_CLASS). Si l'une des deux
implémentations dérive, son CI casse -> divergence silencieuse impossible.
NE PAS MODIFIER sans répliquer dans l'autre repo.
"""
from dashboard.utils.drought import _classify

GOLDEN_Z_TO_CLASS = [
    (-3.00, "EXTREMEMENT_BAS"),
    (-1.76, "EXTREMEMENT_BAS"),
    (-1.75, "TRES_BAS"),          # borne basse inclusive
    (-1.50, "TRES_BAS"),
    (-1.29, "TRES_BAS"),
    (-1.28, "BAS"),               # borne
    (-1.00, "BAS"),
    (-0.85, "BAS"),
    (-0.84, "NORMAL"),            # borne
    (0.00, "NORMAL"),
    (0.83, "NORMAL"),
    (0.84, "HAUT"),               # borne
    (1.00, "HAUT"),
    (1.27, "HAUT"),
    (1.28, "TRES_HAUT"),          # borne
    (1.50, "TRES_HAUT"),
    (1.74, "TRES_HAUT"),
    (1.75, "EXTREMEMENT_HAUT"),   # borne
    (3.00, "EXTREMEMENT_HAUT"),
]


def test_classify_golden_contract():
    for z, expected in GOLDEN_Z_TO_CLASS:
        assert _classify(z) == expected, f"z={z} -> {_classify(z)} (attendu {expected})"
    assert _classify(None) == "UNKNOWN"
