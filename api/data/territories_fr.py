# api/data/territories_fr.py
"""Static French department -> region (INSEE COG) lookup. No DB.

Region codes match the bundled frontend regions.geojson `code` property.
"""
from __future__ import annotations

REGION_NAMES: dict[str, str] = {
    "11": "Île-de-France", "24": "Centre-Val de Loire",
    "27": "Bourgogne-Franche-Comté", "28": "Normandie",
    "32": "Hauts-de-France", "44": "Grand Est", "52": "Pays de la Loire",
    "53": "Bretagne", "75": "Nouvelle-Aquitaine", "76": "Occitanie",
    "84": "Auvergne-Rhône-Alpes", "93": "Provence-Alpes-Côte d'Azur",
    "94": "Corse", "01": "Guadeloupe", "02": "Martinique",
    "03": "Guyane", "04": "La Réunion", "06": "Mayotte",
}

DEPT_TO_REGION: dict[str, str] = {
    "01": "84", "03": "84", "07": "84", "15": "84", "26": "84", "38": "84",
    "42": "84", "43": "84", "63": "84", "69": "84", "73": "84", "74": "84",
    "21": "27", "25": "27", "39": "27", "58": "27", "70": "27", "71": "27",
    "89": "27", "90": "27",
    "22": "53", "29": "53", "35": "53", "56": "53",
    "18": "24", "28": "24", "36": "24", "37": "24", "41": "24", "45": "24",
    "2A": "94", "2B": "94",
    "08": "44", "10": "44", "51": "44", "52": "44", "54": "44", "55": "44",
    "57": "44", "67": "44", "68": "44", "88": "44",
    "02": "32", "59": "32", "60": "32", "62": "32", "80": "32",
    "75": "11", "77": "11", "78": "11", "91": "11", "92": "11", "93": "11",
    "94": "11", "95": "11",
    "14": "28", "27": "28", "50": "28", "61": "28", "76": "28",
    "16": "75", "17": "75", "19": "75", "23": "75", "24": "75", "33": "75",
    "40": "75", "47": "75", "64": "75", "79": "75", "86": "75", "87": "75",
    "09": "76", "11": "76", "12": "76", "30": "76", "31": "76", "32": "76",
    "34": "76", "46": "76", "48": "76", "65": "76", "66": "76", "81": "76",
    "82": "76",
    "44": "52", "49": "52", "53": "52", "72": "52", "85": "52",
    "04": "93", "05": "93", "06": "93", "13": "93", "83": "93", "84": "93",
    "971": "01", "972": "02", "973": "03", "974": "04", "976": "06",
}

DEPARTMENT_NAMES: dict[str, str] = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche",
    "08": "Ardennes", "09": "Ariège", "10": "Aube", "11": "Aude",
    "12": "Aveyron", "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal",
    "16": "Charente", "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze",
    "2A": "Corse-du-Sud", "2B": "Haute-Corse", "21": "Côte-d'Or",
    "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne", "25": "Doubs",
    "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir", "29": "Finistère",
    "30": "Gard", "31": "Haute-Garonne", "32": "Gers", "33": "Gironde",
    "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre",
    "37": "Indre-et-Loire", "38": "Isère", "39": "Jura", "40": "Landes",
    "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire",
    "44": "Loire-Atlantique", "45": "Loiret", "46": "Lot",
    "47": "Lot-et-Garonne", "48": "Lozère", "49": "Maine-et-Loire",
    "50": "Manche", "51": "Marne", "52": "Haute-Marne", "53": "Mayenne",
    "54": "Meurthe-et-Moselle", "55": "Meuse", "56": "Morbihan",
    "57": "Moselle", "58": "Nièvre", "59": "Nord", "60": "Oise", "61": "Orne",
    "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin",
    "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône",
    "71": "Saône-et-Loire", "72": "Sarthe", "73": "Savoie",
    "74": "Haute-Savoie", "75": "Paris", "76": "Seine-Maritime",
    "77": "Seine-et-Marne", "78": "Yvelines", "79": "Deux-Sèvres",
    "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne", "83": "Var",
    "84": "Vaucluse", "85": "Vendée", "86": "Vienne", "87": "Haute-Vienne",
    "88": "Vosges", "89": "Yonne", "90": "Territoire de Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise", "971": "Guadeloupe",
    "972": "Martinique", "973": "Guyane", "974": "La Réunion",
    "976": "Mayotte",
}


def region_of(dept_code: str | None) -> str | None:
    if dept_code is None:
        return None
    return DEPT_TO_REGION.get(dept_code)
