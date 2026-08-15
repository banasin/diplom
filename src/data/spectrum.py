"""Панель белков-мишеней и множества известных мишеней по аптамеру для анализа
спектра мишеней и кросс-реактивности (Часть 3)."""


def build_target_panel(pairs, protein_emb) -> list:
    """Упорядоченный список уникальных белков-мишеней, у которых есть ESM-2
    эмбеддинг (панель P для профилей)."""
    seen = dict.fromkeys(pairs["target_key"])
    return [k for k in seen if k in protein_emb]


def known_targets_by_aptamer(pairs) -> dict:
    """Для каждого аптамера — множество его известных белков-мишеней."""
    out: dict = {}
    for seq, key in zip(pairs["aptamer_seq"], pairs["target_key"]):
        out.setdefault(seq, set()).add(key)
    return out
