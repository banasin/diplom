"""Кластеризация последовательностей аптамеров по сходству (аналог CD-HIT).

Идентичность — доля совпавших позиций по глобальному выравниванию
(Needleman–Wunsch) относительно длины более короткой последовательности.
Кластеризация — жадная: сортировка по длине, приклеивание к первому подходящему
представителю. Нормализация U→T перед сравнением, чтобы ДНК/РНК-гомологи не
разъезжались между train/test."""
from src.features.aptamer import normalize_seq


def _nw_matches(a: str, b: str) -> int:
    """Число совпавших позиций в оптимальном глобальном выравнивании
    (match=1, mismatch=0, gap=0)."""
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (1 if ai == b[j - 1] else 0)
            cur[j] = max(diag, prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def _kmers(seq: str, k: int = 5) -> set[str]:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def identity(a: str, b: str) -> float:
    a, b = normalize_seq(a), normalize_seq(b)
    if not a or not b:
        return 0.0
    return _nw_matches(a, b) / min(len(a), len(b))


def greedy_cluster(seqs: list[str], threshold: float) -> list[int]:
    norm = [normalize_seq(s) for s in seqs]
    order = sorted(range(len(norm)), key=lambda i: len(norm[i]), reverse=True)
    reps: list[tuple[int, set[str]]] = []      # (индекс представителя, его k-меры)
    labels = [-1] * len(norm)
    for i in order:
        s = norm[i]
        ks = _kmers(s)
        assigned = None
        for rep_i, rep_ks in reps:
            if ks and rep_ks and ks.isdisjoint(rep_ks):
                continue                        # префильтр: нет общих k-меров
            if identity(s, norm[rep_i]) >= threshold:
                assigned = labels[rep_i]
                break
        if assigned is None:
            assigned = len(reps)
            reps.append((i, ks))
        labels[i] = assigned
    return labels
