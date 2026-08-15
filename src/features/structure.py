"""Вторичная структура аптамера алгоритмом Nussinov (максимизация числа
комплементарных пар) и структурные признаки — третий тип кодирования (Часть 4).
Без внешних зависимостей; U≡T."""
import re
import numpy as np

_PAIRS = {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}
STRUCT_FEATURE_NAMES = ["frac_paired", "n_pairs", "n_hairpins", "max_stem", "energy"]


def _norm(seq: str) -> str:
    return seq.strip().upper().replace("U", "T")


def nussinov(seq: str, min_loop: int = 3) -> str:
    s = _norm(seq)
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for span in range(min_loop + 1, n):
        for i in range(n - span):
            j = i + span
            best = max(dp[i + 1][j], dp[i][j - 1])
            if (s[i], s[j]) in _PAIRS:
                best = max(best, dp[i + 1][j - 1] + 1)
            for k in range(i + 1, j):
                best = max(best, dp[i][k] + dp[k + 1][j])
            dp[i][j] = best

    struct = ["."] * n

    def trace(i: int, j: int) -> None:
        if i >= j:
            return
        if dp[i][j] == dp[i + 1][j]:
            trace(i + 1, j)
        elif dp[i][j] == dp[i][j - 1]:
            trace(i, j - 1)
        elif (s[i], s[j]) in _PAIRS and dp[i][j] == dp[i + 1][j - 1] + 1:
            struct[i], struct[j] = "(", ")"
            trace(i + 1, j - 1)
        else:
            for k in range(i + 1, j):
                if dp[i][j] == dp[i][k] + dp[k + 1][j]:
                    trace(i, k)
                    trace(k + 1, j)
                    return

    if n > 0:
        trace(0, n - 1)
    return "".join(struct)


def structure_features(seq: str) -> np.ndarray:
    st = nussinov(seq)
    n = len(st) if len(st) else 1
    n_pairs = st.count("(")
    frac_paired = 2 * n_pairs / n
    n_hairpins = len(re.findall(r"\(\.+\)", st))            # ( петля )
    stems = re.findall(r"\(+", st)
    max_stem = max((len(x) for x in stems), default=0)
    energy = -n_pairs / n                                    # проще = ниже
    return np.array([frac_paired, n_pairs, n_hairpins, max_stem, energy],
                    dtype=np.float32)
