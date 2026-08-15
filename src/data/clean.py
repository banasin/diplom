"""
Задача 1 Части 1: чистка агрегированного датасета до рабочих наборов пар
«аптамер–мишень» с единой логарифмической шкалой аффинности (log-Kd).

Мишени двух физически разных классов готовятся в ОТДЕЛЬНЫЕ таблицы:
  * белки        (Type Target == Protein)        → для Части 1;
  * малые молекулы (LMW compound / Small molecule) → для Части 2.
Общие шаги очистки (аффинность, валидация последовательности, дедуп) едины;
различается только ключ мишени.

Запуск из корня репозитория:
    ./.venv/Scripts/python.exe -m src.data.clean

Вход:  data/dataset.xlsx  (агрегированный из 6 баз)
Выход: data/dataset_protein.parquet  / .csv  — очищенные пары аптамер–белок
       data/dataset_smallmol.parquet / .csv  — очищенные пары аптамер–малая молекула
       data/cleaning_report.txt               — отчёт об отсеве (для текста диплома)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Печатаем отчёт в UTF-8 независимо от кодовой страницы консоли Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --- Параметры чистки (все решения собраны здесь, чтобы легко менять) ------
RAW_PATH = Path("data/dataset.xlsx")
REPORT_PATH = Path("data/cleaning_report.txt")

OUT = {
    "protein":  {"parquet": Path("data/dataset_protein.parquet"),
                 "csv":     Path("data/dataset_protein.csv")},
    "smallmol": {"parquet": Path("data/dataset_smallmol.parquet"),
                 "csv":     Path("data/dataset_smallmol.csv")},
}

# Значения колонки "Type Target", относящиеся к каждому классу мишеней.
PROTEIN_TYPES = {"Protein"}
SMALLMOL_TYPES = {"LMW compound", "Small molecule"}

# Физически осмысленный диапазон Kd (в молях). Вне его — артефакты/опечатки.
KD_MIN_M = 1e-12   # 1 пМ  — предел самых сильных реальных аптамеров
KD_MAX_M = 1e-4    # 100 мкМ — слабее уже не считаем специфичным связыванием

# Минимальная длина последовательности аптамера (короче — обрывки/ошибки).
MIN_SEQ_LEN = 10

VALID_NT = set("ACGTU")  # допустимые символы нуклеотидной последовательности


def clean_class(df_raw: pd.DataFrame, target_types: set[str], title: str,
                report) -> pd.DataFrame:
    """Провести общие шаги очистки на срезе одного класса мишеней и
    вернуть дедуплицированную таблицу уникальных пар. `report` — функция
    логирования (печать + запись в отчёт)."""
    report("")
    report(f"########## {title} ##########")

    n0 = len(df_raw)
    df = df_raw[df_raw["Type Target"].isin(target_types)].copy()
    report(f"[1] Класс мишени {sorted(target_types)}: {len(df):5d}  (из {n0})")

    # 2. Только измерения Kd ----------------------------------------------
    n = len(df)
    df = df[df["Affinity type"] == "Kd"].copy()
    report(f"[2] Affinity type == Kd:               {len(df):5d}  (−{n - len(df)})")

    # 3. Единица nM (в срезе Kd практически всё в nM; отсекаем прочее) -----
    n = len(df)
    df = df[df["Affinity unit"] == "nM"].copy()
    report(f"[3] Affinity unit == nM:               {len(df):5d}  (−{n - len(df)})")

    # 4. Числовое положительное значение аффинности -----------------------
    n = len(df)
    df["kd_nM"] = pd.to_numeric(df["Affinity value exact"], errors="coerce")
    df = df[df["kd_nM"].notna() & (df["kd_nM"] > 0)].copy()
    report(f"[4] Значение Kd числовое и > 0:        {len(df):5d}  (−{n - len(df)})")

    # 5. Перевод в единую логарифмическую шкалу ----------------------------
    #    Kd[M] = Kd[nM] * 1e-9;  log_Kd = log10(Kd[M]);  pKd = −log_Kd.
    df["kd_M"] = df["kd_nM"] * 1e-9
    df["log_Kd"] = np.log10(df["kd_M"])     # целевая переменная модели
    df["pKd"] = -df["log_Kd"]               # удобная шкала: больше = сильнее

    # 6. Отсечение физически невозможных значений --------------------------
    n = len(df)
    df = df[(df["kd_M"] >= KD_MIN_M) & (df["kd_M"] <= KD_MAX_M)].copy()
    report(f"[5] Kd в [{KD_MIN_M:g}; {KD_MAX_M:g}] M:      {len(df):5d}  (−{n - len(df)})")

    # 7. Валидация последовательности аптамера -----------------------------
    df["aptamer_seq"] = df["Sequence"].astype(str).str.strip().str.upper()

    n = len(df)
    valid = df["aptamer_seq"].apply(lambda s: len(set(s) - VALID_NT) == 0 and len(s) > 0)
    df = df[valid].copy()
    report(f"[6] Последовательность только из ACGTU: {len(df):5d}  (−{n - len(df)})")

    n = len(df)
    df = df[df["aptamer_seq"].str.len() >= MIN_SEQ_LEN].copy()
    report(f"[7] Длина последовательности ≥ {MIN_SEQ_LEN}:     {len(df):5d}  (−{n - len(df)})")

    # 8. Идентификаторы мишени и флаг «неточного» значения ------------------
    df["aptamer_type"] = df["Type Aptamer"]                    # DNA / RNA
    df["target_id"] = df["Target ID"]      # UniProt (белки) / PubChem CID (молекулы)
    df["target_name"] = df["Target Name"]
    df["target_organism"] = df["Target organism"]
    df["qualified"] = (df["Affinity flag"] == "!")            # значение оговорённое
    # ключ мишени: стабильный ID, а при его отсутствии — имя мишени
    df["target_key"] = df["target_id"].fillna(df["target_name"])

    # 9. Схлопывание ТОЛЬКО полностью идентичных записей --------------------
    #    Одну строку получают лишь записи, у которых совпадают
    #    последовательность аптамера, его тип, мишень И численное значение Kd
    #    (одно и то же измерение, продублированное между источниками).
    #    Разные измерения одной пары НЕ сливаются — остаются отдельными
    #    строками, чтобы не подменять реальный разброс аффинности медианой.
    #    Тип (DNA/RNA) входит в ключ: одинаковая последовательность как ДНК
    #    и РНК — разные молекулы.
    key = ["aptamer_seq", "aptamer_type", "target_key", "kd_nM"]
    n_before = len(df)
    agg = (
        df.groupby(key, dropna=False)
        .agg(
            log_Kd=("log_Kd", "first"),   # внутри группы все значения равны
            pKd=("pKd", "first"),
            n_duplicates=("log_Kd", "size"),  # сколько идентичных записей слито
            qualified_any=("qualified", "any"),
            target_id=("target_id", "first"),
            target_name=("target_name", "first"),
            target_organism=("target_organism", "first"),
        )
        .reset_index()
    )
    agg["seq_len"] = agg["aptamer_seq"].str.len()

    # Пометка пар с несколькими РАЗНЫМИ измерениями Kd (для анализа разброса).
    pair_key = ["aptamer_seq", "aptamer_type", "target_key"]
    agg["n_measurements"] = agg.groupby(pair_key, dropna=False)["log_Kd"].transform("size")

    report(f"[8] Записей после схлопывания дублей:  {len(agg):5d}  "
           f"(слито {n_before - len(agg)} точных дублей)")
    report(f"    уникальных пар аптамер–мишень:     {agg[pair_key].drop_duplicates().shape[0]:5d}  "
           f"(у {(agg['n_measurements']>1).sum()} записей пара имеет >1 измерения Kd)")

    return agg


def summarize(agg: pd.DataFrame, id_label: str, report) -> None:
    pair_key = ["aptamer_seq", "aptamer_type", "target_key"]
    n_pairs = agg[pair_key].drop_duplicates().shape[0]
    report("")
    report("=== Итоговый очищенный набор ===")
    report(f"Записей (уникальных измерений): {len(agg)}")
    report(f"Уникальных пар аптамер–мишень : {n_pairs}")
    report(f"Уникальных аптамеров         : {agg['aptamer_seq'].nunique()}")
    report(f"Уникальных мишеней (target_key): {agg['target_key'].nunique()}")
    report(f"С {id_label:<22}: {agg['target_id'].notna().sum()} "
           f"({agg['target_id'].notna().mean()*100:.0f}%)")
    report(f"DNA / RNA                    : "
           f"{(agg['aptamer_type']=='DNA').sum()} / {(agg['aptamer_type']=='RNA').sum()}")
    report(f"Оговорённых (qualified)      : {agg['qualified_any'].sum()}")
    report(f"log_Kd: min={agg['log_Kd'].min():.2f}  "
           f"median={agg['log_Kd'].median():.2f}  max={agg['log_Kd'].max():.2f}")


def main() -> None:
    log_lines: list[str] = []

    def report(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    df_raw = pd.read_excel(RAW_PATH)
    report(f"Исходно строк: {len(df_raw)}")

    # --- Белки ------------------------------------------------------------
    prot = clean_class(df_raw, PROTEIN_TYPES, "БЕЛКИ (для Части 1)", report)
    prot_cols = ["aptamer_seq", "aptamer_type", "seq_len",
                 "target_key", "target_id", "target_name", "target_organism",
                 "log_Kd", "pKd", "kd_nM", "n_duplicates", "n_measurements",
                 "qualified_any"]
    prot = prot[prot_cols]
    summarize(prot, "UniProt ID (для ESM-2)", report)
    prot.to_parquet(OUT["protein"]["parquet"], index=False)
    prot.to_csv(OUT["protein"]["csv"], index=False, encoding="utf-8")

    # --- Малые молекулы ---------------------------------------------------
    #     У молекул нет организма; идентификатор — PubChem CID (для SMILES в Части 2).
    lmw = clean_class(df_raw, SMALLMOL_TYPES, "МАЛЫЕ МОЛЕКУЛЫ (для Части 2)", report)
    lmw = lmw.rename(columns={"target_id": "target_cid"})
    lmw_cols = ["aptamer_seq", "aptamer_type", "seq_len",
                "target_key", "target_cid", "target_name",
                "log_Kd", "pKd", "kd_nM", "n_duplicates", "n_measurements",
                "qualified_any"]
    lmw = lmw[lmw_cols]
    # для сводки временно переименуем обратно, чтобы переиспользовать summarize
    summarize(lmw.rename(columns={"target_cid": "target_id"}),
              "PubChem CID (для SMILES)", report)
    lmw.to_parquet(OUT["smallmol"]["parquet"], index=False)
    lmw.to_csv(OUT["smallmol"]["csv"], index=False, encoding="utf-8")

    REPORT_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    report("")
    report(f"Сохранено:")
    report(f"  белки          : {OUT['protein']['parquet']}, {OUT['protein']['csv']}")
    report(f"  малые молекулы : {OUT['smallmol']['parquet']}, {OUT['smallmol']['csv']}")
    report(f"  отчёт          : {REPORT_PATH}")


if __name__ == "__main__":
    main()
