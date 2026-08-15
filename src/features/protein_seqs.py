"""Скачивание последовательностей белков из UniProt по accession с кэшем.

Кэш — JSON в data/ (вне git). Из сети тянутся только отсутствующие записи."""
import json
from pathlib import Path

import requests

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def parse_fasta(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith(">")]
    return "".join(lines)


def fetch_uniprot(accession: str) -> str | None:
    try:
        r = requests.get(UNIPROT_URL.format(acc=accession), timeout=30)
        if r.status_code == 200 and r.text.startswith(">"):
            return parse_fasta(r.text)
    except requests.RequestException:
        return None
    return None


def load_sequences(accessions: list[str], cache_path: str) -> dict[str, str]:
    cache_file = Path(cache_path)
    cache: dict[str, str] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    changed = False
    for acc in accessions:
        if acc in cache or acc is None:
            continue
        seq = fetch_uniprot(acc)
        if seq:
            cache[acc] = seq
            changed = True
    if changed:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return {acc: cache[acc] for acc in accessions if acc in cache}
