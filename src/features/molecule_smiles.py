"""Получение SMILES малых молекул из PubChem по CID с кэшем.

Кэш — JSON в data/ (вне git). Из сети тянется только отсутствующее."""
import json
import re
from pathlib import Path

import requests

PUBCHEM_URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
               "{cid}/property/CanonicalSMILES/TXT")


def parse_cid(target_cid) -> int | None:
    if target_cid is None:
        return None
    m = re.search(r"(\d+)", str(target_cid))
    return int(m.group(1)) if m else None


def fetch_smiles(cid: int) -> str | None:
    try:
        r = requests.get(PUBCHEM_URL.format(cid=cid), timeout=30)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip().splitlines()[0].strip()
    except requests.RequestException:
        return None
    return None


def load_smiles(keys_cids: dict, cache_path: str) -> dict:
    cache_file = Path(cache_path)
    cache: dict = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    changed = False
    for key, target_cid in keys_cids.items():
        if key in cache:
            continue
        cid = parse_cid(target_cid)
        if cid is None:
            continue
        smi = fetch_smiles(cid)
        if smi:
            cache[key] = smi
            changed = True
    if changed:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return {k: cache[k] for k in keys_cids if k in cache}
