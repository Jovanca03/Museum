"""Live smoke test against local Ollama + the real Chroma database (no mocks)."""
import argparse
from api_server import _load_catalog, _retrieve

CASES = [
    ("CAKRA_ART013", "Apa fungsi dari artefak ini?", ("pagelaran", "menyambut tamu")),
    ("CAKRA_ART013", "Di mana gamelan ini digunakan?", ("Pendopo Keraton Bangkalan",)),
    ("CAKRA_ART013", "Siapa nama artefak ini?", ("Gamelan Se Lajing",)),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    artifacts = _load_catalog()
    by_id = {a.get('artifact_id'): a for a in artifacts}
    passed = 0
    for artifact_id, question, expected_any in CASES:
        artifact = by_id[artifact_id]
        name = artifact.get('name') or artifact.get('vr_display_name') or artifact_id
        query, docs = _retrieve(question, artifact_id, name, args.top_k)
        combined = '\n'.join(d['text'] for d in docs).casefold()
        ok = bool(docs) and any(term.casefold() in combined for term in expected_any)
        print(f'\n[{"PASS" if ok else "FAIL"}] {question}')
        print(f'query={query}')
        for i, doc in enumerate(docs, 1):
            print(f'  {i}. chunk={doc["chunk_id"]} | source={doc["source"]}')
            print('     ' + doc['text'][:320].replace('\n', ' '))
        if not docs:
            print('  Tidak ada chunk yang ditemukan untuk artifact_id aktif.')
        if not ok:
            print(f'  Fakta yang dicari tidak muncul: {expected_any}')
        passed += int(ok)
    print(f'\nHASIL RETRIEVAL: {passed}/{len(CASES)} lulus')
    raise SystemExit(0 if passed == len(CASES) else 1)

if __name__ == '__main__':
    main()
