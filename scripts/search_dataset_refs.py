import os
from pathlib import Path

search_terms = [
    'Nutrient Deficiency Obj',
    'Soybean Crop Disease',
    'Soybean Crop Disease.v10',
    'Nutrient Deficiency Obj.v1i.yolov11',
    'Soybean Crop Disease.v10-version_1.yolov11'
]

import os
from pathlib import Path

search_terms = [
    'Nutrient Deficiency Obj',
    'Soybean Crop Disease.v10',
    'Nutrient Deficiency Obj.v1i.yolov11',
    'Soybean Crop Disease.v10-version_1.yolov11'
]

targets = ['src', 'configs', 'scripts', 'tests', 'docs', 'outputs', 'app.py', 'run_pipeline.py']

matches = {}
for target in targets:
    p = Path(target)
    if not p.exists():
        continue
    if p.is_file():
        candidates = [p]
    else:
        candidates = [f for f in p.rglob('*') if f.is_file() and f.suffix in ['.py', '.yaml', '.yml', '.json', '.md', '.txt', '.sh', '.bat']]

    for fpath in candidates:
        try:
            content = fpath.read_text(encoding='utf-8', errors='ignore')
            for term in search_terms:
                if term in content:
                    matches.setdefault(str(fpath), []).append(term)
        except Exception:
            pass

print(f"Total matching files: {len(matches)}")
for path, terms in sorted(matches.items()):
    print(f"{path}: {list(set(terms))}")

print(f"Total matching files: {len(matches)}")
for path, terms in sorted(matches.items()):
    print(f"{path}: {list(set(terms))}")
