"""Check only the public candidate tree; never inspect private project assets."""
import csv
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.py', '.pyx', '.cpp', '.h', '.md', '.txt', '.json', '.csv',
           '.drawio', '.pdf', '.svg', '.png'}
TEXT = {'.py', '.pyx', '.cpp', '.h', '.md', '.txt', '.json', '.csv', '.drawio', '.svg'}
PATTERNS = [re.compile(r'(?i)\b[A-Z]:[\\/]'),
            re.compile(r'/(?:Users|home)/[^/\s]+/'),
            re.compile(r'sk-[A-Za-z0-9]{20,}'), re.compile(r'AKIA[0-9A-Z]{16}'),
            re.compile('-----BEGIN ' + '(?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]


def main():
    failures, files = [], []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        files.append(str(relative))
        if path.suffix not in ALLOWED and relative.as_posix() not in {'.gitignore', 'LICENSE'}:
            failures.append(f'Unapproved extension: {relative}')
            continue
        if any(part in {'data', 'runs', 'build', 'qa', '.cache', '__pycache__', '.git', '.venv'} for part in relative.parts):
            failures.append(f'Generated/private directory: {relative}')
        if path.suffix in TEXT or path.name in {'LICENSE', '.gitignore'}:
            text = path.read_text(encoding='utf-8-sig')
            for pattern in PATTERNS:
                if pattern.search(text):
                    failures.append(f'Possible secret or machine path: {relative}')
    expected = {'final_per_seed.csv': (48, {'density_percent', 'seed', 'method', 'n', 'mae', 'rmse'}),
                'final_aggregate.csv': (16, {'density_percent', 'method', 'seeds', 'mae_mean', 'mae_sample_sd', 'rmse_mean', 'rmse_sample_sd'})}
    for name, (count, columns) in expected.items():
        with (ROOT / 'results' / name).open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if set(reader.fieldnames) != columns or len(list(reader)) != count:
                failures.append(f'Unexpected result schema/count: {name}')
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'PASS: {len(files)} allowed files; no model/data archives, binaries, caches, or detected machine paths/secret patterns.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
