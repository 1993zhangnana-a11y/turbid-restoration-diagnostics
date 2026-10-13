"""Run every self-check without pytest: python tests/run_all_checks.py"""
import sys, inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_selfchecks as a
mods = [a]
try:
    import test_models as b; mods.append(b)
except Exception as e:
    print('[skip] model tests (torch missing?)', e)
n = 0
for m in mods:
    for name, f in inspect.getmembers(m, inspect.isfunction):
        if name.startswith('test_'):
            f(); n += 1; print('PASS', name)
print(f'{n} checks passed')
