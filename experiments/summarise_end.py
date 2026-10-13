"""Four-level mean of a prior effect at given epochs from the output of experiments/signif.py
(mean of the per-level means and of the per-level CI bounds, as in Table V of the paper).
Usage: python experiments/summarise_end.py SIGNIF_CSV ARM REGION EPOCHS   e.g. ... W:none background 80,160"""
import csv, sys
path, arm, region = sys.argv[1], sys.argv[2], sys.argv[3]
epochs = [int(e) for e in sys.argv[4].split(',')]
rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
for ep in epochs:
    r = [x for x in rows if x['arm'] == arm and x['region'] == region and int(x['epoch']) == ep]
    if len(r) != 4:
        print(ep, 'levels', len(r)); continue
    m = sum(float(x['mean']) for x in r) / 4
    lo = sum(float(x['ci_lo']) for x in r) / 4
    hi = sum(float(x['ci_hi']) for x in r) / 4
    sd = sorted(float(x['seed_sd']) for x in r)
    per = ' '.join(f"{x['rho'].replace('NUT_DAN', 'NTU')}:{float(x['mean']):+.3f}({x['sign_consistent']})" for x in r)
    print(f'ep{ep:4d} mean {m:+.3f} [{lo:+.2f}, {hi:+.2f}]  seedsd {sd}  {per}')
