"""Insert the rows of re_table.py into the Reynolds-number table of the paper (between markers).
Usage: python fill_retable.py path/to/paper.tex"""
import subprocess, re, os, sys
out = subprocess.run(['python3', 're_table.py'], capture_output=True, text=True, env=dict(os.environ, OPENBLAS_NUM_THREADS='1')).stdout.strip().splitlines()
inner = [l for l in out[:5]]; outer = [l for l in out[5:]]
inner[0] = 'inner' + inner[0]; outer[0] = 'outer' + outer[0]
f = sys.argv[1]
s = open(f).read()
for tag, rows in [('INNER', inner), ('OUTER', outer)]:
    block = f'% RETABLE-{tag}-BEGIN\n' + '\n'.join(rows) + f'\n% RETABLE-{tag}-END'
    if f'RETABLE{tag}' in s:
        s = s.replace(f'RETABLE{tag}', block)
    else:
        s = re.sub(rf'% RETABLE-{tag}-BEGIN.*?% RETABLE-{tag}-END', lambda m: block, s, flags=re.S)
open(f, 'w').write(s)
print('\n'.join(inner + outer))
