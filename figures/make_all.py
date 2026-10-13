"""Build all figures, or some of them.

    python make_all.py                 # all figures
    python make_all.py fig03 fig07     # only these (prefix of the script name)

Each figNN_*.py script is run in its own process.  Figure 1 (TikZ) is compiled with
pdflatex if it is available, and skipped otherwise.
"""
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import OUT


def build_tikz():
    if shutil.which('pdflatex') is None:
        print('skip fig01_schematic.tex (pdflatex not found)')
        return True
    r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-jobname', 'fig1_schematic',
                        '-output-directory', OUT, 'fig01_schematic.tex'], cwd=HERE, capture_output=True, text=True)
    for ext in ('aux', 'log'):
        f = os.path.join(OUT, 'fig1_schematic.' + ext)
        if r.returncode == 0 and os.path.exists(f):
            os.remove(f)
    print('wrote' if r.returncode == 0 else 'FAILED', os.path.join(OUT, 'fig1_schematic.pdf'))
    return r.returncode == 0


def main(which):
    scripts = sorted(glob.glob(os.path.join(HERE, 'fig[0-9][0-9]_*.py')))
    if which:
        scripts = [s for s in scripts if any(os.path.basename(s).startswith(w) for w in which)]
    failed = []
    if not which or any('fig01'.startswith(w) or w.startswith('fig01') for w in which):
        if not build_tikz():
            failed.append('fig01_schematic.tex')
    for s in scripts:
        r = subprocess.run([sys.executable, s], cwd=HERE)
        if r.returncode != 0:
            failed.append(os.path.basename(s))
    if failed:
        print('FAILED:', ', '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1:])
