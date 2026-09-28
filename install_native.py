"""Separate environments keep native framework dependencies away from LAYA/Torch."""
import argparse,json,subprocess,sys,venv
from pathlib import Path
PACKAGES={'freqtrade':'freqtrade==2026.8','jesse':'jesse==3.2.3'}
def install(name,root):
 target=Path(root)/name
 venv.EnvBuilder(with_pip=True).create(target)
 python=target/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
 subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check',PACKAGES[name]],check=True)
 return str(python.resolve())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('framework',choices=PACKAGES);p.add_argument('--root',default='.native')
 a=p.parse_args();print(install(a.framework,a.root))
