"""Verify or restore only the project-local frozen task environments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore',action='store_true');parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    if os.environ.get('CODE_AGENT_ROOT')!=str(ROOT):raise SystemExit('Source scripts/env.sh first')
    uv=ROOT/'runtime/tools/uv-package/bin/uv'
    manifest=json.loads((ROOT/'runtime/gym-baseline/environment-runtime-manifest.json').read_text())
    for item in manifest:
        env=Path(item['env']);python=Path(item['python']);lock=Path(item['lock'])
        assert env.is_relative_to(ROOT/'runtime/gym-baseline/envs') and lock.is_relative_to(env.parent)
        assert hashlib.sha256(lock.read_bytes()).hexdigest()==item['lock_sha256']
        if args.restore:
            version=item['python_version'].split()[1]
            if not python.exists():
                subprocess.run([str(uv),'python','install',version,*(['--offline'] if args.offline else [])],check=True)
                subprocess.run([str(uv),'venv','--python',version,str(env)],check=True)
            subprocess.run([str(uv),'pip','sync','--python',str(python),'--require-hashes',str(lock),
                            *(['--offline'] if args.offline else [])],check=True)
        actual=subprocess.check_output([str(python),'--version'],text=True).strip()
        assert actual==item['python_version'],(env,actual)
        subprocess.run([str(uv),'pip','check','--python',str(python)],check=True)
        expected=(env.parent/(env.name+'.freeze.txt')).read_text().strip()
        installed=subprocess.check_output([str(uv),'pip','freeze','--python',str(python)],text=True).strip()
        assert installed==expected,env
        print(env.name,'verified',actual,flush=True)


if __name__=='__main__':main()
