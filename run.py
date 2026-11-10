"""Portable entry point: python run.py [--port 8000] [--reload]."""
import argparse
import os
from pathlib import Path


def load_local_env():
    root=Path(__file__).resolve().parent
    os.chdir(root)
    env=root/'.env'
    if env.exists():
        for line in env.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key,value=line.split('=',1)
            if key.strip().startswith('EVOLLM_'):
                os.environ.setdefault(key.strip(),value.strip().strip('"').strip("'"))

if __name__ == '__main__':
    load_local_env()
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    import uvicorn
    print(f'EvoLLM Enterprise: http://127.0.0.1:{args.port}')
    uvicorn.run('enterprise.api:create_app', factory=True, host='127.0.0.1', port=args.port, reload=args.reload)
