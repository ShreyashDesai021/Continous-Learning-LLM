"""Enable local quantized Qwen while retaining neural BGE retrieval."""
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MODEL='evollm-qwen:3b'
URL='http://127.0.0.1:11434'

def main():
    try:
        with urllib.request.urlopen(URL+'/api/tags',timeout=10) as response:
            names={m['name'] for m in json.load(response).get('models',[])}
    except OSError as exc:
        raise SystemExit('Ollama is not reachable. Open Ollama from the Start menu and retry.') from exc
    if MODEL not in names:
        raise SystemExit('Qwen is not imported. Run setup-local-qwen.bat.')
    path=ROOT/'.env'
    lines=path.read_text(encoding='utf-8').splitlines() if path.exists() else []
    updates={'EVOLLM_LLM_BACKEND':'ollama','EVOLLM_OLLAMA_MODEL':MODEL,
             'EVOLLM_OLLAMA_URL':URL,'EVOLLM_GENERATOR':'qwen','EVOLLM_RETRIEVAL':'bge'}
    kept=[line for line in lines if line.split('=',1)[0].strip() not in updates]
    if path.exists():
        from datetime import datetime
        backup=ROOT/'qwen_runtime_backups'/('env-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        backup.parent.mkdir(exist_ok=True);backup.write_bytes(path.read_bytes())
    path.write_text('\n'.join(kept+[k+'='+v for k,v in updates.items()])+'\n',encoding='utf-8')
    print('Configured BGE retrieval + local quantized Qwen2.5-3B. Restart with start.bat.')
    print('All comparison pipelines now use the same Qwen runtime. This changes inference precision; remeasure performance.')

if __name__=='__main__':main()
