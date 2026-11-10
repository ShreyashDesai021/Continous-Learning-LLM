"""Download pinned official Qwen GGUF weights and import them into local Ollama."""
import argparse,hashlib,json,os,subprocess,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL='evollm-qwen:3b'
URL='https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf'
EXPECTED='626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d'

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--ollama-exe',required=True);args=p.parse_args()
    try:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as response:json.load(response)
    except OSError as exc:raise SystemExit('Start Ollama from the Windows Start menu and retry.') from exc
    folder=ROOT/'storage/ollama';folder.mkdir(parents=True,exist_ok=True)
    weights=folder/'qwen2.5-3b-instruct-q4_k_m.gguf'
    if not weights.exists() or digest(weights)!=EXPECTED:
        part=weights.with_suffix('.gguf.part')
        offset=part.stat().st_size if part.exists() else 0
        req=urllib.request.Request(URL,headers={'Range':f'bytes={offset}-'} if offset else {})
        print('Downloading official quantized Qwen2.5-3B weights, approximately 2.1 GB.',flush=True)
        with urllib.request.urlopen(req,timeout=120) as response:
            resume=offset>0 and response.status==206
            if not resume:offset=0
            total=int(response.headers.get('Content-Length',0))+offset;done=offset;previous=-1
            with part.open('ab' if resume else 'wb') as f:
                while block:=response.read(1024*1024):
                    f.write(block);done+=len(block)
                    pct=int(done/total*10)*10 if total else -1
                    if pct!=previous:print(str(pct)+'%',flush=True);previous=pct
        if digest(part)!=EXPECTED:
            raise SystemExit('Model checksum mismatch. Do not use this file; remove the .part file and retry.')
        os.replace(part,weights)
    modelfile=folder/'Modelfile'
    modelfile.write_text('FROM "'+weights.as_posix()+'"\nSYSTEM You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nTEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{- end }}<|im_start|>assistant\n"""\nPARAMETER stop <|im_end|>\nPARAMETER stop <|im_start|>\n',encoding='utf-8')
    subprocess.run([args.ollama_exe,'create',MODEL,'-f',str(modelfile)],check=True)
    print('Imported '+MODEL+'. The weights checksum matches the tested model.')

if __name__=='__main__':main()
