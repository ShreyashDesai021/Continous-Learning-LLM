"""Explicitly download optional model weights. No model weights are bundled."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from run import load_local_env

def main():
    load_local_env()
    parser=argparse.ArgumentParser(description='Download EvoLLM pretrained models into your Hugging Face cache.')
    parser.add_argument('--only', choices=['all','embedding','llm'],default='all')
    args=parser.parse_args()
    from huggingface_hub import snapshot_download,HfApi
    models=[]
    if args.only in ('all','embedding'):
        models.append(os.getenv('EVOLLM_EMBEDDING_MODEL','BAAI/bge-m3'))
    if args.only in ('all','llm'):
        models.append(os.getenv('EVOLLM_LLM_MODEL','Qwen/Qwen2.5-1.5B-Instruct'))
    for model in models:
        if Path(model).is_dir():
            print('Using existing local model:',model,flush=True)
            continue
        print('Downloading:',model,flush=True)
        files=HfApi().list_repo_files(model)
        ignored=['*.onnx','*.msgpack','*.h5','*.ot','*.gguf','onnx/*','openvino/*']
        # Prefer safe tensors when present, but support repositories with bin weights.
        if any(name.endswith('.safetensors') and '/' not in name for name in files):
            ignored.append('*.bin')
        print(snapshot_download(model,ignore_patterns=ignored),flush=True)
    print('Downloads complete. Select Qwen in the app. Set EVOLLM_RETRIEVAL=bge before launch for dense retrieval.')

if __name__=='__main__':main()
