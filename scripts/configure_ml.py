"""Enable the downloaded local neural engines while preserving workspace settings."""
from pathlib import Path
root=Path(__file__).resolve().parent.parent
path=root/'.env'
lines=path.read_text(encoding='utf-8').splitlines() if path.exists() else []
updates={'EVOLLM_RETRIEVAL':'bge','EVOLLM_GENERATOR':'qwen'}
kept=[line for line in lines if line.split('=',1)[0].strip() not in updates]
path.write_text('\n'.join(kept+[key+'='+value for key,value in updates.items()])+'\n',encoding='utf-8')
print('Local BGE-M3 and Qwen enabled. Restart the application.')
