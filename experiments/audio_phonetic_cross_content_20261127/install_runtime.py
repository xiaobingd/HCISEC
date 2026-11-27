"""Install only verified public wheels into a short workspace prefix for eSpeak."""
import json,hashlib,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
wheelhouse=ROOT/'.runtime/phonetic_wheels'
manifest=json.loads((wheelhouse/'manifest.json').read_text(encoding='utf-8-sig'))
target=Path(os.environ.get('PHONETIC_RUNTIME',str(ROOT.parents[1]/'.phonetic-runtime')))
paths=[]
core={'transformers','tokenizers','huggingface-hub','safetensors','filelock','regex','requests','tqdm','fsspec','typing-extensions','urllib3','charset-normalizer','idna','certifi','colorama','espeakng-loader'}
for row in manifest:
    if row['package'] not in core:continue
    path=wheelhouse/row['filename'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];paths.append(str(path))
assert len(paths)==len(core),'Verified core runtime wheel missing'
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(target),*paths],check=True)
print('SHORT_RUNTIME_INSTALLED',target)
