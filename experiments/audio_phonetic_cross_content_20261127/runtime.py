"""Workspace-only frozen teacher runtime and exact eSpeak IPA token generation."""
import os,sys,json,ctypes,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUNTIME=Path(os.environ.get('PHONETIC_RUNTIME',str(ROOT.parents[1]/'.phonetic-runtime')))
sys.path.insert(0,str(RUNTIME))
os.environ['HF_HOME']=str(RUNTIME/'cache/huggingface')
os.environ['HF_HUB_CACHE']=str(RUNTIME/'cache/huggingface/hub')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
MODEL=ROOT/'.runtime/models/wav2vec2-lv-60-espeak-cv-ft'
REVISION='ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4'
WEIGHT_SHA='3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e'

class IPA:
    """Same en-us eSpeak backend as the model tokenizer, no character segmentation."""
    def __init__(self):
        import espeakng_loader
        self.lib=ctypes.CDLL(espeakng_loader.get_library_path())
        init=self.lib.espeak_Initialize;init.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_int];init.restype=ctypes.c_int
        data=espeakng_loader.get_data_path();assert len(data)<160
        if init(2,0,data.encode('utf-8'),0)<=0:raise RuntimeError('Portable eSpeak initialization failed')
        voice=self.lib.espeak_SetVoiceByName;voice.argtypes=[ctypes.c_char_p];voice.restype=ctypes.c_int
        if voice(b'en-us')!=0:raise RuntimeError('en-us voice unavailable')
        info=self.lib.espeak_Info;info.argtypes=[ctypes.POINTER(ctypes.c_char_p)];info.restype=ctypes.c_char_p
        path=ctypes.c_char_p();self.version=info(ctypes.byref(path)).decode();self.data_path=path.value.decode()
        self.fn=self.lib.espeak_TextToPhonemes;self.fn.argtypes=[ctypes.POINTER(ctypes.c_char_p),ctypes.c_int,ctypes.c_int];self.fn.restype=ctypes.c_char_p
    def tokens(self,text):
        ptr=ctypes.pointer(ctypes.c_char_p(text.lower().encode('utf-8')));pieces=[]
        for _ in range(50):
            if ptr.contents.value is None:break
            out=self.fn(ptr,1,(ord('_')<<8)|2)
            if out:pieces.append(out.decode('utf-8'))
        else:raise RuntimeError('eSpeak text pointer did not terminate')
        clean=' '.join(pieces).replace('ˈ','').replace('ˌ','').replace('_',' ')
        return clean.split()

def lexicon(texts):
    from transformers import Wav2Vec2PhonemeCTCTokenizer
    tokenizer=Wav2Vec2PhonemeCTCTokenizer.from_pretrained(str(MODEL),local_files_only=True,do_phonemize=False)
    vocab=json.loads((MODEL/'vocab.json').read_text(encoding='utf-8'));assert len(vocab)==392 and vocab['<pad>']==0
    engine=IPA();out={}
    for word,text in texts.items():
        phones=engine.tokens(text);missing=[p for p in phones if p not in vocab];ids=[vocab[p] for p in phones] if not missing else []
        if ids:
            check=tokenizer(' '.join(phones),add_special_tokens=False)['input_ids']
            assert check==ids,'Actual tokenizer disagrees with whole-phoneme IDs'
        out[int(word)]=dict(text=text,phones=phones,ids=ids,missing=missing)
    return out,dict(espeak_version=engine.version,tokenizer='Wav2Vec2PhonemeCTCTokenizer',do_phonemize=False,
      front='direct eSpeak-ng en-us C API,stressremoved,underscore wholephone separator;tokenizer validatesIPAsequence',vocab_size=len(vocab),blank_id=0)

def teacher_model():
    import torch
    from transformers import Wav2Vec2Config,Wav2Vec2ForCTC,Wav2Vec2FeatureExtractor
    path=MODEL/'pytorch_model.bin';assert hashlib.sha256(path.read_bytes()).hexdigest()==WEIGHT_SHA
    config=Wav2Vec2Config.from_pretrained(str(MODEL),local_files_only=True)
    config._attn_implementation='sdpa'
    with torch.device('meta'):model=Wav2Vec2ForCTC(config)
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    expected=model.state_dict();converted={}
    for key,value in state.items():
        new=key
        if key not in expected and key.endswith('.weight_g'):new=key[:-len('weight_g')]+'parametrizations.weight.original0'
        if key not in expected and key.endswith('.weight_v'):new=key[:-len('weight_v')]+'parametrizations.weight.original1'
        converted[new]=value
    model.load_state_dict(converted,strict=True,assign=True);model.eval();model.requires_grad_(False)
    assert not any(p.is_meta for p in model.parameters())
    extractor=Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL),local_files_only=True)
    receptive,stride=1,1
    for kernel,step in zip(config.conv_kernel,config.conv_stride):receptive+=(kernel-1)*stride;stride*=step
    return model,extractor,dict(revision=REVISION,weight_sha256=WEIGHT_SHA,weight_key_renames=[k for k in state if k not in expected],
      parameters=sum(p.numel() for p in model.parameters()),receptive_field_samples=receptive,stride_samples=stride,
      sampling_rate=16000,blank_id=config.pad_token_id,attention_implementation=config._attn_implementation,precision='float32',loading='torchweights_only,mmap,strictassign',frozen=True)

if __name__=='__main__':
    print(json.dumps(lexicon({7:'five',15:'nine',0:'Back',2:'Confirm'})[0],ensure_ascii=False))
