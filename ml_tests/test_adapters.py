"""Optional adapter integration tests using tiny LOCAL fixtures, not pretrained models.

Run after installing requirements-ml.txt: python -m pytest ml_tests -q
These tests do not measure pretrained BGE/Qwen accuracy or performance.
"""
import pytest
pytest.importorskip('torch')
pytest.importorskip('sentence_transformers')

from enterprise.engine import Retriever, Generator


def test_sentence_transformer_and_qdrant_fusion(tmp_path, monkeypatch):
    from sentence_transformers import SentenceTransformer, models
    model=SentenceTransformer(modules=[models.BoW(['remote','maximum','days','backup','retention'])],device='cpu')
    location=tmp_path/'local-embedding';model.save(str(location))
    monkeypatch.setenv('EVOLLM_EMBEDDING_MODEL',str(location))
    retriever=Retriever('bge')
    corpus=[{'id':'a','text':'remote maximum days','meta':{'title':'Remote policy','family':'remote'}},
            {'id':'b','text':'backup retention','meta':{'title':'Backup policy','family':'backup'}}]
    vector,hit=retriever.vector('remote maximum',True)
    assert not hit
    ranked=retriever.search('remote maximum',corpus,2,vector)
    assert ranked[0]['id']=='a'
    assert retriever.vector('remote maximum',True)[1]


def test_local_qwen_loading_and_invalid_citation_abstention(tmp_path, monkeypatch):
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    import torch
    torch.set_num_threads(1)
    vocab={'[UNK]':0,'[PAD]':1,'[EOS]':2,'remote':3,'days':4,'policy':5}
    tok=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));tok.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
    tokenizer.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}assistant: "
    cfg=Qwen2Config(vocab_size=len(vocab),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                   num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=2048,
                   bos_token_id=2,eos_token_id=2,pad_token_id=1)
    model=Qwen2ForCausalLM(cfg)
    folder=tmp_path/'tiny-qwen';model.save_pretrained(folder);tokenizer.save_pretrained(folder)
    monkeypatch.setenv('EVOLLM_LLM_MODEL',str(folder))
    g=Generator()
    streamed=[]
    answer,citations=g.generate('remote days',[{'id':'a#C1','text':'remote days','meta':{'version':'1','issuer':'HR'}}],None,'qwen',on_token=streamed.append)
    assert 'valid source citations' in answer and citations==[]
    assert streamed, 'Real generation did not call the streaming adapter'
