import torch
import numpy as np
from dataclasses import dataclass
import transformers

@dataclass
class BertConfig:
    vocab_size: int = 30522
    emb_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ff_dim: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout: float = 0.1
    attn_dropout: float = 0.1
    max_seq_len: int = 512
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

bert_base_conf = lambda max_seq_len: BertConfig(max_seq_len=max_seq_len)

bert_large_conf = lambda max_seq_len: BertConfig(
    emb_dim=1024,
    num_layers=24,
    num_heads=16,
    ff_dim=4096,
    max_seq_len=max_seq_len
)

class BertTransformerLayer(torch.nn.Module):
    def __init__(self, cfg : BertConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = self.cfg.emb_dim // self.cfg.num_heads

        assert self.cfg.num_heads * self.head_dim == self.cfg.emb_dim

        self.query = torch.nn.Linear(self.cfg.emb_dim, self.cfg.num_heads * self.head_dim)
        self.key = torch.nn.Linear(self.cfg.emb_dim, self.cfg.num_heads * self.head_dim)
        self.value = torch.nn.Linear(self.cfg.emb_dim, self.cfg.num_heads * self.head_dim)
        self.attn_drop = torch.nn.Dropout(self.cfg.attn_dropout)
        self.sf = np.sqrt(self.head_dim)

        self.y0w0 = torch.nn.Linear(self.cfg.emb_dim, self.cfg.emb_dim)
        self.drop0 = torch.nn.Dropout(cfg.hidden_dropout)
        self.ln0 = torch.nn.LayerNorm(cfg.emb_dim, eps=cfg.layer_norm_eps)

        self.y1w1 = torch.nn.Linear(cfg.emb_dim, cfg.ff_dim)
        self.gelu = torch.nn.GELU()

        self.y2w2 = torch.nn.Linear(cfg.ff_dim, cfg.emb_dim)
        self.drop2 = torch.nn.Dropout(cfg.hidden_dropout)
        self.ln2 = torch.nn.LayerNorm(cfg.emb_dim, eps=cfg.layer_norm_eps)

    def forward(self, x : torch.Tensor):
        # X: [N, S, E]
        batch_size, seq_len, _emb_dim = x.size()

        # W: [E, H*D], B: [H*D]
        # X*W: [N, S, H*D] -> [N, S, H, D]
        # -> Permute(0, 2, 1, 3): [N, H, S, D]
        query_out = self.query(x) \
            .reshape(batch_size, seq_len, self.cfg.num_heads, self.head_dim) \
            .permute(0, 2, 1, 3)

        # W: [E, H*D], B: [H*D]
        # X*W: [N, S, H*D] -> [N, S, H, D]
        # -> Permute(0, 2, 3, 1): [N, H, D, S]
        key_out = self.key(x) \
            .reshape(batch_size, seq_len, self.cfg.num_heads, self.head_dim) \
            .permute(0, 2, 3, 1)

        # W: [E, H*D], B: [H*D]
        # X*W: [N, S, H*D] -> [N, S, H, D]
        # -> Permute(0, 2, 1, 3): [N, H, S, D]
        value_out = self.value(x) \
            .reshape(batch_size, seq_len, self.cfg.num_heads, self.head_dim) \
            .permute(0, 2, 1, 3)

        # attn_probs: [N*H, S, S]
        attn_probs = self.attn_drop(torch.softmax(
            torch.matmul(query_out, key_out) / self.sf, dim=-1))

        # attn_out: [N, S, H*D]
        attn_out = torch.matmul(attn_probs, value_out) \
            .reshape(batch_size, self.cfg.num_heads, seq_len, self.head_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, -1)

        y0w0_out = self.ln0(self.drop0(self.y0w0(attn_out)) + x)
        y1w1_out = self.gelu(self.y1w1(y0w0_out))
        return self.ln2(self.drop2(self.y2w2(y1w1_out)) + y0w0_out)

class BertEmbedding(torch.nn.Module):
    def __init__(self, cfg : BertConfig):
        super().__init__()
        self.word_embed = torch.nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=0)
        self.pos_embed = torch.nn.Embedding(cfg.max_seq_len, cfg.emb_dim)
        self.seg_embed = torch.nn.Embedding(2, cfg.emb_dim)
        self.ln = torch.nn.LayerNorm(cfg.emb_dim, eps=cfg.layer_norm_eps)
        self.drop = torch.nn.Dropout(cfg.hidden_dropout)
        self.position_ids = torch.arange(cfg.max_seq_len).expand((1, -1))

    def forward(self, input_ids, seg_ids=None):
        [_, seq_len] = input_ids.size()
        position_ids = torch.arange(seq_len).expand(1, -1)

        inputs_embeds = self.word_embed(input_ids)
        position_embeddings = self.pos_embed(position_ids)

        if seg_ids is None: seg_ids = torch.zeros_like(input_ids, dtype=torch.long)
        token_type_embeddings = self.seg_embed(seg_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        return self.drop(self.ln(embeddings))

class Bert(torch.nn.Module):
    def __init__(self, cfg : BertConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = BertEmbedding(cfg)
        self.layers = torch.nn.Sequential(*[
            BertTransformerLayer(cfg) for _ in range(cfg.num_layers)
        ])

    def forward(self, input_ids, seg_ids=None):
        ix = self.embed(input_ids, seg_ids)
        return self.layers(ix)


class BertSquad(torch.nn.Module):
    def __init__(self, cfg : BertConfig):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            'bert-large-uncased-whole-word-masking-finetuned-squad')
        self.bert = Bert(cfg)
        self.qa = torch.nn.Linear(cfg.emb_dim, 2)

    def forward(self, input_ids, seg_ids):
        return self.qa(self.bert(input_ids, seg_ids))


    def query(self, context_str, query_str):
        tokens = self.tokenizer(query_str, context_str, return_tensors='pt')
        print(tokens['input_ids'].shape)
        print(tokens['token_type_ids'].shape)
        bert_out = self.bert.forward(
            tokens['input_ids'],
            tokens['token_type_ids'])
        qa_out = self.qa(bert_out)
        ans_start = qa_out[0, :, 0].argmax()
        ans_end = qa_out[0, :, 1].argmax()
        ans_toks = tokens['input_ids'].numpy()[0][ans_start:ans_end + 1]
        return self.tokenizer.decode(ans_toks)


