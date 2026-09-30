"""Typed Concept Bottleneck — vision state for a Kev decision model.

adapted from jaredpalmer/kev@5920c5f (Apache-2.0): kev/model.py encode()/rows_of()/forward_rows_batch() row form,
kev/train.py delta mode (Checkpoint.warm_start).
Smallest diff: the Kev decision model (LoRA + pointer head + temperature) is unchanged; the base's own vision tower
(Qwen3_5ForConditionalGeneration.model.visual, dropped by Kev's text-only loader) is attached frozen, and the image's
tokens are spliced into the state segment right after the <state> delimiter. Positions: channel 0 = sequential text
positions (what the text model derives from Kev's 2-D position_ids), channels 1-3 = Qwen3.5 M-RoPE for the state
(model.get_rope_index) and max(state position)+1.. for the branch. Every question runs as its own causal row (state + its
branch), exactly Kev's row form for hybrid backbones.
"""
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, Qwen3_5ForConditionalGeneration

from kev.checkpoint import Checkpoint, LoadOptions, Meta, resolve_run
from kev.model import DecisionModel, encode, load_tokenizer, rows_of

IMG_SIDE = 448  # fixed resize -> 28x28 patches of 16 -> 14x14 = 196 merged image tokens


def load_eval(run="jaredpalmer/kev-0.8b", device="cuda", temperature=None):
    """Released checkpoint, fp32 exact path, LoRA merged, eval mode (kev.checkpoint)."""
    ck = Checkpoint(run)
    tok, m = ck.load(device, LoadOptions(temperature=temperature))
    return tok, m, ck.meta


def load_trainable(run="jaredpalmer/kev-0.8b", device="cuda", checkpointing=True, warm=True):
    """kev.train delta mode: a fresh LoRA DecisionModel built like `run`, warm-started from its adapter + pointer head.
    warm=False (ablation A1): same architecture on the untrained base, LoRA and pointer head from scratch (no Kev training)."""
    ck = Checkpoint(run)
    meta = ck.meta
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    m = DecisionModel(meta.base, tok, device, lora=meta.lora, revision=meta.base_revision, head_dim=meta.head_dim)
    ours = Meta(base=meta.base, base_revision=meta.base_revision, lora=meta.lora, head_dim=meta.head_dim,
                option_isolation=False, special_embeddings=False, weights_dtype="fp32", holdout=[], weights="lora")
    src = ck.warm_start(m, ours) if warm else {"init_from": f"{meta.base}@{meta.base_revision} (no Kev warm start)", "tensors": 0}
    if checkpointing:
        m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.lm.config.use_cache = False
    m.head.temperature = 1.0
    return tok, m, meta, src


class VisionKev:
    def __init__(self, tok, m, meta, processor_repo="Qwen/Qwen3.5-0.8B", device="cuda"):
        self.tok, self.m, self.meta, self.device = tok, m, meta, device
        base_dir = resolve_run(f"{meta.base}@{meta.base_revision or ''}")
        full = Qwen3_5ForConditionalGeneration.from_pretrained(base_dir, dtype=torch.float32)
        self.visual = full.model.visual.to(device).eval().requires_grad_(False)
        self.mm = full.model                      # get_rope_index only (config-driven)
        self.mm.language_model = None             # drop the duplicate un-adapted LM; Kev's adapted one is self.m.lm
        self.cfg = full.config
        self.ip = AutoImageProcessor.from_pretrained(resolve_run(processor_repo))
        del full

    @classmethod
    def released(cls, run="jaredpalmer/kev-0.8b", device="cuda", temperature=None):
        return cls(*load_eval(run, device, temperature), device=device)

    # ---- encoding -------------------------------------------------------------------------------------------------
    def image_inputs(self, img):
        img = img.convert("RGB").resize((IMG_SIDE, IMG_SIDE), Image.BICUBIC)
        o = self.ip(images=[img], return_tensors="pt")
        return o["pixel_values"].to(self.device), o["image_grid_thw"].to(self.device)

    @torch.no_grad()
    def image_features(self, pixel_values, grid):
        """Frozen vision tower -> [n_tokens, d] merged image embeddings."""
        return torch.cat(self.mm.get_image_features(pixel_values, grid, return_dict=True).pooler_output, 0)

    def encode(self, rec, grid=None):
        """kev encode() of the text record, then n image tokens (+vision start/end) spliced after <state>."""
        enc = encode(self.tok, rec)
        if grid is None:
            return enc, 0
        n = int(grid.prod().item()) // self.visual.spatial_merge_size ** 2
        ins = [self.cfg.vision_start_token_id] + [self.cfg.image_token_id] * n + [self.cfg.vision_end_token_id]
        k = len(ins)
        enc = dict(enc)
        enc["ids"] = enc["ids"][:1] + ins + enc["ids"][1:]
        enc["seg"] = enc["seg"][:1] + [0] * k + enc["seg"][1:]
        enc["pos"] = enc["pos"][:1] + [0] * k + enc["pos"][1:]   # replaced by M-RoPE below
        enc["opt"] = enc["opt"][:1] + [-1] * k + enc["opt"][1:]
        enc["decide_idx"] = [d + k for d in enc["decide_idx"]]
        enc["opt_idx"] = [[o + k for o in oi] for oi in enc["opt_idx"]]
        return enc, n

    # ---- forward --------------------------------------------------------------------------------------------------
    def logits(self, rec, img_feats=None, grid=None):
        """-> list over questions of (logits [K], decide hidden [d]); one causal row per question, batched in one pass.
        Differentiable w.r.t. the LoRA and pointer head (vision features are given, frozen)."""
        enc, _ = self.encode(rec, grid)
        S, _, brs = rows_of(enc)
        emb = self.m.lm.get_input_embeddings()
        dev = self.device
        S_t = torch.tensor([S], device=dev)
        if grid is not None:
            pos_s, _ = self.mm.get_rope_index(S_t, mm_token_type_ids=(S_t == self.cfg.image_token_id).int(), image_grid_thw=grid)
        else:
            pos_s = torch.arange(len(S), device=dev).view(1, 1, -1).expand(3, 1, -1)
        nxt = int(pos_s.max().item()) + 1
        L = max(len(S) + len(r["ids"]) for r in brs)
        ids = torch.full((len(brs), L), self.m.pad_id, device=dev)
        pos = torch.zeros((4, len(brs), L), dtype=torch.long, device=dev)
        att = torch.zeros((len(brs), L), dtype=torch.long, device=dev)
        for i, r in enumerate(brs):
            n = len(S) + len(r["ids"])
            ids[i, :n] = torch.tensor(S + r["ids"], device=dev)
            pos[0, i, :n] = torch.arange(n, device=dev)
            pos[1:, i, :len(S)] = pos_s[:, 0]
            pos[1:, i, len(S):n] = (torch.arange(len(r["ids"]), device=dev) + nxt)[None]
            att[i, :n] = 1
        e = emb(ids)
        if img_feats is not None:
            mask = (ids == self.cfg.image_token_id).unsqueeze(-1)
            e = e.masked_scatter(mask, img_feats.to(e.dtype).repeat(len(brs), 1))
        h = self.m.lm(inputs_embeds=e, position_ids=pos, attention_mask=att).last_hidden_state.float()
        out = []
        for i, r in enumerate(brs):
            d = len(S) + r["decide"]; oi = torch.tensor([len(S) + o for o in r["opts"]], device=dev)
            out.append((self.m.head(h[i, d], h[i, oi]), h[i, d]))
        return out

    @torch.no_grad()
    def probs(self, rec, image=None, return_decide=False):
        feats = grid = None
        if image is not None:
            pv, grid = self.image_inputs(image)
            feats = self.image_features(pv, grid)
        out = self.logits(rec, feats, grid)
        ps = [F.softmax(z, -1).cpu() for z, _ in out]
        return (ps, [h.cpu() for _, h in out]) if return_decide else ps
