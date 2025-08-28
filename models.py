# models.py  -- Steps 1..3 applied
from torch import Tensor
import torch
from torch import nn
import torch.utils.checkpoint
import contextlib
import torchvision
from einops import rearrange

import math
from stgcn_layers import Graph, get_stgcn_chain
from deformable_attention_2d import DeformableAttention2D
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import warnings
from config import mt5_path

# -------------------------
# init helpers
# -------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is >2 std from [a, b] in trunc_normal_.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# -------------------------
# main model
# -------------------------
class Uni_Sign(nn.Module):
    def __init__(self, args):
        super(Uni_Sign, self).__init__()
        self.args = args

        # parts and dims
        self.modes = ['body', 'left', 'right', 'face_all']
        self.per_part_dim = 256               # ST-GCN final dim from your chain
        self.concat_dim = self.per_part_dim * len(self.modes)

        # ---------- Graph + per-part ST-GCN encoders ----------
        self.graph, A = {}, []
        self.proj_linear = nn.ModuleDict()
        hidden_dim_for_rgb = args.hidden_dim  # used by RGB fusion blocks
        for mode in self.modes:
            self.graph[mode] = Graph(layout=f'{mode}', strategy='distance', max_hop=1)
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False))
            # (x, y, score) -> 64
            self.proj_linear[mode] = nn.Linear(3, 64)

        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        spatial_kernel_size = A[0].size(0)
        for index, mode in enumerate(self.modes):
            self.gcn_modules[mode], _ = get_stgcn_chain(
                64, 'spatial', (1, spatial_kernel_size), A[index].clone(), True
            )
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(
                self.per_part_dim, 'temporal', (5, spatial_kernel_size), A[index].clone(), True
            )

        # NOTE (Step 1): DO NOT tie left/right weights. We intentionally
        # remove lines like:
        #   self.gcn_modules['left'] = self.gcn_modules['right']
        #   self.fusion_gcn_modules['left'] = self.fusion_gcn_modules['right']
        #   self.proj_linear['left'] = self.proj_linear['right']

        # ---------- Step 2: Per-part temporal Transformer (over time) ----------
        # small 2-layer encoder; shared hyperparams across parts
        def make_temporal_tx():
            layer = nn.TransformerEncoderLayer(
                d_model=self.per_part_dim, nhead=8, dim_feedforward=1024,
                dropout=0.1, batch_first=True, norm_first=True
            )
            return nn.TransformerEncoder(layer, num_layers=2)
        self.temporal_tx = nn.ModuleDict({m: make_temporal_tx() for m in self.modes})
        # learned 1D pos embedding along time (shared across parts)
        self.temporal_pos = nn.Parameter(torch.zeros(args.max_length, self.per_part_dim))
        trunc_normal_(self.temporal_pos, std=0.02)

        # ---------- Step 3: Cross-part transformer (4 tokens per time step) ----------
        # learnable part embeddings (body/left/right/face)
        self.part_embed = nn.Embedding(len(self.modes), self.per_part_dim)
        trunc_normal_(self.part_embed.weight, std=0.02)

        cross_layer = nn.TransformerEncoderLayer(
            d_model=self.per_part_dim, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.cross_part_tx = nn.TransformerEncoder(cross_layer, num_layers=1)

        # a small bias after concat (kept for compatibility with your code)
        self.part_para = nn.Parameter(torch.zeros(self.concat_dim))

        # project pose tokens into MT5 hidden size (=768 in your setup)
        self.pose_proj = nn.Linear(self.concat_dim, 768)

        # ---------- Language selection ----------
        if "CSL" in self.args.dataset:
            self.lang = 'Chinese'
        else:
            self.lang = 'English'

        # ---------- Optional RGB support path (unchanged) ----------
        self.rgb_support = getattr(self.args, "rgb_support", False)
        if self.rgb_support:
            self.rgb_support_backbone = torch.nn.Sequential(
                *list(torchvision.models.efficientnet_b0(pretrained=True).children())[:-2]
            )
            self.rgb_proj = nn.Conv2d(1280, hidden_dim_for_rgb, kernel_size=1)
            self.fusion_pose_rgb_linear = nn.Linear(hidden_dim_for_rgb, hidden_dim_for_rgb)
            self.fusion_pose_rgb_DA = DeformableAttention2D(
                dim=hidden_dim_for_rgb, dim_head=32, heads=8, dropout=0.,
                downsample_factor=1, offset_scale=None, offset_groups=None, offset_kernel_size=1
            )
            self.fusion_gate = nn.Sequential(
                nn.Conv1d(hidden_dim_for_rgb * 2, hidden_dim_for_rgb, 1),
                nn.GELU(),
                nn.Conv1d(hidden_dim_for_rgb, 1, 1),
                nn.Tanh(),
                nn.ReLU(),
            )
            for layer in self.fusion_gate:
                if isinstance(layer, nn.Conv1d):
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)

        # ---------- MT5 ----------
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)

        # init linears / norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def maybe_autocast(self, dtype=torch.float32):
        return torch.cuda.amp.autocast(dtype=dtype) if True else contextlib.nullcontext()

    # ---------------- RGB fusion (unchanged) ----------------
    def gather_feat_pose_rgb(self, gcn_feat, rgb_feat, indices, rgb_len, pose_init):
        b, c, T, n = gcn_feat.shape
        assert rgb_feat.shape[0] == indices.shape[0]
        rgb_feat = self.rgb_proj(rgb_feat)

        assert len(rgb_len) == b
        start = 0
        for batch in range(b):
            index = indices[start:start + rgb_len[batch]].to(torch.long)
            if rgb_len[batch] == 1 and -1 in index:
                start = start + rgb_len[batch]; continue

            gcn_feat_selected  = gcn_feat[batch, :, index]
            rgb_feat_selected  = rgb_feat[start:start + rgb_len[batch]]
            pose_init_selected = pose_init[start:start + rgb_len[batch]]

            gcn_feat_selected  = rearrange(gcn_feat_selected,  'c t n -> t c n')
            pose_init_selected = rearrange(pose_init_selected, 't n c -> t c n')

            with self.maybe_autocast():
                fused_transposed = self.fusion_pose_rgb_DA(
                    pose_feat=gcn_feat_selected,
                    rgb_feat=rgb_feat_selected,
                    pose_init=pose_init_selected,
                )
            fused_transposed = fused_transposed.to(gcn_feat.dtype)
            gate_feature = torch.concat([fused_transposed, gcn_feat_selected], dim=-2)
            gate_score = self.fusion_gate(gate_feature)
            fused_transposed_post = (gate_score) * fused_transposed + (1 - gate_score) * gcn_feat_selected

            gcn_feat = gcn_feat.clone()
            fused_transposed_post = rearrange(fused_transposed_post, 't c n -> c t n')
            gcn_feat[batch, :, index] = fused_transposed_post
            start = start + rgb_len[batch]

        assert start == rgb_feat.shape[0]
        return gcn_feat

    # ---------------- forward ----------------
    def forward(self, src_input, tgt_input):
        # ---- RGB branch ----
        if self.rgb_support:
            rgb_support_dict = {}
            for index_key, rgb_key in zip(['left_sampled_indices', 'right_sampled_indices'],
                                          ['left_hands', 'right_hands']):
                rgb_feat = self.rgb_support_backbone(src_input[rgb_key])
                rgb_support_dict[index_key] = src_input[index_key]
                rgb_support_dict[rgb_key] = rgb_feat

        # ---- Pose branch ----
        per_part_tokens = []  # list of (B, T, 256)

        body_feat = None
        for part in self.modes:
            # (B,T,V,3) -> (B,64,T,V)
            proj_feat = self.proj_linear[part](src_input[part]).permute(0, 3, 1, 2)

            # spatial GCN
            gcn_feat = self.gcn_modules[part](proj_feat)  # (B,256,T,V)

            if part == 'body':
                body_feat = gcn_feat
            else:
                assert body_feat is not None
                if part == 'left':
                    if self.rgb_support:
                        gcn_feat = self.gather_feat_pose_rgb(
                            gcn_feat,
                            rgb_support_dict[f'{part}_hands'],
                            rgb_support_dict[f'{part}_sampled_indices'],
                            src_input[f'{part}_rgb_len'],
                            src_input[f'{part}_skeletons_norm'],
                        )
                    gcn_feat = gcn_feat + body_feat[..., -2][..., None].detach()
                elif part == 'right':
                    if self.rgb_support:
                        gcn_feat = self.gather_feat_pose_rgb(
                            gcn_feat,
                            rgb_support_dict[f'{part}_hands'],
                            rgb_support_dict[f'{part}_sampled_indices'],
                            src_input[f'{part}_rgb_len'],
                            src_input[f'{part}_skeletons_norm'],
                        )
                    gcn_feat = gcn_feat + body_feat[..., -1][..., None].detach()
                elif part == 'face_all':
                    gcn_feat = gcn_feat + body_feat[..., 0][..., None].detach()
                else:
                    raise NotImplementedError

            # temporal GCN
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)   # (B,256,T,V)

            # pool over joints -> (B,T,256)
            tokens = gcn_feat.mean(-1).transpose(1, 2)

            # ----- Step 2: per-part temporal transformer over time -----
            T = tokens.shape[1]
            pos = self.temporal_pos[:T].unsqueeze(0)             # (1,T,256)
            tokens = tokens + pos
            tokens = self.temporal_tx[part](tokens)              # (B,T,256)

            per_part_tokens.append(tokens)

        # ----- Step 3: cross-part transformer per time step -----
        # stack parts -> (B,T,4,256)
        feats = torch.stack(per_part_tokens, dim=2)

        # add learnable part embeddings
        part_ids = torch.arange(len(self.modes), device=feats.device)
        part_emb = self.part_embed(part_ids)[None, None, :, :]   # (1,1,4,256)
        feats = feats + part_emb

        # run tiny transformer across the 4 part tokens for each (B,T)
        B, T, P, C = feats.shape
        feats_bt = feats.reshape(B * T, P, C)                    # (B*T,4,256)
        feats_bt = self.cross_part_tx(feats_bt)                  # (B*T,4,256)
        feats = feats_bt.reshape(B, T, P, C).reshape(B, T, P * C)  # (B,T,1024)

        # optional learned bias (kept from original code)
        feats = feats + self.part_para

        # project into MT5 hidden size
        inputs_embeds = self.pose_proj(feats)                    # (B,T,768)

        # ---- MT5 prefix + attention mask ----
        prefix_token = self.mt5_tokenizer(
            [f"Translate sign language video to {self.lang}: "] * len(tgt_input["gt_sentence"]),
            padding="longest", truncation=True, return_tensors="pt",
        ).to(inputs_embeds.device)

        prefix_embeds = self.mt5_model.encoder.embed_tokens(prefix_token['input_ids'])
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        attention_mask = torch.cat(
            [prefix_token['attention_mask'], src_input['attention_mask']], dim=1
        )

        tgt_tok = self.mt5_tokenizer(
            tgt_input['gt_sentence'], return_tensors="pt",
            padding=True, truncation=True, max_length=50
        )
        labels = tgt_tok['input_ids']
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100

        out = self.mt5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels.to(inputs_embeds.device),
            return_dict=True,
        )

        label = labels.reshape(-1)
        out_logits = out['logits']
        logits = out_logits.reshape(-1, out_logits.shape[-1])
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing, ignore_index=-100)
        loss = loss_fct(logits, label.to(out_logits.device, non_blocking=True))

        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'loss': loss,
        }

    @torch.no_grad()
    def generate(self, pre_compute_item, max_new_tokens, num_beams):
        inputs_embeds = pre_compute_item['inputs_embeds']
        attention_mask = pre_compute_item['attention_mask']
        out = self.mt5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return out

# unchanged
def get_requires_grad_dict(model):
    param_requires_grad = {name: True for name, param in model.named_parameters()}
    param_requires_grad_right = {}
    for key in param_requires_grad.keys():
        if 'left' in key:
            param_requires_grad_right[key.replace("left", 'right')] = param_requires_grad[key]
    param_requires_grad = {**param_requires_grad, **param_requires_grad_right}
    params_to_update = {k: v for k, v in model.state_dict().items() if param_requires_grad.get(k, True)}
    return params_to_update
