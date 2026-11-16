"""ACC-only enrollment-conditioned verifier; Audio supplies offline score targets.

Inspired by arXiv:2606.16115; this is a small cross-modal adaptation,
not the authors' hybrid TD/TI enrollment implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ACCEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for ci, co in [(3, 32), (32, 64), (64, 128)]:
            blocks += [nn.Conv2d(ci, co, 3, stride=(2, 1), padding=1),
                       nn.GroupNorm(8, co), nn.GELU()]
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x).mean(2).transpose(1, 2)


class Verifier(nn.Module):
    def __init__(self, local=True):
        super().__init__()
        self.local = local
        self.attention = nn.MultiheadAttention(128, 4, dropout=.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(515 if local else 3, 64),
                                  nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, query, enrollment, base_logit, bound=1.):
        # query B,T,C; enrollment B,K,T,C. No enrollment positional encoding.
        b, k, t, c = enrollment.shape
        q = query[:, None].expand(-1, k, -1, -1).reshape(b*k, -1, c)
        r = enrollment.reshape(b*k, t, c)
        pair_cos = F.cosine_similarity(q.mean(1), r.mean(1)).reshape(b, k)
        global_features = torch.stack([pair_cos.mean(1),
                                       pair_cos.std(1, unbiased=False), base_logit], 1)
        if self.local:
            uq = self.attention(q, r, r, need_weights=False)[0].mean(1)
            ur = self.attention(r, q, q, need_weights=False)[0].mean(1)
            pairs = torch.cat([uq, ur], 1).reshape(b, k, 256)
            features = torch.cat([global_features, pairs.mean(1),
                                   pairs.std(1, unbiased=False)], 1)
        else:
            features = global_features
        correction = bound * torch.tanh(self.head(features).squeeze(1))
        return base_logit + correction, correction


def objective(logit, correction, labels, teacher_logit=None,
              kd_weight=.1, regularization=.01, temperature=2.):
    loss = F.binary_cross_entropy_with_logits(logit, labels.float())
    if teacher_logit is not None:
        # Bernoulli KL, teacher detached; unlike multiclass TRKD.
        p = torch.sigmoid(teacher_logit.detach()/temperature).clamp(1e-6, 1-1e-6)
        ls = F.logsigmoid(logit/temperature)
        lns = F.logsigmoid(-logit/temperature)
        kl = (p*(p.log()-ls)+(1-p)*((1-p).log()-lns)).mean()
        loss = loss + kd_weight*temperature**2*kl
    return loss + regularization*correction.square().mean()
