"""Cross-word supervised contrast and row-distribution relation KL."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from crossword_neural_verifier import ACCEncoder


class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=ACCEncoder()
        self.projection=nn.Linear(256,128)

    def forward(self,x):
        h=self.backbone(x)
        pooled=torch.cat([h.mean(1),h.std(1,unbiased=False)],1)
        return F.normalize(self.projection(pooled),dim=1)


def masks(speaker,word):
    same=speaker[:,None].eq(speaker[None,:])
    sameword=word[:,None].eq(word[None,:])
    positive=same&~sameword
    negative=~same
    valid=positive|negative
    assert positive.any(1).all()
    return positive,negative,sameword,valid


def crossword_supcon(z,speaker,word,temperature=.1):
    z=F.normalize(z,dim=1)
    positive,negative,sameword,valid=masks(speaker,word)
    logits=z@z.T/temperature+(negative&sameword).float()*math.log(2.)
    logprob=F.log_softmax(logits.masked_fill(~valid,-1e4),dim=1)
    return -((logprob*positive).sum(1)/positive.sum(1)).mean()


def relation_kl(z,teacher,speaker,word,temperature=.2):
    z=F.normalize(z,dim=1)
    teacher=F.normalize(teacher.detach(),dim=1)
    _,_,_,valid=masks(speaker,word)
    student_logits=(z@z.T/temperature).masked_fill(~valid,-1e4)
    teacher_logits=(teacher@teacher.T/temperature).masked_fill(~valid,-1e4)
    return F.kl_div(F.log_softmax(student_logits,dim=1),
                    F.softmax(teacher_logits,dim=1),reduction='batchmean')
