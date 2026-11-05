import torch
import torch.nn as nn


class WGANLoss(nn.Module):
    """
    标准 WGAN 损失（无 abs）。
    判别器最小化：E_fake - E_real
    生成器最小化：- E_fake
    """
    def __init__(self):
        super(WGANLoss, self).__init__()

    def discriminator_loss(self, real_scores, fake_scores):
        # D 惩罚生成样本得分高、奖励真实样本得分高
        return fake_scores.mean() - real_scores.mean()

    def generator_loss(self, fake_scores):
        # G 希望让假样本得分尽可能大 → 最小化 -mean(fake)
        return -fake_scores.mean()


def reconstruction_loss(real_data, fake_data):
    l1_loss = nn.L1Loss()
    return l1_loss(real_data, fake_data)


def cycle_consistency_loss(real_data, cycled_data):
    l1_loss = nn.L1Loss()
    return l1_loss(real_data, cycled_data)


def identity_loss(real_data, identity_mapped_data):
    l1_loss = nn.L1Loss()
    return l1_loss(real_data, identity_mapped_data)


def generator_loss(disc_fake_scores, real_data, fake_data, cycled_data,
                   identity_data=None, lambda_adv=1.0, lambda_rec=10.0,
                   lambda_cycle=10.0, lambda_identity=1.0):
    """
    综合生成器损失 = 对抗 + 重建 + 循环一致性 + 身份映射
    对抗项使用上面修正后的 WGAN 形式。
    """
    wgan = WGANLoss()

    # 对抗损失（已修正符号）
    adv_loss = wgan.generator_loss(disc_fake_scores)

    # 重建/循环/身份
    rec_loss = reconstruction_loss(real_data, fake_data)
    cyc_loss = cycle_consistency_loss(real_data, cycled_data)

    id_loss = torch.tensor(0.0, device=real_data.device)
    if lambda_identity > 0 and identity_data is not None:
        id_loss = identity_loss(real_data, identity_data)

    total_loss = lambda_adv * adv_loss + lambda_rec * rec_loss + \
                 lambda_cycle * cyc_loss + lambda_identity * id_loss

    return total_loss, adv_loss, rec_loss, cyc_loss, id_loss


