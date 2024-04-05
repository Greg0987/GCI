import torch.nn as nn
import torch.nn.functional as F

# ----------------
# 双向交叉熵
def BKLD_loss(student, teacher, **kwargs):
    loss = F.binary_cross_entropy_with_logits(student, teacher.softmax(1))
    return loss

# ----------------
# 距离损失

# 余弦相似度
def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)
# 皮尔森相关性
def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)
# 类内相关性
def inter_class_relation(y_stu, y_tea):
    return 1- pearson_correlation(y_stu, y_tea).mean()
# 类间相关性
def intra_class_relation(y_stu, y_tea):
    return inter_class_relation(y_stu.transpose(0, 1), y_tea.transpose(0, 1))

# 距离损失
def DIST_loss(student, teacher, tau=1.0, beta=1.0, gamma=1.0, **kwargs):
    y_stu = (student / tau).softmax(1)
    y_tea = (teacher / tau).softmax(1)
    inter_loss = tau ** 2 * inter_class_relation(y_stu, y_tea)
    intra_loss = tau ** 2 * intra_class_relation(y_stu, y_tea)
    loss = beta * inter_loss + gamma * intra_loss
    return loss


# ----------------
# KLD损失
def KLD_loss(student, teacher, temp=1.0, **kwargs):
    log_pred_student = F.log_softmax(student / temp, dim=1)
    pred_teacher = F.softmax(teacher / temp, dim=1)
    loss = F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1).mean()
    loss = loss * temp * temp
    return loss

# ---------------
# 交叉熵损失
def CE_loss(student, y):
    loss = F.cross_entropy(student, y)
    return loss

# ---------------
# 均方差损失
def MSE_loss(student, y):
    loss = F.mse_loss(student, y)
    return loss