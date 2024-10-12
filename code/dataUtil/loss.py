import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = None

    def forward(self, logits, targets):
        if self.weights is not None:
            self.weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=self.weights)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else torch.tensor([1.0, 1.0, 1.0])
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
        loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        
        intersection = preds * targets_one_hot
        intersection = intersection.sum(dim=0)
        
        cardinality = preds.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha, beta, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        
        true_positives = (preds * targets_one_hot).sum(dim=0)
        false_positives = (preds * (1 - targets_one_hot)).sum(dim=0)
        false_negatives = ((1 - preds) * targets_one_hot).sum(dim=0)
        
        tversky_index = (true_positives + self.smooth) / (
            true_positives + self.alpha * false_positives + self.beta * false_negatives + self.smooth)
        
        tversky_loss = 1 - tversky_index
        return tversky_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        """
        Label smoothing loss constructor.

        :param smoothing: The amount of label smoothing to apply. Default is 0.1.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        """
        Forward pass for label smoothing loss.

        :param inputs: Logits from the model (before softmax).
        :param targets: Ground truth labels (integer indices, not one-hot encoded).
        :return: The label smoothed loss value.
        """
        n_classes = inputs.size(1)
        # Apply log softmax to inputs
        log_preds = F.log_softmax(inputs, dim=1)

        # Create one-hot encoding of the targets
        with torch.no_grad():
            targets_one_hot = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
            # Apply label smoothing
            targets_one_hot = (1 - self.smoothing) * targets_one_hot + self.smoothing / n_classes
        
        # Compute the loss
        loss = (-targets_one_hot * log_preds).sum(dim=1).mean()
        return loss

if __name__ == "__main__":
    # 使用示例
    
    # 初始化随机测试数据
    # logits = torch.randn(10, 3, requires_grad=True)
    # targets = torch.tensor([0, 1, 2, 1, 1, 0, 2, 2, 0, 1], dtype=torch.long)
    
    # weights = [1.0, 2.0, 3.0]
    # criterion = WeightedCrossEntropyLoss(weights=weights)

    # loss = criterion(logits, targets)
    # print("Weighted Cross-Entropy Loss:", loss.item())

    # alpha = torch.tensor([0.4215, 0.3701, 0.2084])
    # criterion = FocalLoss(alpha=alpha, gamma=2.0)

    # loss = criterion(logits, targets)
    # print("Focal Loss:", loss.item())


    # criterion = DiceLoss()

    # loss = criterion(logits, targets)
    # print("Dice Loss:", loss.item())

    # criterion = TverskyLoss(alpha=0.2083, beta=0.7917)

    # loss = criterion(logits, targets)
    # print("Tversky Loss:", loss.item())
    
    # 假设我们有一个五分类任务
    num_classes = 5
    batch_size = 4

    # 创建模型输入（logits）和目标标签
    inputs = torch.randn(batch_size, num_classes, requires_grad=True)  # 模型输出的 logits
    targets = torch.tensor([0, 2, 1, 4])  # ground truth 标签，类别为 [0, 2, 1, 4]

    # 使用标签平滑交叉熵损失
    smoothing = 0.1  # 设置平滑参数
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    # 计算损失
    loss = criterion(inputs, targets)
    
    # 打印损失值
    print("Loss:", loss.item())

    # 反向传播
    loss.backward()