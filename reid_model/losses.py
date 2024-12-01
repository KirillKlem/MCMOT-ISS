import random

import torch
from torch import nn
from torch.nn import functional as F



#Triplet loss with exponent
class TripletLossWithExp(nn.Module):
    def __init__(self):
        """
        Triplet Loss через экспоненту.
        """
        super(TripletLossWithExp, self).__init__()

    def forward(self, features, labels):
        """
        Вычисление Triplet Loss через экспоненту.
        features: torch.Tensor, эмбеддинги (размер: batch_size x embedding_dim).
        labels: torch.Tensor, метки классов (размер: batch_size).
        """
        triplets = self.sample_triplets(features, labels)
        if not triplets:
            # Если нет подходящих триплетов, возвращаем 0
            return torch.tensor(0.0, requires_grad=True, device=features.device)

        # Разделяем триплеты на якоря, положительные и отрицательные примеры
        anchors, positives, negatives = zip(*triplets)
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        # Вычисляем расстояния между эмбеддингами
        positive_distances = F.pairwise_distance(anchors, positives, p=2)
        negative_distances = F.pairwise_distance(anchors, negatives, p=2)

        # Вычисляем Triplet Loss с использованием экспоненты
        loss = torch.log(1 + torch.exp(positive_distances - negative_distances))
        return loss.mean()

    def sample_triplets(self, features, labels):
        """
        Генерация триплетов (anchor, positive, negative).
        features: torch.Tensor, эмбеддинги (размер: batch_size x embedding_dim).
        labels: torch.Tensor, метки классов (размер: batch_size).
        """
        triplets = []
        label_to_indices = {}

        # Сопоставляем метки с их индексами
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label.item(), []).append(idx)

        # Генерируем триплеты
        for label, indices in label_to_indices.items():
            if len(indices) < 2:
                continue  # Пропускаем, если недостаточно положительных примеров

            negative_labels = [l for l in label_to_indices.keys() if l != label]
            if not negative_labels:
                continue  # Пропускаем, если нет отрицательных классов

            for anchor_idx in indices:
                for positive_idx in indices:
                    if anchor_idx == positive_idx:
                        continue

                    # Выбираем случайный отрицательный пример из другого класса
                    negative_label = random.choice(negative_labels)
                    negative_idx = random.choice(label_to_indices[negative_label])

                    # Добавляем триплет (anchor, positive, negative)
                    triplets.append(
                        (features[anchor_idx], features[positive_idx], features[negative_idx])
                    )

        return triplets
    

# Triplet loss with dynamic temperature 
class TripletLossWithExpUpd(nn.Module):
    def __init__(self, temperature=0.3):
        """
        Triplet Loss через экспоненту.
        """
        super(TripletLossWithExpUpd, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, epoch):
        """
        Вычисление Triplet Loss через экспоненту.
        features: torch.Tensor, эмбеддинги (размер: batch_size x embedding_dim).
        labels: torch.Tensor, метки классов (размер: batch_size).
        """
        triplets = self.sample_triplets(features, labels)
        if not triplets:
            # Если нет подходящих триплетов, возвращаем 0
            return torch.tensor(0.0, requires_grad=True, device=features.device)

        # Разделяем триплеты на якоря, положительные и отрицательные примеры
        anchors, positives, negatives = zip(*triplets)
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        # Вычисляем расстояния между эмбеддингами
        positive_distances = F.pairwise_distance(anchors, positives, p=2)
        negative_distances = F.pairwise_distance(anchors, negatives, p=2)

        # Вычисляем Triplet Loss с использованием экспоненты
        loss = torch.log(1 + torch.exp((positive_distances - negative_distances)/min((self.temperature + 0.04 * epoch), 1)))
        return loss.mean()

    def sample_triplets(self, features, labels):
        triplets = []
        label_to_indices = {}

        # Сопоставляем метки с их индексами
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label.item(), []).append(idx)

        features = F.normalize(features, p=2, dim=1)  # Нормализуем эмбеддинги

        # Генерируем триплеты
        for label, indices in label_to_indices.items():
            if len(indices) < 2:
                continue  # Пропускаем, если недостаточно положительных примеров

            negative_indices = [idx for l, idxs in label_to_indices.items() if l != label for idx in idxs]
            if not negative_indices:
                continue  # Пропускаем, если нет отрицательных примеров

            for anchor_idx in indices:
                positive_indices = [idx for idx in indices if idx != anchor_idx]
                if not positive_indices:
                    continue

                positive_idx = random.choice(positive_indices)
                anchor_feat = features[anchor_idx]
                positive_feat = features[positive_idx]

                # Находим трудный отрицательный пример
                negative_feats = features[negative_indices]
                distances = (anchor_feat - negative_feats).pow(2).sum(1)
                hardest_negative_idx = negative_indices[distances.argmin().item()]

                triplets.append(
                    (features[anchor_idx], features[positive_idx], features[hardest_negative_idx])
                )

        return triplets


# Default centroid-loss
class CentroidLossExcludingSelf(nn.Module):
    def __init__(self, embedding_dim, device, weight=5e-4):
        """
        Centroid Loss с исключением текущего образца при вычислении центроида.
        Args:
            embedding_dim (int): Размерность эмбеддингов.
            device (torch.device): Устройство (CPU или GPU).
            weight (float): Вес для Centroid Loss.
        """
        super(CentroidLossExcludingSelf, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.weight = weight

    def forward(self, features, labels):
        """
        Вычисление Centroid Loss с исключением текущего образца.
        Args:
            features (torch.Tensor): Эмбеддинги (batch_size x embedding_dim).
            labels (torch.Tensor): Метки классов (batch_size).
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Нормализуем эмбеддинги
        features = F.normalize(features, p=2, dim=1)

        loss = 0.0
        batch_size = features.size(0)

        for i in range(batch_size):
            label = labels[i].item()
            # Извлекаем все эмбеддинги текущего класса
            class_mask = labels == label
            class_features = features[class_mask]
            if class_features.size(0) <= 1:
                continue  # Пропускаем, если нет других образцов в классе

            # Вычисляем центроид, исключая текущий образец
            positive_features = class_features[torch.arange(class_features.size(0)) != (i if class_mask[i] else -1)]
            centroid = positive_features.mean(dim=0)

            # Вычисляем расстояние до центроида
            distance = F.mse_loss(features[i], centroid, reduction='mean')
            loss += distance

        loss = loss / batch_size
        return self.weight * loss


# Centroid-loss with exponent and dynamic temperature
class CentroidLossExponential(nn.Module):
    def __init__(self, weight=1.0, temperature=0.3):
        """
        Centroid Loss с использованием формулы ln(1 + exp(d(A, c_p) - d(A, c_n))), где d - евклидово расстояние.
        Args:
            weight (float): Вес для Centroid Loss.
        """
        super(CentroidLossExponential, self).__init__()
        self.weight = weight
        self.temperature = temperature


    def forward(self, features, labels, epoch):
        """
        Вычисление Centroid Loss с исключением текущего образца при вычислении позитивного центроида.
        Args:
            features (torch.Tensor): Эмбеддинги (batch_size x embedding_dim).
            labels (torch.Tensor): Метки классов (batch_size).
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Нормализуем эмбеддинги
        features = F.normalize(features, p=2, dim=1)

        batch_size = features.size(0)
        unique_labels = labels.unique()
        device = features.device
        loss = 0.0
        count = 0

        # Предварительно вычисляем центроиды для каждого класса в батче
        centroids = {}
        for label in unique_labels:
            class_mask = labels == label
            class_features = features[class_mask]
            centroids[label.item()] = class_features.mean(dim=0)

        for i in range(batch_size):
            label = labels[i].item()
            feature = features[i]

            # Позитивный центроид, исключая текущий образец
            class_mask = labels == label
            class_indices = class_mask.nonzero(as_tuple=False).squeeze()
            if class_indices.numel() <= 1:
                continue  # Пропускаем, если нет других образцов в классе
            # Исключаем текущий образец
            positive_indices = class_indices[class_indices != i]
            positive_features = features[positive_indices]
            c_p = positive_features.mean(dim=0)

            # Расстояние между якорем и позитивным центроидом
            d_p = F.pairwise_distance(feature.unsqueeze(0), c_p.unsqueeze(0), p=2).squeeze()

            # Негативные центроиды (другие классы)
            negative_labels = unique_labels[unique_labels != label]
            if negative_labels.numel() == 0:
                continue  # Пропускаем, если нет отрицательных классов
            negative_centroids = [centroids[l.item()] for l in negative_labels]
            negative_centroids = torch.stack(negative_centroids)

            # Расстояния между якорем и негативными центроидами
            d_ns = F.pairwise_distance(feature.unsqueeze(0).expand_as(negative_centroids), negative_centroids, p=2)

            # Выбираем ближайший негативный центроид
            d_n = d_ns.min()

            # Вычисляем потерю для текущего образца
            loss_sample = torch.log(1 + torch.exp((d_p - d_n)/min((self.temperature + 0.02 * epoch), 1)))
            loss += loss_sample
            count += 1

        if count > 0:
            loss = loss / count
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=device)

        return self.weight * loss



# Combined advanced triplet and centroid losses
class CombinedLoss(nn.Module):
    def __init__(self, triplet_loss_weight=1.0, centroid_loss_weight=5e-4, 
                 triplet_temperature=0.3, centroid_temperature=0.3):
        """
        Объединённая функция потерь.
        Args:
            triplet_loss_weight (float): Вес для Triplet Loss.
            centroid_loss_weight (float): Вес для Centroid Loss.
        """
        super(CombinedLoss, self).__init__()
        self.triplet_loss = TripletLossWithExpUpd(
            temperature=triplet_temperature
        )

        self.centroid_loss = CentroidLossExponential(
            weight=centroid_loss_weight,
            temperature=centroid_temperature
        )
        self.triplet_loss_weight = triplet_loss_weight

    def forward(self, features, labels, epoch):
        """
        Вычисление объединённой функции потерь.
        Args:
            features (torch.Tensor): Эмбеддинги (batch_size x embedding_dim).
            labels (torch.Tensor): Метки классов (batch_size).
        Returns:
            torch.Tensor: Scalar loss.
        """
        triplet_loss = self.triplet_loss(features, labels, epoch)
        centroid_loss = self.centroid_loss(features, labels, epoch)
        total_loss = self.triplet_loss_weight * triplet_loss + centroid_loss
        return total_loss



# Another (not yet used) technique for metric learning
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss as described in:
    https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Compute loss for model. If both `features` and `labels` are None, we return 0.

        Args:
            features: hidden vector of shape [batch_size, feature_dim].
            labels: ground truth of shape [batch_size].
        Returns:
            A loss scalar.
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size < 2:
            # Недостаточно примеров для вычисления потерь
            return torch.tensor(0.0, requires_grad=True, device=device)

        labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [batch_size, batch_size]

        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_count = 1

        # Вычисляем сходство между эмбеддингами
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )  # [batch_size, batch_size]

        # Для численной стабильности
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Маска для положительных примеров
        mask = mask.repeat(anchor_count, 1)

        # Убираем совпадения с самим собой
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # Вычисляем логиты
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Вычисляем потерю
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Потеря
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss