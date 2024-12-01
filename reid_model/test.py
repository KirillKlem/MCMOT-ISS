import numpy as np
from tqdm import tqdm

import torch

from train_utils import create_feature_extractor
from train_reid_model import log_dir, device, test_loader
from metrics import cmc_at_k, precision_at_k_embeddings


feature_extractor = create_feature_extractor()  # Используйте ту же архитектуру, что и при сохранении
train_just_now = True

if train_just_now:
    feature_extractor.load_state_dict(torch.load(f"{log_dir}/best_model.pth"))
else:
    log_dir_new = '/...' # Укажите до нужного чекпоинта модели
    feature_extractor.load_state_dict(torch.load(f"{log_dir_new}/best_model.pth"))

feature_extractor.eval()

with torch.no_grad():
    test_features = []
    test_labels = []
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        features = feature_extractor(images)  # Получение эмбеддингов
        test_features.append(features.cpu().numpy())
        test_labels.append(labels.cpu().numpy())

    test_features = np.vstack(test_features)
    test_labels = np.hstack(test_labels)

precision_k_test = precision_at_k_embeddings(test_features, test_features, test_labels, k=5)
cmc1 = cmc_at_k(test_features, test_features, test_labels, k=1)

print(f'Precision@5 на тестовом наборе: {precision_k_test:.4f}')
print(f'CMC@1 на тестовом наборе: {cmc1:.4f}')