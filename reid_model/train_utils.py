import os
import json
from tqdm import tqdm
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import Dataset
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models.vision_transformer import VisionTransformer, vit_b_16


# Feature Extractor Class
class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
        )



    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Function for define feature extractor
def create_feature_extractor():
    '''
    Create feature extractor (model to extract features)
    Output: feature_extractor
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.resnet152(weights="IMAGENET1K_V2")
    feature_extractor = FeatureExtractor(base_model=base_model)
    feature_extractor.to(device)
    feature_extractor.train()
    return feature_extractor


# same, but based on ViT
class ViTFeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(ViTFeatureExtractor, self).__init__()
        self.conv_proj = base_model.conv_proj  # Для разделения изображения на патчи
        self.encoder = base_model.encoder     # Основной Transformer-энкодер
        self.cls_token = nn.Parameter(torch.zeros(1, 1, base_model.hidden_dim))  # Добавляем [CLS] токен
        self.pos_embedding = nn.Parameter(base_model.encoder.pos_embedding)  # Используем предобученные позиции
        self.hidden_dim = base_model.hidden_dim

    def forward(self, x):
        # Преобразуем изображение в патчи
        x = self.conv_proj(x)  # (batch_size, hidden_dim, num_patches_sqrt, num_patches_sqrt)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, hidden_dim)

        # Добавляем [CLS] токен
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, hidden_dim)

        # Добавляем позиционные эмбеддинги
        x = x + self.pos_embedding[:, :x.size(1), :]

        # Подаем в энкодер
        x = self.encoder(x)

        # Возвращаем только представление [CLS]
        return x[:, 0]  # (batch_size, hidden_dim)

def create_feature_extractor_vit():
    '''
    Create ViT-based feature extractor.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = vit_b_16(weights="IMAGENET1K_V1")  # Загружаем предобученную модель
    feature_extractor = ViTFeatureExtractor(base_model=base_model)
    feature_extractor.to(device)
    feature_extractor.train()
    return feature_extractor


# Create dataset for reid_model training
def generate_dataset(images_root, annotations_root, output_csv):
    image_id_counter = 1
    dataset = []
    global_class_id_counter = 0
    unique_person_dict = {}

    # Список камер (папок) в директориях изображений
    cameras = os.listdir(images_root)
    cameras = [cam for cam in cameras if os.path.isdir(os.path.join(images_root, cam))]

    for cam in cameras:
        # Определение номера камеры (super_class_id)
        super_class_id = cam  # При необходимости можно изменить метод определения

        images_dir = os.path.join(images_root, cam)
        annotations_dir = os.path.join(annotations_root, cam.replace('-', '_'), 'annotations')

        # Загрузка аннотаций из JSON файлов
        instances_train_global_path = os.path.join(annotations_dir, 'instances_Train_global.json')

        if not os.path.exists(instances_train_global_path):
            print(f'Annotation file not found: {instances_train_global_path}')
            continue

        with open(instances_train_global_path, 'r') as f:
            annotations_data = json.load(f)

        # Создание словарей для быстрого доступа
        images_info = {img['id']: img for img in annotations_data['images']}
        image_annotations = {}
        for ann in annotations_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)

        # Обработка каждого изображения
        for image_id, image_info in tqdm(images_info.items(), desc=f'Processing camera {cam}'):
            image_filename = image_info['file_name']
            # Корректировка пути к изображению
            image_path = os.path.join(images_root, image_filename)
            if not os.path.exists(image_path):
                image_path = os.path.join(images_dir, os.path.basename(image_filename))
                if not os.path.exists(image_path):
                    print(f'Image file not found: {image_path}')
                    continue

            # Извлечение номера временного сегмента из имени файла
            base_filename = os.path.basename(image_filename)
            parts = base_filename.split('_')
            if len(parts) >= 2:
                time_segment = parts[1]
            else:
                print(f'Failed to extract time segment from filename: {image_filename}')
                continue

            # Определение идентификатора последовательности (sequence_id)
            sequence_id = f"{super_class_id}_{time_segment}"

            # Получение аннотаций для этого изображения
            annotations = image_annotations.get(image_id, [])

            for ann in annotations:
                track_id = ann['attributes']['track_id']
                bbox = ann['bbox']  # [x, y, width, height]
                bbox_x, bbox_y, bbox_width, bbox_height = bbox

                # Создание уникального ключа для человека
                unique_person_key = (sequence_id, track_id)

                # Проверяем, есть ли этот ключ в словаре
                if unique_person_key not in unique_person_dict:
                    unique_person_dict[unique_person_key] = global_class_id_counter
                    global_class_id_counter += 1

                class_id = unique_person_dict[unique_person_key]

                dataset.append({
                    'image_id': image_id_counter,
                    'class_id': class_id,
                    'super_class_id': super_class_id,
                    'path': image_path,
                    'bbox_x': bbox_x,
                    'bbox_y': bbox_y,
                    'bbox_width': bbox_width,
                    'bbox_height': bbox_height
                })
                image_id_counter += 1

    # Создание DataFrame и сохранение в CSV
    df = pd.DataFrame(dataset)
    df.to_csv(output_csv, index=False)
    return df


def crop_images(df, cropped_images_dir, dataset_csv=None):
    # Путь к датасету с координатами bbox
    dataset_csv = 'dataset.csv'

    # Директория для сохранения обрезанных изображений
    os.makedirs(cropped_images_dir, exist_ok=True)

    if dataset_csv:
        # Загрузка датасета
        df = pd.read_csv(dataset_csv)

    # Обработка каждой записи в датасете
    for index, row in tqdm(df.iterrows()):
        image_path = row['path']
        class_id = row['class_id']
        image_id = row['image_id']
        bbox_x = int(row['bbox_x'])
        bbox_y = int(row['bbox_y'])
        bbox_width = int(row['bbox_width'])
        bbox_height = int(row['bbox_height'])

        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue

        # Получение размеров изображения
        img_height, img_width = image.shape[:2]

        # Корректировка координат bbox, чтобы они находились в пределах изображения
        x1 = max(0, bbox_x)
        y1 = max(0, bbox_y)
        x2 = min(bbox_x + bbox_width, img_width)
        y2 = min(bbox_y + bbox_height, img_height)

        # Обрезка изображения
        cropped_image = image[y1:y2, x1:x2]

        # Проверка на корректность обрезки
        if cropped_image.size == 0:
            print(f"Invalid crop for image {image_path} at index {index}")
            continue

        # Создание имени файла для обрезанного изображения
        cropped_image_filename = f"{image_id}_{class_id}.jpg"
        cropped_image_path = os.path.join(cropped_images_dir, cropped_image_filename)

        # Сохранение обрезанного изображения
        cv2.imwrite(cropped_image_path, cropped_image)

        # Опционально: можно добавить путь к обрезанному изображению обратно в датасет
        df.at[index, 'cropped_path'] = cropped_image_path

    # Опционально: сохранить обновленный датасет с путями к обрезанным изображениям
    df.to_csv('dataset_with_cropped_paths.csv', index=False)
    return df


def create_transform():
    ''''create transformation of image (preprocess image)'''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform


def drop_one_freq_class_id(train_metadata):
    temp = train_metadata["class_id"].value_counts().reset_index()
    temp.rename(columns={'count': 'n_objects'}, inplace=True)
    labels_drop = list(temp.loc[temp['n_objects'] == 1, 'class_id'])

    return train_metadata[~train_metadata['class_id'].isin(labels_drop)]


def visualize_distribution_obj_counts(train_metadata):
    temp = train_metadata["class_id"].value_counts().reset_index()
    temp.rename(columns={'count': 'n_objects'}, inplace=True)
    temp = temp['n_objects'].value_counts().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(temp["n_objects"], temp["count"], color="skyblue")
    plt.xlabel("Number of Objects in Class")
    plt.ylabel("Number of Classes")
    plt.title("Distribution of Object Counts per Class")
    plt.xticks(temp["n_objects"])  # Ensure x-axis ticks match the counts
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0, 100)
    plt.xticks(rotation=90)
    plt.show()


class PersonDataset(Dataset):
    def __init__(self, metadata, transform=None):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing the dataset information.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.loc[idx, 'cropped_path']
        image = Image.open(img_path).convert("RGB")
        label = self.metadata.loc[idx, 'class_id']

        if self.transform:
            image = self.transform(image)

        return image, label
    

# Визуализация преобразованного тензора
def imshow_tensor(img_tensor: torch.Tensor, label: int):
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))  # Конвертируем из Tensor CxHxW в HxWxC
    mean = np.array([0.485, 0.456, 0.406])  # Среднее значение для нормализации
    std = np.array([0.229, 0.224, 0.225])   # Стандартное отклонение для нормализации
    img = std * img + mean  # Денормализуем
    img = np.clip(img, 0, 1)  # Ограничиваем значения пикселей
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()