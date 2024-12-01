import os
import json
import argparse
import multiprocessing
import logging
import random
import time
from datetime import datetime

from tqdm import tqdm
import pandas as pd
import numpy as np

from rich.logging import RichHandler
from rich.console import Console
from rich.progress import track
from rich_logger import RichTablePrinter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train_utils import generate_dataset, crop_images, create_transform, \
                        drop_one_freq_class_id, visualize_distribution_obj_counts, \
                        PersonDataset, imshow_tensor, create_feature_extractor
from metrics import precision_at_k_embeddings, cmc_at_k
from losses import CombinedLoss



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', '-v', type=str, required=True, 
                        help='Path to directory with test videos.')
    parser.add_argument('--mount', '-m', type=str, required=True, 
                        help='Path to directory with trained models and other required data.')
    parser.add_argument('--save_dir', '-s', type=str, required=True, 
                        help='Path to directory where predictions will be saved.')
    return parser.parse_args()


args = parse()

output_csv = 'dataset.csv'
annotations_root = 'annotations'
images_root = '"C:\Users\Kirill\Downloads\train (1)"' # Вставить папку, где хранится трэйн!!!

train_metadata = generate_dataset(images_root, annotations_root, output_csv)

cropped_images_dir = 'cropped'
train_metadata = crop_images(train_metadata, cropped_images_dir)

transform = create_transform()
train_metadata = drop_one_freq_class_id(train_metadata)
visualize_distribution_obj_counts(train_metadata)

max_cores = multiprocessing.cpu_count()

# Test creation dataset and dataloader
# Create an instance of the dataset
dataset = PersonDataset(metadata=train_metadata, transform=transform)
# Create a DataLoader for batching
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=max_cores)

i = 1488
sample_tensor_image, sample_label = dataset[i]
print(f"Tensor Image size: {sample_tensor_image.size()}")  # Torch Tensor имеет метод .size()
print(f"Label: {sample_label}")
imshow_tensor(sample_tensor_image, sample_label)

# Beautiful logging 
def setup_logger_and_tensorboard(model_name, log_dir):
    """
    Setting up a logger and TensorBoard for logging and tracking metrics.
    """
    logger = logging.getLogger("rich")
    if logger.hasHandlers():
        logger.handlers.clear()  # Удаляем только обработчики этого логгера

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f'{log_dir}/logs.log'

    # Настраиваем обработчик для записи в файл
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Настраиваем обработчик для консоли
    console_handler = RichHandler(rich_tracebacks=True,
                                  console=Console(stderr=True),
                                  markup=True)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # Добавляем обработчики
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger.info(f"[bold green]Training started for model: {model_name} 🚀[/bold green]")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Logs saved to: {log_filename}")
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard initialized. Logs directory: {log_dir}")

    return logger, writer


def set_seed():
    '''set seed to avoid different random splits'''
    seed = random.randint(0, 2**16 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


# Main training loop
def exponential_decay_lr(epoch, lr_start, decay_rate, max_epoch):
    return lr_start * (decay_rate ** (epoch / max_epoch))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model_name, feature_extractor, dataloader, val_loader,
                criterion, optimizer, num_epochs, callbacks, log_dir,
                experiment_name, seed, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger, writer = setup_logger_and_tensorboard(model_name, log_dir)

    logger.info(f"[#afb3ab]Model Configuration: {config['experiment_name']}[/#afb3ab]")
    for key, value in config.items():
        logger.info(f"{key.capitalize()}: {value}")

    feature_extractor.to(config['device'])

    logger_fields = {
        "epoch": {},
        "train_loss": {"goal": "lower_is_better", "format": "{:.4f}"},
        "val_loss": {"goal": "lower_is_better", "format": "{:.4f}"},
        "precision@5": {"goal": "higher_is_better", "format": "{:.4f}"},
        "CMC@1": {"goal": "higher_is_better", "format": "{:.4f}"},
        "learning_rate": {"format": "{:.6f}"},
        "duration": {"format": "{:.1f}", "name": "dur(s)"},
    }
    printer = RichTablePrinter(key="epoch", fields=logger_fields)
    printer.hijack_tqdm()

    for epoch in range(config['n_epochs']):
        start_time = time.time()
        current_lr = exponential_decay_lr(epoch, lr_start=config['learning_rate'], decay_rate=0.05, max_epoch=config['n_epochs'])

        # Устанавливаем новый learning rate для оптимизатора
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        feature_extractor.train()
        running_loss = 0.0
        logger.info(f"[yellow]Starting Epoch {epoch + 1}/{num_epochs}[/yellow]")

        with tqdm(dataloader, desc=f"Epoch {epoch + 1} - {model_name}", unit="batch", colour='blue') as tepoch:
            for batch_idx, (images, labels) in enumerate(tepoch):
                batch_start_time = time.time()
                images, labels = images.to(device), labels.to(device)
                features = feature_extractor(images)
                loss = criterion(features, labels, epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_time = time.time() - batch_start_time
                tepoch.set_postfix(loss=loss.item(), sec_per_batch=batch_time)

                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("Batch Loss", loss.item(), global_step)

        avg_train_loss = running_loss / len(dataloader)
        logger.info(f"[#e7c697]Epoch {epoch + 1}: Training Loss = {avg_train_loss:.4f}[/#e7c697]")
        writer.add_scalar("Epoch Average Loss", avg_train_loss, epoch + 1)


        feature_extractor.eval()
        val_loss = 0.0
        with torch.no_grad():
            all_features = []
            all_labels = []
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                features = feature_extractor(images)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                loss = criterion(features, labels, epoch)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            writer.add_scalar("Validation loss", val_loss, epoch + 1)
            callbacks.validation_end(epoch, val_loss)

        # Post-process validation features
        all_features = np.vstack(all_features)
        all_labels = np.hstack(all_labels)

        # Calculate metrics
        precision_k = precision_at_k_embeddings(all_features, all_features, all_labels, k=5)
        cmc1 = cmc_at_k(all_features, all_features, all_labels, k=1)

        # Log metrics
        writer.add_scalar('Precision@5', precision_k, epoch)
        writer.add_scalar('CMC@1', cmc1, epoch)
        logger.info(
            f"[bold yellow]Epoch {epoch + 1}/{config['n_epochs']}: Validation Loss = {val_loss:.4f}, "
            f"Precision@5 = {precision_k:.4f}, CMC@1 = {cmc1:.4f}[/bold yellow]"
        )

        # Log learning rate
        writer.add_scalar("Learning Rate", current_lr, epoch + 1)
        logger.info(f"[#00ff00]Epoch {epoch + 1}: Learning Rate = {current_lr:.6f}[/#00ff00]")

        # Update logging table
        printer.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "precision@5": precision_k,
            "CMC@1": cmc1,
            "learning_rate": current_lr,
            "duration": time.time() - start_time
        })

        # Early stopping and saving best model
        if callbacks.early_stopping(cmc1, epoch):
            logger.info("[red]Early stopping triggered.[/red]")
            break

        callbacks.save_best_model(feature_extractor, epoch, cmc1)


    logger.info(f"[#6666ff]Training completed for model: {config['model_name']}[/#6666ff]")
    writer.close()
    printer.finalize()


# Define hyperparameters
seed = set_seed()


learning_rate = 8e-5
n_epochs = 20
batch_size = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = create_feature_extractor()

criterion = CombinedLoss(
    triplet_loss_weight=0.4,
    centroid_loss_weight=0.6,
    triplet_temperature=0.5,
    centroid_temperature=0.2,
)

optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=learning_rate)


config = {
    'seed': seed,
    'experiment_name': 'resnet_152',
    'model_name': type(feature_extractor).__name__,
    'loss': type(criterion).__name__,
    'optimizer': type(optimizer).__name__,
    'n_epochs': n_epochs,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'device': device.type,
}

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'logs/{config["experiment_name"]}_{current_time}'
os.makedirs(log_dir, exist_ok=True)


# Callbacks
class TrainingCallbacks:
    def __init__(self, on_epoch_end=None, on_batch_end=None, on_validation_end=None,
                 early_stopping=None, model_checkpoint=None):
        self.on_epoch_end = on_epoch_end
        self.on_validation_end = on_validation_end
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint

    def epoch_end(self, epoch, n_epochs, average_loss):
        if self.on_epoch_end:
            self.on_epoch_end(epoch, n_epochs, average_loss)

    def validation_end(self, epoch, val_loss):
        if self.on_validation_end:
            self.on_validation_end(epoch, val_loss)

    def early_stop(self, val_loss, epoch):
        if self.early_stopping:
            return self.early_stopping(val_loss, epoch)

    def save_best_model(self, model, epoch, val_loss):
        if self.model_checkpoint:
            return self.model_checkpoint(model, epoch, val_loss)
        

class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        """
        Инициализация механизма ранней остановки.
        Args:
            patience (int): Количество эпох ожидания до остановки при отсутствии улучшений.
            mode (str): Режим отслеживания — "max" для максимизации или "min" для минимизации метрики.
        """
        self.patience = patience
        self.mode = mode
        self.best_value = -float('inf') if mode == "max" else float('inf')
        self.counter = 0

    def __call__(self, monitor_value, epoch):
        """
        Проверяет, следует ли остановить обучение на основе заданной метрики.
        Args:
            monitor_value (float): Текущее значение отслеживаемой метрики.
            epoch (int): Текущая эпоха.
        Returns:
            bool: True, если нужно остановить обучение, иначе False.
        """
        if epoch == 0:
            self.counter = 0

        # Проверяем, улучшилась ли метрика
        is_improved = (self.mode == "max" and monitor_value > self.best_value) or \
                      (self.mode == "min" and monitor_value < self.best_value)

        if is_improved:
            self.best_value = monitor_value
            self.counter = 0
        else:
            self.counter += 1

        print(
            f"EarlyStopping: Epoch {epoch + 1}, Metric: {monitor_value:.4f}, Best Value: {self.best_value:.4f}, "
            f"Patience Counter: {self.counter}/{self.patience}"
        )

        if self.counter >= self.patience:
            print("Early stopping triggered.")
            return True
        return False
    
class ModelCheckpoint:
    def __init__(self, log_dir, save_path="best_model.pth", monitor="CMC@1", mode="max"):
        self.save_path = f'{log_dir}/{save_path}'
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else -float('inf')

    def __call__(self, model, epoch, monitor_value):
        is_better = (self.mode == "min" and monitor_value < self.best_value) or \
                    (self.mode == "max" and monitor_value > self.best_value)

        if is_better:
            self.best_value = monitor_value
            torch.save(model.state_dict(), self.save_path)
            print(f"Model saved; Epoch {epoch + 1}, {self.monitor}: {monitor_value:.4f}")



# Split dataset on train, valid and test (not only random sampling)
def dataset_split(metadata, test_size):
    all_classes = list(metadata['class_id'].unique())

    size = int(len(all_classes) * (1 - test_size))
    train_labels = random.sample(all_classes, size)

    train = metadata[metadata['class_id'].isin(train_labels)]
    test = metadata[~metadata['class_id'].isin(train_labels)]

    return train, test


# Main function
if __name__ == '__main__':
    early_stopping = EarlyStopping(patience=4)
    model_checkpoint = ModelCheckpoint(save_path='best_model.pth', log_dir=log_dir)

    callbacks = TrainingCallbacks(
        early_stopping=early_stopping,
        model_checkpoint=model_checkpoint
    )


    train, val = dataset_split(train_metadata, 0.2)
    train, test = dataset_split(train, 0.2)

    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    val = val.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define datasets
    train_dataset = PersonDataset(metadata=train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=max_cores)

    val_dataset = PersonDataset(metadata=val, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=max_cores)

    test_dataset = PersonDataset(metadata=test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=max_cores)


    train_model(
        model_name=config["model_name"],
        feature_extractor=feature_extractor,
        dataloader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config["n_epochs"],
        seed=seed,
        callbacks=callbacks,
        log_dir=log_dir,
        experiment_name=config['experiment_name'],
        config=config
    )