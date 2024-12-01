import argparse
import copy
import os
import shutil
from time import time
from typing import List, Dict, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.nn import functional as F
from ultralytics import YOLO
from torch import nn
from scipy.optimize import linear_sum_assignment

from reid_model.train_utils import create_feature_extractor


SHOW_PREDICTIONS = False


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', '-v', type=str, required=True, help='Path to directory with test videos.')
    parser.add_argument('--mount', '-m', type=str, required=True, help='Path to directory with trained models and other required data.')
    parser.add_argument('--save_dir', '-s', type=str, required=True, help='Path to directory where predictions will be saved.')
    return parser.parse_args()


def get_avg_fps(processing_times: List[float]) -> float:
    return 1 / np.mean(processing_times)


def create_reid_model(checkpoint: str):
    '''
    Возвращает модель для REID.
    '''
    reid_model = create_feature_extractor()

    reid_model.cuda()
    reid_model = nn.DataParallel(reid_model)

    reid_model.load_state_dict(torch.load(checkpoint), strict=False)
    reid_model.eval()

    return reid_model


def clean_dist_matrix(matrix: np.ndarray, value: float = 100.0):
    non_100_rows = (matrix != value).any(dim=1)
    matrix = matrix[non_100_rows]

    non_100_cols = (matrix != value).any(dim=0)
    matrix = matrix[:, non_100_cols]

    return matrix


class MCMOT:
    def __init__(
        self,
        person_detector: YOLO,
        feature_extractor: nn.Module,
        num_cams: int,
        similarity_threshold: float,
        gallery_similarity_threshold: float,
        alpha: float,
        confirmation_threshold: int,
        log_dir: str,
        tracker_path: str,
    ):
        """
        Инициализация класса MCMOT.

        :param person_detector: Модель детекции объектов (YOLO).
        :param feature_extractor: Модель извлечения признаков для ReID.
        :param num_cams: Количество камер.
        :param similarity_threshold: Порог сходства для подтверждения идентификатора.
        :param gallery_similarity_threshold: Порог сходства для обновления галереи.
        :param alpha: Коэффициент обновления центроидов.
        :param confirmation_threshold: Количество эмбеддингов для подтверждения идентификатора.
        :param log_dir: Директория для сохранения логов.
        :param tracker_path: Путь к конфигурационному файлу трекера.
        """
        self._detectors = [copy.deepcopy(person_detector) for _ in range(num_cams)]

        self._feature_extractor = feature_extractor
        self._feature_extractor.eval()
        self._fe_transforms = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._similarity_threshold = similarity_threshold
        self._gallery_similarity_threshold = gallery_similarity_threshold
        self._alpha = alpha
        self.confirmation_threshold = confirmation_threshold
        self.buffer_size = 12

        self._log_dir = log_dir
        self._tracker_path = tracker_path
        self._num_cams = num_cams

        # Идентификаторы
        self._global_id = 1000  # Начальный глобальный ID
        self.next_temp_id = 0    # Счётчик для временных ID

        # Словари для хранения данных
        self.id_centroids = {}         # global_id -> centroid embedding
        self.id_embedding_buffers = {} # global_id -> list of embeddings

        # Галерея
        self.gallery_centroids = {}          # gallery_id -> centroid_embedding
        self.global_id_to_gallery_id = {}    # global_id -> gallery_id
        self.next_gallery_id = 1             # Счётчик для присвоения новых gallery_id

        # Создание директории для логов
        os.makedirs(self._log_dir, exist_ok=True)

    def _update_gallery(self, global_id, embedding):
        if global_id not in self.gallery_centroids:
            self.gallery_centroids[global_id] = embedding
        else:
            # Обновление центроида галереи как скользящего среднего
            self.gallery_centroids[global_id] = self._alpha * embedding + (1 - self._alpha) * self.gallery_centroids[global_id]
            self.gallery_centroids[global_id] = F.normalize(self.gallery_centroids[global_id], p=2, dim=0)

    def _match_with_centroids(self, embeddings: List[torch.Tensor], local_ids: List[str]) -> Dict[str, Tuple[int, float]]:
        """
        Сопоставляет текущие эмбеддинги с сохранёнными центроидами и галереей.

        :param embeddings: Список эмбеддингов.
        :param local_ids: Список локальных ID, соответствующих эмбеддингам.
        :return: Словарь, отображающий локальные ID в кортеж (global_id, similarity).
        """
        matches = {}
        for emb, local_id in zip(embeddings, local_ids):
            emb_normalized = F.normalize(emb, p=2, dim=0)

            # Вычисляем сходства с существующими центроидами
            similarities = {}
            for global_id, centroid in self.id_centroids.items():
                similarity = torch.dot(emb_normalized, centroid).item()
                similarities[global_id] = similarity

            # Вычисляем сходства с галереей
            gallery_similarities = {}
            for gallery_id, gallery_centroid in self.gallery_centroids.items():
                similarity = torch.dot(emb_normalized, gallery_centroid).item()
                gallery_similarities[gallery_id] = similarity

            # Находим лучшее совпадение в текущих треках
            if similarities:
                best_global_id = max(similarities, key=similarities.get)
                max_similarity = similarities[best_global_id]
            else:
                best_global_id = None
                max_similarity = -1

            # Если не нашли хорошее совпадение, ищем в галерее
            if max_similarity < self._similarity_threshold:
                if gallery_similarities:
                    best_gallery_id = max(gallery_similarities, key=gallery_similarities.get)
                    gallery_max_similarity = gallery_similarities[best_gallery_id]
                    if gallery_max_similarity > self._gallery_similarity_threshold:
                        best_global_id = best_gallery_id
                        max_similarity = gallery_max_similarity
                        # Обновляем центроид текущего трека на основе галереи
                        self.id_centroids[best_global_id] = self._alpha * emb_normalized + (1 - self._alpha) * self.id_centroids.get(best_global_id, emb_normalized)
                        self.id_centroids[best_global_id] = F.normalize(self.id_centroids[best_global_id], p=2, dim=0)
                    else:
                        best_global_id = None
                else:
                    best_global_id = None

            if best_global_id is not None and max_similarity > self._similarity_threshold:
                matches[local_id] = (best_global_id, max_similarity)
                # Обновляем центроид
                self._update_centroid(best_global_id, emb_normalized)
            else:
                # Если нет хорошего совпадения, проверяем галерею
                if gallery_similarities and max(gallery_similarities.values()) > self._gallery_similarity_threshold:
                    best_gallery_id = max(gallery_similarities, key=gallery_similarities.get)
                    matches[local_id] = (best_gallery_id, gallery_similarities[best_gallery_id])
                    # Обновляем центроид и галерею
                    self.id_centroids[best_gallery_id] = self._alpha * emb_normalized + (1 - self._alpha) * self.id_centroids.get(best_gallery_id, emb_normalized)
                    self.id_centroids[best_gallery_id] = F.normalize(self.id_centroids[best_gallery_id], p=2, dim=0)
                    self._update_gallery(best_gallery_id, emb_normalized)
                else:
                    # Назначаем новый global_id
                    new_global_id = self._global_id
                    self._global_id += 1
                    self.id_centroids[new_global_id] = emb_normalized
                    matches[local_id] = (new_global_id, 0.0)
                    # Добавляем в буфер эмбеддингов
                    self.id_embedding_buffers[new_global_id] = [emb_normalized]
        return matches

    def _update_centroid(self, global_id: int, new_embedding: torch.Tensor):
        """
        Обновляет центроид для global_id новым эмбеддингом и обновляет буфер эмбеддингов.

        :param global_id: Глобальный идентификатор.
        :param new_embedding: Новый эмбеддинг.
        """
        # Обновляем центроид (скользящее среднее)
        old_centroid = self.id_centroids.get(global_id, new_embedding)
        updated_centroid = self._alpha * new_embedding + (1 - self._alpha) * old_centroid
        updated_centroid = F.normalize(updated_centroid, p=2, dim=0)
        self.id_centroids[global_id] = updated_centroid

        # Обновляем буфер эмбеддингов
        if global_id not in self.id_embedding_buffers:
            self.id_embedding_buffers[global_id] = []
        self.id_embedding_buffers[global_id].append(new_embedding)
        if len(self.id_embedding_buffers[global_id]) > self.buffer_size:
            self.id_embedding_buffers[global_id].pop(0)

    def _assign_global_ids(self, matches: Dict[str, Tuple[int, float]], frames_bboxes: List[Dict[int, Dict]]) -> List[Dict[int, Dict]]:
        """
        Обновляет bounding boxes с глобальными ID и сходствами.

        :param matches: Словарь, отображающий локальные ID в кортеж (global_id, similarity).
        :param frames_bboxes: Список словарей bbox по кадрам.
        :return: Обновлённый список словарей bbox по кадрам.
        """
        for cam_id, frame_bboxes in enumerate(frames_bboxes):
            updated_bboxes = {}
            for local_id, data in frame_bboxes.items():
                if local_id in matches:
                    global_id, similarity = matches[local_id]
                    bbox_info = data.copy()
                    bbox_info['similarity'] = similarity
                    updated_bboxes[global_id] = bbox_info
                else:
                    # Назначаем новый global_id, если нет совпадения
                    new_global_id = self._global_id
                    self._global_id += 1
                    updated_bboxes[new_global_id] = {
                        'bbox': data['bbox'],
                        'similarity': None
                    }
                    # Инициализируем центроид, если необходимо
                    emb = self.id_centroids.get(new_global_id)
                    if emb is not None:
                        self.id_embedding_buffers[new_global_id] = [emb]
            frames_bboxes[cam_id] = updated_bboxes
        return frames_bboxes

    def _vizualize(self, frames: List[np.ndarray], frames_bboxes: List[Dict[int, Dict]]) -> List[Tuple[str, np.ndarray]]:
        """
        Визуализирует результаты трекинга на кадрах.

        :param frames: Список исходных кадров.
        :param frames_bboxes: Список словарей bbox по кадрам.
        :return: Список кортежей (название камеры, визуализированный кадр).
        """
        det_frames = []
        for cam_i, (frame, frame_bboxes) in enumerate(zip(frames, frames_bboxes)):
            det_frame = frame.copy()
            for object_id, object_info in frame_bboxes.items():
                bbox = object_info['bbox']
                similarity = object_info.get('similarity', None)
                x1, y1, x2, y2 = bbox
                text = f"ID: {object_id}"
                # Если есть соответствие с галереей
                gallery_id = self.global_id_to_gallery_id.get(object_id)
                if gallery_id is not None:
                    text += f" (Gallery ID: {gallery_id})"
                if similarity is not None:
                    text += f" Conf: {similarity:.2f}"
                # Рисуем прямоугольник и текст
                det_frame = cv2.rectangle(det_frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                det_frame = cv2.rectangle(det_frame, (x1, y1), (x2, y2), (0, 0, 250), 2)
                det_frame = cv2.putText(det_frame, text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 250), 2)
            det_frames.append((f'cam # {cam_i}', det_frame))
        return det_frames

    def _write_logs(self, timestamp: int, frames_bboxes: List[Dict[int, Dict]], log_filenames: List[str]):
        """
        Записывает логи для каждого кадра.

        :param timestamp: Текущий временной штамп.
        :param frames_bboxes: Список словарей bbox по кадрам.
        :param log_filenames: Список названий файлов логов для каждой камеры.
        """
        for frame_bboxes, log_filename in zip(frames_bboxes, log_filenames):
            log_path = os.path.join(self._log_dir, log_filename)
            with open(log_path, 'a') as log:
                for track_id, data in frame_bboxes.items():
                    bbox = data['bbox']
                    x1, y1, x2, y2 = bbox
                    log.write(f"{timestamp}, {track_id}, {x1}, {y1}, {x2 - x1}, {y2 - y1}, 1, 1, 1, 1\n")

    def _periodic_gallery_check(self):
        """
        Периодически проверяет и обновляет галерею с помощью накопленных эмбеддингов.
        """
        for global_id, buffer in list(self.id_embedding_buffers.items()):
            if len(buffer) == self.buffer_size:
                # Вычисляем центроид
                embeddings_stack = torch.stack(buffer)
                centroid = embeddings_stack.mean(dim=0)
                centroid = F.normalize(centroid, p=2, dim=0)

                # Сравниваем с галереей
                max_similarity = -1
                best_gallery_id = None
                for gallery_id, gallery_centroid in self.gallery_centroids.items():
                    similarity = torch.dot(centroid, gallery_centroid).item()
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_gallery_id = gallery_id

                if max_similarity > self._gallery_similarity_threshold:
                    # Обновляем галерейный центроид
                    self.gallery_centroids[best_gallery_id] = centroid
                    # Обновляем соответствие
                    self.global_id_to_gallery_id[global_id] = best_gallery_id
                else:
                    # Добавляем новый gallery_id
                    new_gallery_id = self.next_gallery_id
                    self.next_gallery_id += 1
                    self.gallery_centroids[new_gallery_id] = centroid
                    self.global_id_to_gallery_id[global_id] = new_gallery_id
                    print(f"Добавлен новый человек в галерею: Gallery ID {new_gallery_id} для Global ID {global_id}")

                # Очищаем буфер эмбеддингов
                self.id_embedding_buffers[global_id] = []

    def track(self, timestamp: int, frames: List[np.ndarray], log_filenames: List[str]) -> List[Tuple[str, np.ndarray]]:
        """
        Основной метод трекинга.

        :param timestamp: Текущий временной штамп.
        :param frames: Список кадров изображений.
        :param log_filenames: Список названий файлов логов для каждой камеры.
        :return: Список кортежей (название камеры, визуализированный кадр).
        """
        cam_ids = list(range(len(frames)))
        frames_embeddings, frames_bboxes = self._get_persons_embeddings(frames, cam_ids)
        det_frames = []

        # Собираем все эмбеддинги и локальные ID из всех камер
        all_embeddings = []
        all_local_ids = []
        for cam_id, embeddings_dict, frame_bboxes in zip(cam_ids, frames_embeddings, frames_bboxes):
            if embeddings_dict:
                for det_id, emb in embeddings_dict.items():
                    all_embeddings.append(emb)
                    all_local_ids.append(f"{cam_id}-{det_id}")

        if all_embeddings:
            # Сопоставляем эмбеддинги с сохранёнными центроидами и галереей
            matches = self._match_with_centroids(all_embeddings, all_local_ids)

            # Обновляем bounding boxes с глобальными ID и сходствами
            frames_bboxes = self._assign_global_ids(matches, frames_bboxes)

            # Визуализируем кадры
            det_frames = self._vizualize(frames, frames_bboxes)

            # Записываем логи
            self._write_logs(timestamp, frames_bboxes, log_filenames)

            # Периодическая проверка с галереей
            if timestamp % self.buffer_size == 0:
                self._periodic_gallery_check()
        else:
            # Если нет эмбеддингов, возвращаем исходные кадры с метками камер
            det_frames = [(f'cam # {cam_id}', frame) for cam_id, frame in zip(cam_ids, frames)]

        return det_frames

    def _assign_global_ids(self, matches: Dict[str, Tuple[int, float]], frames_bboxes: List[Dict[int, Dict]]) -> List[Dict[int, Dict]]:
        """
        Обновляет bounding boxes с глобальными ID и сходствами.

        :param matches: Словарь, отображающий локальные ID в кортеж (global_id, similarity).
        :param frames_bboxes: Список словарей bbox по кадрам.
        :return: Обновлённый список словарей bbox по кадрам.
        """
        for cam_id, frame_bboxes in enumerate(frames_bboxes):
            updated_bboxes = {}
            for local_id, data in frame_bboxes.items():
                id_code = f'{cam_id}-{local_id}'
                if id_code in matches:
                    global_id, similarity = matches[id_code]
                    bbox_info = data.copy()
                    bbox_info['similarity'] = similarity
                    updated_bboxes[global_id] = bbox_info
                else:
                    # Назначаем новый global_id, если нет совпадения
                    new_global_id = self._global_id
                    self._global_id += 1
                    updated_bboxes[new_global_id] = {
                        'bbox': data['bbox'],
                        'similarity': None
                    }
                    # Инициализируем центроид, если необходимо
                    emb = self.id_centroids.get(new_global_id)
                    if emb is not None:
                        self.id_embedding_buffers[new_global_id] = [emb]
            frames_bboxes[cam_id] = updated_bboxes
        return frames_bboxes

    def _get_persons_embeddings(
        self,
        frames: List[np.ndarray],
        cam_ids: List[int]
    ) -> Tuple[List[Dict[int, torch.Tensor]], List[Dict[int, Dict]]]:
        """
        Tracking people in frames and getting embeddings.

        Returns:
            frames_embeddings (List[Dict[int, torch.Tensor]]): List of embeddings for each frame.
            frames_bboxes (List[Dict[int, Dict]]): Bounding boxes for each frame.
        """
        frames_embeddings = []
        frames_bboxes = []
        for cam_id, frame in zip(cam_ids, frames):
            results = self._detectors[cam_id].track(frame, persist=True, classes=[0], verbose=True, tracker=self._tracker_path)
            boxes = results[0].boxes
            embeddings, ids = self._get_features(frame, boxes)
            frames_embeddings.append({lid: emb for lid, emb in zip(ids, embeddings) if emb is not None})
            frames_bboxes.append(self._get_bboxes(boxes))
        return frames_embeddings, frames_bboxes

    def _get_features(
        self,
        frame: np.ndarray,
        boxes,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Extract embeddings and IDs from detected boxes.

        Returns:
            embeddings (List[torch.Tensor]): List of embeddings for each person.
            persons_ids (List[str]): List of person IDs corresponding to the embeddings.
        """
        persons_imgs = []
        persons_ids = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().int().numpy().reshape(-1)
            # Ensure coordinates are within the image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes
            person_img = frame[y1:y2, x1:x2, :]
            person_img = cv2.resize(person_img, (128, 256))  # Note: (width, height)
            persons_imgs.append(person_img)
            try:
                track_id = str(int(box.id.item()))
            except AttributeError:
                track_id = f"temp_{self.next_temp_id}"
                self.next_temp_id += 1
            persons_ids.append(track_id)

        if not persons_imgs:
            return [], []  # Return empty lists if no persons detected

        # Convert images to tensors
        persons_imgs = np.stack(persons_imgs, axis=0)
        persons_imgs = torch.tensor(persons_imgs.transpose(0, 3, 1, 2)).float() / 255.0
        persons_imgs = self._fe_transforms(persons_imgs).to(next(self._feature_extractor.parameters()).device)

        with torch.no_grad():
            embeddings = self._feature_extractor(persons_imgs)

        # Ensure embeddings is a list of tensors
        embeddings_list = [embedding.cpu() for embedding in embeddings]

        return embeddings_list, persons_ids

    def _get_bboxes(self, boxes) -> Dict[int, Dict]:
        """
        Извлекает bounding boxes из детекций.

        :param boxes: Детекции YOLO.
        :return: Словарь с track_id как ключом и bbox как значением.
        """
        bboxes = {}
        for box in boxes:
            try:
                track_id = int(box.id.item())
            except AttributeError:
                track_id = self.next_temp_id
                self.next_temp_id += 1
            bbox_coords = box.xyxy.cpu().int().numpy().reshape(-1).tolist()
            bboxes[track_id] = {'bbox': bbox_coords}
        return bboxes



if __name__ == '__main__':
    args = parse()

    processing_times = []

    videos_list = Path(args.videos_dir).rglob('*.mp4')
    videos_list = sorted(videos_list)

    mcmot = MCMOT(
        person_detector=YOLO(os.path.join(args.mount, "yolov11s_trained.pt")),
        feature_extractor=create_reid_model(os.path.join(args.mount, 'best_reid_model.pth')),
        similarity_threshold=0.75,
        gallery_similarity_threshold=0.85,
        alpha=0.1,
        confirmation_threshold=12,
        num_cams=len(videos_list),
        log_dir=args.save_dir,
        tracker_path=os.path.join(args.mount, 'bytetrack.yaml')
    )

    caps = [cv2.VideoCapture(video_path) for video_path in videos_list]
    videos_basenames = [os.path.basename(video_path) for video_path in videos_list]
    video_logs = [os.path.splitext(video_name)[0] + '.txt' for video_name in videos_basenames]

    assert args.save_dir != './' and args.save_dir != '.'
    if os.path.exists(args.save_dir) and os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    # Инициализация VideoWriter для каждого видео
    video_writers = []
    for idx, cap in enumerate(caps):
        # Получаем свойства видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Определяем кодек и создаем объект VideoWriter
        output_filename = os.path.join(args.save_dir, f"tracked_{videos_basenames[idx]}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Вы можете изменить кодек, если нужно
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
        video_writers.append(video_writer)

    ret = True
    frames = []
    timestamp = 1

    # Чтение первого кадра из каждого видео
    for cap in caps:
        cap_ret, frame = cap.read()
        frames.append(frame)
        ret = ret and cap_ret  # Убеждаемся, что ret становится False, если cap_ret False

    while ret:
        start_time = time()
        det_frames = mcmot.track(timestamp, frames, video_logs)
        end_time = time()

        process_time = end_time - start_time
        processing_times.append(process_time)

        # Запись обработанных кадров в соответствующие видеофайлы
        for idx, (title, det_frame) in enumerate(det_frames):
            video_writers[idx].write(det_frame)

        if SHOW_PREDICTIONS:
            for title, det_frame in det_frames:
                cv2.imshow(title, det_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        #Changed
        frames = []
        ret = True
        for cap in caps:
            cap_ret, frame = cap.read()
            frames.append(frame)
            ret = ret and cap_ret  # Убеждаемся, что ret становится False, если cap_ret False
        timestamp += 1

    avg_fps = get_avg_fps(processing_times)
    print(f'Average FPS: {avg_fps}')

    for cap in caps:
        cap.release()
    for writer in video_writers:
        writer.release()
