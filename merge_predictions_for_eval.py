'''Скрипт для объединения предсказаний с разных камер в один файл (для последующей валидации)'''

import argparse
import os
from pathlib import Path

import cv2


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_dir', '-p', type=str, required=True, help='Path to directory with model predictions (txt files).')
    parser.add_argument('--videos_dir', '-v', type=str, required=True, help='Path to directory with test videos.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    predictions_files = Path(args.predictions_dir).rglob('*.txt')
    predictions_files = sorted(predictions_files)

    videos_list = Path(args.videos_dir).rglob('*.mp4')
    videos_list = sorted(videos_list)

    print(f'Generating prediction by single-camera predictions: {predictions_files}')
    frames_cnt = []
    for prediction_file in predictions_files:
        video_name = os.path.basename(prediction_file).split('.')[0]
        video_file = list(filter(lambda x: os.path.basename(x).split('.')[0] == video_name, videos_list))[0]
        video_frames_cnt = int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_COUNT))
        frames_cnt.append(video_frames_cnt)
    
    with open(os.path.join(args.predictions_dir, 'prediction.txt'), 'w') as total_prediction_f:
        for i, prediction_file, in enumerate(predictions_files):
            add_frames = sum(frames_cnt[j] for j in range(0, i))
            print(prediction_file)
            with open(prediction_file, 'r') as single_prediction_f:
                single_predictions = single_prediction_f.readlines()

                if add_frames == 0:
                    for pred in single_predictions:
                        total_prediction_f.write(pred)
                
                else:
                    for pred in single_predictions:
                        frame_id, object_id, bx, by, bw, bh, conf, x, y, z = pred.split(',')
                        frame_id = int(frame_id) + add_frames
                        cor_pred = f"{frame_id}, {object_id}, {bx}, {by}, {bw}, {bh}, {conf}, {x}, {y}, {z}"
                        total_prediction_f.write(cor_pred)
