import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os

# TrajectoryDataset 정의
class TrajectoryDataset(Dataset):
    def __init__(self, data, seq_length=5):
        self.data = data
        self.seq_length = seq_length
        self.sequences = self.create_sequences()

    def create_sequences(self):
        sequences = []
        grouped_data = self.data.groupby('id')
        
        for _, group in grouped_data:
            group = group.sort_values(by='frame')
            
            if len(group) < self.seq_length:
                continue
            
            for i in range(len(group) - self.seq_length + 1):
                sequence = []
                for j in range(self.seq_length):
                    center_x = group.iloc[i + j]['bbox_x'] + group.iloc[i + j]['bbox_w'] / 2
                    center_y = group.iloc[i + j]['bbox_y'] + group.iloc[i + j]['bbox_h'] / 2
                    frame = group.iloc[i + j]['frame']
                    id = group.iloc[i + j]['id']
                    bx, by, bw, bh = group.iloc[i + j]['bbox_x'], group.iloc[i + j]['bbox_y'], group.iloc[i + j]['bbox_w'], group.iloc[i + j]['bbox_h']
                    
                    sequence.append((frame, id, bx, by, bw, bh, center_x, center_y))
                
                # 시퀀스를 전체 저장하도록 수정
                sequences.append(sequence)

        return sequences


    def save_sequences_as_csv(self, output_csv_path):
        # 시퀀스 전체를 DataFrame으로 변환
        flattened_sequences = []
        for sequence in self.sequences:
            for frame_data in sequence:
                flattened_sequences.append(frame_data)
        
        # 필요한 컬럼을 지정하여 DataFrame 생성
        sequence_df = pd.DataFrame(flattened_sequences, columns=['frame', 'id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'center_x', 'center_y'])
        
        # ID와 frame 기준으로 정렬
        sequence_df = sequence_df.sort_values(by=['id', 'frame'])
        
        # CSV로 저장
        sequence_df.to_csv(output_csv_path, index=False)
        print(f"Sequences saved to {output_csv_path}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 시퀀스의 좌표 정보와 id/frame 정보를 나누어 반환
        return torch.tensor([s[2:] for s in self.sequences[idx]], dtype=torch.float32), torch.tensor([self.sequences[idx][0][0:2]], dtype=torch.float32)

# 루트 디렉토리 지정
root_dir = '/home/kim_js/car_project/yolov9/YOLOv9-DeepSORT-Object-Tracking/runs/detect/'

# exp1부터 exp22까지의 모든 디렉토리에 대해 처리
for exp_dir in range(1, 23):
    exp_path = os.path.join(root_dir, f'exp{exp_dir}', 'output.csv')
    if os.path.exists(exp_path):
        print(f"Loading data from {exp_path}...")
        
        # 데이터 로드
        data = pd.read_csv(exp_path)
        
        # 데이터셋 생성
        dataset = TrajectoryDataset(data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"Data from {exp_path} loaded successfully.")
        
        # 처리된 시퀀스를 저장할 경로 지정
        output_csv_path = os.path.join(root_dir, f'exp{exp_dir}', 'processed_sequences.csv')
        
        # 시퀀스를 CSV로 저장
        print(f"Saving processed sequences to {output_csv_path}...")
        dataset.save_sequences_as_csv(output_csv_path)
        print(f"Processed sequences saved successfully to {output_csv_path}")
    else:
        print(f"{exp_path} does not exist.")

print("All files processed successfully.")
