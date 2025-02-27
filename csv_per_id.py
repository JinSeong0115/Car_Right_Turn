import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os

# Define TrajectoryDataset
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
                    obj_id = group.iloc[i + j]['id']
                    bbox_x, bbox_y, bbox_w, bbox_h = group.iloc[i + j]['bbox_x'], group.iloc[i + j]['bbox_y'], group.iloc[i + j]['bbox_w'], group.iloc[i + j]['bbox_h']
                    
                    sequence.append((frame, obj_id, bbox_x, bbox_y, bbox_w, bbox_h, center_x, center_y))
                
                # Store the full sequence
                sequences.append(sequence)

        return sequences

    def save_sequences_as_csv(self, output_csv_path):
        # Convert sequences to a DataFrame
        flattened_sequences = []
        for sequence in self.sequences:
            for frame_data in sequence:
                flattened_sequences.append(frame_data)
        
        # Create DataFrame with required columns
        sequence_df = pd.DataFrame(flattened_sequences, columns=['frame', 'id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'center_x', 'center_y'])
        
        # Sort by ID and frame
        sequence_df = sequence_df.sort_values(by=['id', 'frame'])
        
        # Save as CSV
        sequence_df.to_csv(output_csv_path, index=False)
        print(f"Sequences saved to {output_csv_path}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return coordinate information and ID/frame data separately
        return torch.tensor([s[2:] for s in self.sequences[idx]], dtype=torch.float32), torch.tensor([self.sequences[idx][0][0:2]], dtype=torch.float32)

# Set root directory
root_dir = '/home/kim_js/car_project/yolov9/YOLOv9-DeepSORT-Object-Tracking/runs/detect/'

# Process directories from exp1 to exp22
for exp_dir in range(1, 23):
    exp_path = os.path.join(root_dir, f'exp{exp_dir}', 'output.csv')
    if os.path.exists(exp_path):
        print(f"Loading data from {exp_path}...")
        
        # Load data
        data = pd.read_csv(exp_path)
        
        # Create dataset
        dataset = TrajectoryDataset(data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"Data from {exp_path} loaded successfully.")
        
        # Define path to save processed sequences
        output_csv_path = os.path.join(root_dir, f'exp{exp_dir}', 'processed_sequences.csv')
        
        # Save sequences as CSV
        print(f"Saving processed sequences to {output_csv_path}...")
        dataset.save_sequences_as_csv(output_csv_path)
        print(f"Processed sequences saved successfully to {output_csv_path}")
    else:
        print(f"{exp_path} does not exist.")

print("All files processed successfully.")
