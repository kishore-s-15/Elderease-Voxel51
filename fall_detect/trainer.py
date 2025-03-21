import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
 
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
 
# Define paths
FALL_DIR = os.path.join(os.getcwd(), "Fall")
NO_FALL_DIR = os.path.join(os.getcwd(), "NoFall")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # For Apple Silicon Macs
else:
    DEVICE = torch.device("cpu")
    
print(f"Using device: {DEVICE}")
 
# Define hyperparameters
BATCH_SIZE = 4  # Reduced batch size to accommodate larger sequence length
NUM_EPOCHS = 5
LEARNING_RATE = 0.0002  # Slightly increased learning rate
SEQUENCE_LENGTH = 64  # Increased from 16 to 64 frames based on video analysis
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
VALIDATION_SPLIT = 0.1  # 80% train, 20% validation
FREEZE_BACKBONE_RATIO = 0.7  # Freeze 70% of the backbone layers
 
# Define transformations with augmentation
transform_train = transforms.Compose([
    transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.3),  # Random horizontal flips
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Slight color variations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
# Validation transform without augmentation
transform_val = transforms.Compose([
    transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
# Custom dataset for loading video sequences
class FallDetectionDataset(Dataset):
    def __init__(self, fall_dir, no_fall_dir, sequence_length, transform=None):
        self.fall_dir = fall_dir
        self.no_fall_dir = no_fall_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Get video paths
        self.fall_videos = [os.path.join(fall_dir, f) for f in os.listdir(fall_dir)
                            if f.endswith(('.mp4', '.avi')) and not f.startswith('.')]
        self.no_fall_videos = [os.path.join(no_fall_dir, f) for f in os.listdir(no_fall_dir)
                              if f.endswith(('.mp4', '.avi')) and not f.startswith('.')]
        
        # Create list of (video_path, label) pairs
        self.samples = [(video, 1) for video in self.fall_videos] + [(video, 0) for video in self.no_fall_videos]
        
        print(f"Loaded {len(self.fall_videos)} fall videos and {len(self.no_fall_videos)} no-fall videos")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Load video and extract frames
        frames = self._load_video(video_path)
        
        # Convert to tensor
        if self.transform:
            frames = torch.stack([self.transform(Image.fromarray(frame)) for frame in frames])
        
        return frames, label
    
    def _load_video(self, video_path):
        """Extract frames from video, ensuring we get sequence_length frames evenly sampled."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle videos with fewer frames than sequence_length
        if total_frames <= self.sequence_length:
            # Read all frames and duplicate as needed
            frames = []
            for _ in range(total_frames):
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    # Use a black frame if reading fails
                    frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
            
            # Duplicate frames to reach sequence_length
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])
        else:
            # Sample frames evenly
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    # Use previous frame if reading fails
                    frames.append(frames[-1] if frames else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
        
        cap.release()
        return frames
 
# EfficientNet + GRU model
class EfficientGRU(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.5, freeze_ratio=0.7):
        super(EfficientGRU, self).__init__()
        
        # Load pre-trained EfficientNet
        self.efficient_net = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Calculate how many layers to freeze
        total_layers = len(list(self.efficient_net.features.children()))
        freeze_layers = int(total_layers * freeze_ratio)
        
        # Freeze backbone layers according to the ratio
        for i, param in enumerate(self.efficient_net.features.parameters()):
            if i < freeze_layers:
                param.requires_grad = False
                
        # Replace classifier with identity to get features
        self.efficient_net.classifier = nn.Identity()
        
        # EfficientNet-B0 output features is 1280
        feature_size = 1280
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Final classifier
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary classification: Fall or No Fall
        )
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape input to process frames through CNN
        cnn_input = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features with EfficientNet
        features = self.efficient_net(cnn_input)
        
        # Reshape features for GRU
        features = features.view(batch_size, seq_len, -1)
        
        # Process with GRU
        gru_out, _ = self.gru(features)
        
        # Use the last output from the GRU
        gru_out = gru_out[:, -1, :]
        
        # Pass through classifier
        output = self.classifier(gru_out)
        
        return output
 
# # ConvLSTM based model
# class ConvLSTM(nn.Module):
#     def __init__(self, freeze_ratio=0.7):
#         super(ConvLSTM, self).__init__()
        
#         # Load ResNet as feature extractor
#         self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        
#         # Calculate how many layers to freeze
#         total_layers = sum(1 for _ in self.resnet.parameters())
#         layers_to_freeze = int(total_layers * freeze_ratio)
        
#         # Freeze layers
#         for i, param in enumerate(self.resnet.parameters()):
#             if i < layers_to_freeze:
#                 param.requires_grad = False
        
#         # Remove the final fully connected layer
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
#         # LSTM layer
#         self.lstm = nn.LSTM(
#             input_size=512,  # ResNet18 feature size
#             hidden_size=256,
#             num_layers=2,
#             batch_first=True,
#             dropout=0.5,
#             bidirectional=True
#         )
        
#         # Fully connected layer for classification
#         self.fc = nn.Sequential(
#             nn.Linear(512, 128),  # 256*2 = 512 due to bidirectional
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 2)
#         )
    
#     def forward(self, x):
#         batch_size, seq_len, c, h, w = x.shape
        
#         # Process each frame through ResNet
#         cnn_output = []
#         for t in range(seq_len):
#             # Extract features with ResNet
#             frame_features = self.resnet(x[:, t, :, :, :])
#             cnn_output.append(frame_features)
        
#         # Stack frame features
#         cnn_output = torch.stack(cnn_output, dim=1)
#         cnn_output = cnn_output.squeeze(3).squeeze(3)  # Remove spatial dimensions
        
#         # Process sequence with LSTM
#         lstm_out, _ = self.lstm(cnn_output)
        
#         # Use the last output from LSTM for classification
#         lstm_out = lstm_out[:, -1, :]
        
#         # Classification
#         output = self.fc(lstm_out)
        
#         return output
 
# Learning rate scheduler
def get_lr_scheduler(optimizer):
    """Create a learning rate scheduler."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
 
# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Create learning rate scheduler
    scheduler = get_lr_scheduler(optimizer)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        pbar = tqdm(train_loader, desc=f'Training epoch {epoch+1}')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Statistics
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
        
        # Calculate epoch metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fall_detection_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.4f}')
        
        print()
    
    return model, history
 
# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['NoFall', 'Fall'])
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    return cm, report
 
# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
 
# Main execution
def main():
    # Create model output directory
    os.makedirs('models', exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = FallDetectionDataset(FALL_DIR, NO_FALL_DIR, SEQUENCE_LENGTH, transform_train)
    
    # Split into train and validation sets
    train_size = int((1 - VALIDATION_SPLIT) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create a new validation dataset with validation transform
    val_dataset = FallDetectionDataset(FALL_DIR, NO_FALL_DIR, SEQUENCE_LENGTH, transform_val)
    val_indices = val_dataset.samples.copy()
    np.random.shuffle(val_indices)
    val_indices = val_indices[:val_size]
    val_dataset.samples = val_indices
    
    print(f"Total samples: {len(train_dataset)}")
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    # model_choice = 'efficientgru'  # Options: 'efficientgru' or 'convlstm'
    
    # if model_choice == 'efficientgru':
    print("Initializing EfficientNet+GRU model...")
    model = EfficientGRU(freeze_ratio=FREEZE_BACKBONE_RATIO)
    # else:
    #     print("Initializing ConvLSTM model...")
    #     model = ConvLSTM(freeze_ratio=FREEZE_BACKBONE_RATIO)
    
    model = model.to(DEVICE)
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Train model
    print("Starting training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
    
    # Plot training history
    plot_training_history(history)
    
    # # Evaluate model
    # print("Evaluating model...")
    cm, report = evaluate_model(model, val_loader, DEVICE)
    
    print("Training completed!")
 
if __name__ == "__main__":
    main()