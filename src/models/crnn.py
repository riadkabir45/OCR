import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """
    CNN-RNN-CTC model for text recognition
    """
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, dropout=0.1):
        super(CRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x128
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x64
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x64
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x64
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # 3x63
        )
        
        # Map to RNN input size
        self.map_to_seq = nn.Linear(512 * 3, hidden_size)
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification layer
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # CNN feature extraction
        conv_features = self.cnn(x)  # [B, 512, 3, W]
        
        # Reshape for RNN: [B, W, 512*3]
        batch_size, channels, height, width = conv_features.size()
        conv_features = conv_features.permute(0, 3, 1, 2)  # [B, W, 512, 3]
        conv_features = conv_features.reshape(batch_size, width, channels * height)
        
        # Map to RNN input size
        rnn_input = self.map_to_seq(conv_features)  # [B, W, hidden_size]
        rnn_input = self.dropout(rnn_input)
        
        # RNN
        rnn_output, _ = self.rnn(rnn_input)  # [B, W, hidden_size*2]
        
        # Classification
        output = self.classifier(rnn_output)  # [B, W, vocab_size]
        
        # For CTC loss, we need [W, B, vocab_size]
        output = output.permute(1, 0, 2)
        
        return F.log_softmax(output, dim=2)


def create_model(vocab_size, hidden_size=256, num_layers=2, dropout=0.1):
    """Create CRNN model"""
    model = CRNN(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    return model