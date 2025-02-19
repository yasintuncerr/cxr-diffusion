import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MultiClassifier(nn.Module):
    def __init__(self, input_size=4096, num_classes=14, embedding_dim=768):
        super(MultiClassifier, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, embedding_dim),  # CLIP boyutuna uygun çıktı (768)
            nn.BatchNorm1d(embedding_dim),
        )
        
        # CLIP tarzı L2 normalizasyon
        self.l2_norm = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
        
        # Sınıflandırma başlığı
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # L2 normalizasyon (CLIP tarzı)
        normalized_features = self.l2_norm(features)
        
        # Sınıflandırma
        logits = self.classifier(normalized_features)
        
        return logits, normalized_features

    def get_clip_embedding(self, x):
        """CLIP ile uyumlu embedding üret"""
        x = x.view(x.size(0), -1)
        features = self.feature_extractor(x)
        return self.l2_norm(features)


# CLIP embeddingi kullanım örneği:
"""
# Model oluşturma
model = MultiClassifier(input_size=32*128, num_classes=14, embedding_dim=768)

# Eğitim sonrası CLIP embeddingi alma
with torch.no_grad():
    features = torch.randn(1, 1, 32, 128)  # örnek girdi
    clip_embedding = model.get_clip_embedding(features)  # shape: [1, 768]
"""