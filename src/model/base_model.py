import torch
import torch.nn as nn
import torch.nn.functional as F


class FourViewClassifier4MNIST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.view1_feat_extra = nn.Sequential(
            nn.Linear(14*14, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Sigmoid())
        self.view2_feat_extra = nn.Sequential(
            nn.Linear(14*14, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Sigmoid())
        self.view3_feat_extra = nn.Sequential(
            nn.Linear(14*14, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Sigmoid())
        self.view4_feat_extra = nn.Sequential(
            nn.Linear(14*14, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(32*4, 128),
            nn.ReLU(),
            nn.Linear(128, 10))
        
    def forward(self, x):
        feats = self.get_feats(x)
        return self.classify(feats)
    
    def get_feats(self, x):
        x1 = x[:, :, :14, :14]
        x1 = torch.reshape(x1, (-1, 14*14))
        x2 = x[:, :, 14:, :14]
        x2 = torch.reshape(x2, (-1, 14*14))
        x3 = x[:, :, 14:, 14:]
        x3 = torch.reshape(x3, (-1, 14*14))
        x4 = x[:, :, :14, 14:]
        x4 = torch.reshape(x4, (-1, 14*14))
        
        feat1 = self.view1_feat_extra(x1)
        feat2 = self.view2_feat_extra(x2)
        feat3 = self.view3_feat_extra(x3)
        feat4 = self.view4_feat_extra(x4)
        
        feats = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return feats
    
    def classify(self, feats):
        return F.log_softmax(self.classifier(feats), dim=1)

