"""
추천 시스템용 데이터셋 클래스
"""
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class RecommendationDataset(Dataset):
    """
    추천 시스템 학습용 데이터셋
    """
    
    def __init__(self, interactions: List[Tuple[int, int, float]]):
        """
        Args:
            interactions: List[Tuple[user_idx, item_idx, rating]]
                rating: 0.0 (skip) ~ 1.0 (like/preference)
        """
        self.interactions = interactions
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_idx, item_idx, rating = self.interactions[idx]
        return {
            'user_id': torch.tensor(user_idx, dtype=torch.long),
            'item_id': torch.tensor(item_idx, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float)
        }


