"""
유저 임베딩 저장/로드 유틸리티
밤 모델 임베딩과 낮 모델 임베딩을 별도로 관리
"""
import json
import os
import numpy as np
from typing import Dict, Optional


def load_user_embeddings(file_path: str) -> Dict[str, list]:
    """
    JSON 파일에서 유저 임베딩을 로드합니다.
    
    Args:
        file_path: JSON 파일 경로
    
    Returns:
        Dict[str, list]: user_id -> embedding list
    """
    if not os.path.exists(file_path):
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_user_embeddings(embeddings: Dict[str, np.ndarray], file_path: str):
    """
    유저 임베딩을 JSON 파일에 저장합니다.
    
    Args:
        embeddings: Dict[str, np.ndarray] - user_id -> embedding array
        file_path: 저장할 JSON 파일 경로
    """
    # 디렉토리 생성
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # numpy array를 list로 변환
    embeddings_dict = {
        user_id: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        for user_id, embedding in embeddings.items()
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_dict, f, ensure_ascii=False, indent=2)


def get_user_embedding(user_id: str, file_path: str) -> Optional[np.ndarray]:
    """
    특정 유저의 임베딩을 가져옵니다.
    
    Args:
        user_id: 사용자 ID
        file_path: JSON 파일 경로
    
    Returns:
        np.ndarray 또는 None
    """
    embeddings = load_user_embeddings(file_path)
    if user_id in embeddings:
        return np.array(embeddings[user_id])
    return None


def save_item_embeddings(embeddings: Dict[str, np.ndarray], file_path: str):
    """
    아이템 임베딩을 JSON 파일에 저장합니다.
    
    Args:
        embeddings: Dict[str, np.ndarray] - item_id -> embedding array
        file_path: 저장할 JSON 파일 경로
    """
    # 디렉토리 생성
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # numpy array를 list로 변환
    embeddings_dict = {
        item_id: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        for item_id, embedding in embeddings.items()
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_dict, f, ensure_ascii=False, indent=2)


def load_item_embeddings(file_path: str) -> Dict[str, list]:
    """
    JSON 파일에서 아이템 임베딩을 로드합니다.
    
    Args:
        file_path: JSON 파일 경로
    
    Returns:
        Dict[str, list]: item_id -> embedding list
    """
    if not os.path.exists(file_path):
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


