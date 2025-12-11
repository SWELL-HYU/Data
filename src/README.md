# Data/src 디렉토리 파일 설명

이 디렉토리는 추천 시스템의 핵심 모듈들을 포함합니다. 각 파일의 역할과 알고리즘을 설명합니다.

---

## 📁 파일 목록 및 역할

### 1. `__init__.py`
**역할**: Python 패키지 초기화 파일
- 빈 파일로, `Data/src`를 Python 패키지로 인식하게 함

---

### 2. `neumf_model.py` - Neural Matrix Factorization 모델
**역할**: 추천 시스템의 핵심 신경망 모델 정의

**주요 클래스**: `NeMF`

**알고리즘**:
1. **임베딩 레이어**
   - `user_embedding`: 사용자 임베딩 테이블 `(num_users, embedding_dim)`
   - `item_embedding`: 아이템 임베딩 테이블 `(num_items, embedding_dim)`

2. **Forward Pass (두 가지 경로 결합)**
   ```
   [A] GMF (Generalized Matrix Factorization) 경로:
       - user_emb * item_emb (요소별 곱)
       - 유사도 학습에 유리
       - 출력: (batch_size, embedding_dim)
   
   [B] MLP (Multi-Layer Perceptron) 경로:
       - [user_emb, item_emb] concatenation → (batch_size, embedding_dim * 2)
       - MLP 레이어 통과 (기본: 512 → 64)
       - ReLU 활성화 + Dropout (0.2)
       - 복잡한 비선형 관계 학습
       - 출력: (batch_size, hidden_dims[-1])
   
   [C] 결합 (NeuMF 방식):
       - [GMF_output, MLP_output] concatenation
       - 최종 Linear 레이어 (hidden_dims[-1] + embedding_dim → 1)
       - Sigmoid 활성화 → 0~1 사이의 선호 확률
   ```

3. **가중치 초기화**
   - 임베딩: 정규분포 초기화 (std=0.01)
   - Linear 레이어: Xavier 초기화

**사용 위치**: 
- `night_model_training.py`: 밤 모델 학습
- `day_model_update.py`: 낮 모델 업데이트
- `hybrid_recommendation.py`: 하이브리드 추천

---

### 3. `bpr_loss.py` - BPR Loss 구현
**역할**: Bayesian Personalized Ranking Loss 계산

**주요 클래스**: `BPRLoss`

**알고리즘**:
```
BPR Loss = -log(σ(x_pos - x_neg))
         = log(1 + exp(-(x_pos - x_neg)))
         = softplus(-(x_pos - x_neg))

여기서:
- x_pos: Positive 아이템에 대한 예측 점수
- x_neg: Negative 아이템에 대한 예측 점수
- σ: Sigmoid 함수
```

**핵심 아이디어**:
- Pairwise ranking: Positive 아이템 점수가 Negative 아이템 점수보다 높아야 함
- Implicit feedback에 적합 (명시적 rating 불필요)
- 사용자별 개인화된 순위 학습

**수식**:
```
L_BPR = Σ_{(u,i,j) ∈ D} -ln(σ(x̂_ui - x̂_uj))

D: (user, positive_item, negative_item) 튜플 집합
```

**사용 위치**:
- `night_model_training.py`: 밤 모델 학습
- `day_model_update.py`: 낮 모델 미세 학습

---

### 4. `bpr_dataset.py` - BPR 데이터셋
**역할**: BPR Loss 학습을 위한 (user, positive, negative) 튜플 생성

**주요 클래스**: `BPRDataset`

**알고리즘**:
1. **초기화**
   - Positive 상호작용 저장: `[(user_idx, item_idx, interaction_type), ...]`
   - Skip 상호작용 저장 (Negative 후보군)
   - User별 Positive 아이템 Set 생성 (빠른 조회용)
   - User별 Skip 아이템 Set 생성

2. **Negative 샘플링 전략** (`_sample_negative`):
   ```
   if Skip 데이터가 있고 (70% 확률):
       → Skip 아이템 중 랜덤 선택
   else:
       → 전체 아이템 중 랜덤 샘플링
       → 단, 사용자가 본 적 없는 아이템만 선택
   ```

3. **데이터 반환** (`__getitem__`):
   ```python
   {
       'user_id': user_idx,
       'positive_item_id': pos_item_idx,
       'negative_item_id': neg_item_idx  # 동적 샘플링
   }
   ```

**특징**:
- Skip 데이터를 Negative로 우선 활용 (70% 확률)
- 사용자가 본 적 없는 아이템만 Negative로 선택
- 매 에폭마다 다른 Negative 샘플 생성 (동적 샘플링)

**사용 위치**:
- `night_model_training.py`: 밤 모델 학습 데이터셋
- `day_model_update.py`: 낮 모델 미세 학습 데이터셋

---

### 5. `evaluation.py` - 성능 평가 지표
**역할**: 추천 시스템 성능 평가 지표 계산

**주요 함수들**:

#### 5.1 `hit_rate_at_k`
**알고리즘**:
```
HR@K = 1 if Top-K에 실제 상호작용 아이템이 1개 이상 포함
     = 0 otherwise
```
- 이진 지표: 맞췄는지 여부만 확인

#### 5.2 `precision_at_k`
**알고리즘**:
```
Precision@K = (Top-K 중 실제 상호작용한 아이템 수) / K
```
- 추천의 정확도 측정

#### 5.3 `recall_at_k`
**알고리즘**:
```
Recall@K = (Top-K 중 실제 상호작용한 아이템 수) / (전체 실제 상호작용 아이템 수)
```
- 실제 상호작용 중 얼마나 찾았는지 측정

#### 5.4 `ndcg_at_k` (Normalized Discounted Cumulative Gain)
**알고리즘**:
```
DCG@K = Σ(i=1 to K) rel_i / log2(i + 1)
IDCG@K = 이상적인 경우의 DCG (모든 관련 아이템이 상위에 있을 때)
NDCG@K = DCG@K / IDCG@K
```
- 순위를 고려한 평가 지표
- 상위에 관련 아이템이 있을수록 높은 점수
- 0~1 사이 값 (1이 최고)

#### 5.5 `evaluate_recommendations`
**알고리즘**:
- 모든 사용자에 대해 위 지표들을 계산
- 평균값 반환

**사용 위치**:
- `night_model_training.py`: 밤 모델 학습 중 평가 (5 에폭마다)

---

### 6. `user_embedding_utils.py` - 임베딩 유틸리티
**역할**: User/Item 임베딩 저장/로드 유틸리티 함수

**주요 함수들**:

#### 6.1 `load_user_embeddings`
- JSON 파일에서 User 임베딩 로드
- 반환: `Dict[str, list]` (user_id → embedding list)

#### 6.2 `save_user_embeddings`
- User 임베딩을 JSON 파일에 저장
- numpy array를 list로 변환하여 저장

#### 6.3 `get_user_embedding`
- 특정 User의 임베딩만 가져오기

#### 6.4 `load_item_embeddings` / `save_item_embeddings`
- Item 임베딩 저장/로드 (현재는 사용 안 함, CSV 사용)

**사용 위치**:
- `night_model_training.py`: 밤 모델 임베딩 저장/로드
- `day_model_update.py`: 낮 모델 임베딩 저장/로드
- `hybrid_recommendation.py`: 임베딩 로드

---

### 7. `night_model_training.py` - 밤 모델 학습
**역할**: 하루가 끝나고 모든 상호작용 데이터로 밤 모델 학습

**주요 함수**: `train_night_model`

**알고리즘 흐름**:

1. **데이터 로드 및 전처리**
   ```
   - 상호작용 CSV 로드
   - 고유한 user_id, item_id 추출
   - ID 매핑 생성 (user_id → index, item_id → index)
   ```

2. **Positive/Negative 데이터 분리**
   ```
   - Like: 9배 oversampling (더 많이 학습)
   - Preference: 1번만 추가
   - Skip: Negative 후보군으로 저장
   ```

3. **Train/Test Split**
   ```
   - 80% Train, 20% Test (test_ratio=0.2)
   ```

4. **Item Embedding 로드 및 고정**
   ```
   - CSV 파일(outfit_embeddings.csv)에서 로드
   - 모델의 item_embedding에 설정
   - requires_grad_(False)로 고정 (학습 안 함)
   ```

5. **User Embedding 초기화**
   ```
   - day_user_embedding.json이 있으면 초기값으로 사용
   - 없으면 랜덤 초기화
   ```

6. **학습 설정**
   ```
   - Optimizer: AdamW (User Embedding만 학습)
   - Loss: BPR Loss
   - Scheduler: ReduceLROnPlateau (NDCG 기반)
     → NDCG가 연속 2번 상승하지 않으면 LR 감소 (factor=0.7)
   ```

7. **학습 루프**
   ```
   for epoch in range(num_epochs):
       for batch in dataloader:
           # BPR Loss 계산
           positive_scores = model(user_ids, positive_item_ids)
           negative_scores = model(user_ids, negative_item_ids)
           loss = bpr_loss(positive_scores, negative_scores)
           
           # User Embedding만 업데이트
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
       
       # 5 에폭마다 평가
       if (epoch + 1) % 5 == 0:
           metrics = evaluate_model(...)
           scheduler.step(metrics['NDCG@K'])  # NDCG 기반 LR 조정
   ```

8. **모델 저장**
   ```
   - neumf_night_model.pth: 전체 모델 체크포인트
   - night_user_embedding.json: User Embedding만 저장
   ```

**특징**:
- Item Embedding은 고정 (CSV에서 로드)
- User Embedding만 학습
- Like 데이터 oversampling (9배)
- Skip 데이터를 Negative로 활용
- NDCG 기반 Learning Rate Scheduler

---

### 8. `day_model_update.py` - 낮 모델 업데이트
**역할**: 밤 모델 기반으로 낮 모델 임베딩 초기화 및 미세 학습

**주요 클래스**: `DayModelUpdater`

**알고리즘 흐름**:

1. **밤 모델 로드**
   ```
   - night_model.pth 체크포인트 로드
   - 모델 구조 및 매핑 정보 로드
   ```

2. **Item Embedding 로드 및 고정**
   ```
   - CSV 파일에서 Item Embedding 로드
   - 모델에 설정 및 고정 (학습 안 함)
   ```

3. **낮 모델 임베딩 초기화** (`initialize_day_embeddings`)
   ```
   if day_user_embedding.json 존재:
       → 기존 낮 모델 임베딩 로드
   else:
       → 밤 모델 임베딩을 복사하여 초기화
   ```

4. **새로운 상호작용 로드** (`load_interactions_from_csv`)
   ```
   - CSV에서 Positive 상호작용만 로드 (like, preference)
   - (user_idx, item_idx) 튜플 리스트 반환
   ```

5. **미세 학습** (`fine_tune_user_embeddings`)
   ```
   - BPRDataset 생성
   - BPR Loss 사용
   - User Embedding만 업데이트 (Item Embedding 고정)
   - 1 에폭만 학습 (빠른 업데이트)
   ```

6. **낮 모델 임베딩 저장** (`save_day_user_embeddings`)
   ```
   - day_user_embedding.json에 저장
   - 밤 모델 임베딩은 건들지 않음
   ```

**특징**:
- 밤 모델 임베딩을 기반으로 초기화
- 실시간 상호작용으로 빠른 업데이트 (1 에폭)
- Item Embedding은 고정
- 낮 모델과 밤 모델 임베딩 분리 관리

---

### 9. `hybrid_recommendation.py` - 하이브리드 추천
**역할**: 밤 모델(아이템 정보) + 낮 모델(유저 정보) 결합 추천

**주요 클래스**: `HybridRecommender`

**알고리즘 흐름**:

1. **모델 로드** (`_load_model`)
   ```
   - 밤 모델 체크포인트 로드
   - 모델 구조 및 매핑 정보 로드
   ```

2. **임베딩 로드** (`_load_embeddings`)
   ```
   - Item Embedding: CSV 파일에서 로드 (밤 모델, 고정)
   - Day User Embedding: day_user_embedding.json 로드 (있으면)
   - Night User Embedding: night_user_embedding.json 로드 (백업용)
   ```

3. **유저 임베딩 주입** (`_inject_user_embedding`)
   ```
   if 낮 모델 임베딩 존재:
       → 낮 모델 임베딩 사용 (최신 정보)
   else if 밤 모델 임베딩 존재:
       → 밤 모델 임베딩 사용 (기본값)
   else:
       → 에러
   ```

4. **추천 수행** (`recommend`)
   ```
   - 유저 임베딩 주입
   - 후보 아이템들에 대한 점수 계산
   - 모델 Forward Pass:
     * GMF: user_emb * item_emb
     * MLP: [user_emb, item_emb] → MLP
     * 결합: [GMF, MLP] → Linear → Sigmoid
   - 점수 내림차순 정렬
   - Top-K 반환
   ```

**핵심 아이디어**:
- **밤 모델**: 안정적인 아이템 정보 (고정)
- **낮 모델**: 실시간 업데이트된 유저 정보 (동적)
- **하이브리드**: 두 정보를 결합하여 최적의 추천 제공

**사용 위치**:
- 실제 추천 서비스에서 사용
- 실시간 추천 요청 처리

---

### 10. `dataset.py` - 레거시 데이터셋 (사용 안 함)
**역할**: Rating 기반 학습용 데이터셋 (현재 미사용)

**주요 클래스**: `RecommendationDataset`

**알고리즘**:
- `(user_idx, item_idx, rating)` 튜플을 PyTorch 텐서로 변환
- BCELoss 등 rating 기반 Loss와 함께 사용

**현재 상태**: 
- 사용되지 않음 (BPR Loss 사용으로 인해 `BPRDataset` 사용)
- 레거시 코드

---

## 🔄 전체 워크플로우

```
1. [밤] night_model_training.py
   → 모든 상호작용 데이터로 밤 모델 학습
   → night_user_embedding.json 저장

2. [낮] day_model_update.py
   → 밤 모델 기반으로 낮 모델 초기화
   → 새로운 상호작용으로 미세 학습
   → day_user_embedding.json 저장

3. [추천] hybrid_recommendation.py
   → 밤 모델(아이템) + 낮 모델(유저) 결합
   → 실시간 추천 제공
```

---

## 📊 데이터 흐름

```
상호작용 CSV
    ↓
BPRDataset (Positive/Negative 샘플링)
    ↓
NeMF 모델 (GMF + MLP)
    ↓
BPR Loss (Pairwise Ranking)
    ↓
User Embedding 업데이트
    ↓
JSON 파일 저장
```

---

## 🔑 핵심 개념

1. **BPR Loss**: Pairwise ranking 학습 (Positive > Negative)
2. **NeuMF**: GMF + MLP 결합 모델
3. **Day-Night 분리**: 밤 모델(안정적) + 낮 모델(동적)
4. **Item Embedding 고정**: CSV에서 로드, 학습 안 함
5. **User Embedding만 학습**: 개인화 추천에 집중

