import os

# 프로젝트 루트 디렉토리 (파일 위치 기준 절대경로)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 경로
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "mu")
PREDICT_DATA_DIR = os.path.join(BASE_DIR, "predict")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed_data")
PREDICT_PREPROCESSED_DIR = os.path.join(BASE_DIR, "predict_preprocessed_data")

# 모델 저장 경로
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

# 전처리 파라미터
NUM_INSTRUMENTS = 143   # len(dir(music21.instrument)) + 1 (유효하지 않은 악기 포함)
MAX_TIME_STEPS = 2000   # 500마디 × 4분의 1박자 해상도
TIME_RESOLUTION = 4     # 4분음표 기준 타임스텝 배율

# 학습 하이퍼파라미터
EPOCHS = 20
BATCH_SIZE = 20
VALIDATION_SPLIT = 0.2
