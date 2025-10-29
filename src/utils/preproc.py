# src/utils/preproc.py
# 최소 동작용 전처리 유틸. 추후 확장 가능.
from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None  # opencv 미설치 환경에서도 import만은 되게

def _identity(img: np.ndarray) -> np.ndarray:
    return img

def _clahe_norm(img: np.ndarray) -> np.ndarray:
    """
    간단 CLAHE 명암 보정 (RGB 입력 가정). OpenCV가 없으면 무시(아이덴티티).
    """
    if cv2 is None:
        return img
    # RGB -> LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    # LAB -> RGB
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out

def get_preproc(name: str):
    """
    eval_test.py에서 import만 해도 되도록 최소 구현.
    현재는 이름으로 함수만 돌려줌(실제 사용은 스크립트 내부 로직에 따름).
    """
    if not name or name.lower() in ("none", "identity", "noop"):
        return _identity
    if name.lower() in ("clahe", "clahe_norm"):
        return _clahe_norm
    # 알 수 없는 이름이면 안전하게 아이덴티티 반환
    return _identity
