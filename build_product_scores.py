# build_product_scores.py
# 전처리된 성분 데이터(ingredients_by_skin_type.csv)를 사용해서
# 제품별 피부타입 점수 + 카테고리/서브카테고리를 미리 계산해 product_scores.csv로 저장

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------------
# 1. 경로 / 설정
# --------------------------------------------------------
BASE_DIR = Path(r"C:/Users/Tqan/Desktop/skincare_api")

MODEL_DIR = BASE_DIR / "out" / "skin_ingredient_distilbert_multi_v2"

# 전처리된 성분 데이터
CSV_IN = BASE_DIR / "processed" / "ingredients_by_skin_type.csv"

# 최종 결과 파일
CSV_OUT = BASE_DIR / "out" / "product_scores.csv"

SKIN_LABELS = ["combination", "dry", "normal", "oily"]

print("[INFO] BASE_DIR:", BASE_DIR)
print("[INFO] MODEL_DIR:", MODEL_DIR)
print("[INFO] CSV_IN   :", CSV_IN)
print("[INFO] CSV_OUT  :", CSV_OUT)

# --------------------------------------------------------
# 2. 모델 로딩
# --------------------------------------------------------
print("[INFO] 모델 로딩:", MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()
print("[INFO] 모델 로딩 완료, device =", device)

# --------------------------------------------------------
# 3. 전처리된 성분 데이터 로딩
#    (ingredients_by_skin_type.csv 를 제품 기준으로 정리)
# --------------------------------------------------------
print("[INFO] 전처리된 성분 데이터 로딩:", CSV_IN)
df_raw = pd.read_csv(CSV_IN, encoding="utf-8-sig")

required_cols = {"product_id", "ingredients", "product_name", "brand_name"}
missing = required_cols - set(df_raw.columns)
if missing:
    raise KeyError(f"ingredients_by_skin_type.csv 에 다음 컬럼이 없습니다: {missing}")

# 문자열 정리
df_raw["ingredients"] = df_raw["ingredients"].fillna("").astype(str).str.strip()
df_raw = df_raw[df_raw["ingredients"] != ""].copy()

# 같은 product_id가 여러 행(피부타입별 집계 등)으로 존재하므로
# product_id 기준으로 한 줄씩만 남기기
df_products = (
    df_raw
    .sort_values("product_id")
    .groupby("product_id", as_index=False)
    .agg({
        "product_name": "first",
        "brand_name": "first",
        "ingredients": "first",
    })
)

print("[INFO] 전처리된 제품 수 (unique product_id):", len(df_products))

# --------------------------------------------------------
# 4. 카테고리/서브카테고리 생성 (product_name 기반 규칙)
# --------------------------------------------------------

def classify_main_category(name: str) -> str:
    """
    간단한 규칙 기반 메인 카테고리:
    - 이름에 device/tool/roller/led 등 들어가면 'device'
    - 나머지는 'skincare'
    """
    if not isinstance(name, str):
        return "skincare"

    n = name.lower()

    device_keywords = [
        "device", "tool", "roller", "gua sha", "nu face", "nuface",
        "foreo", "cleansing brush", "brush device", "microcurrent",
        "led mask", "led device", "toning device", "facial toning",
        "bear mini", "trinity", "facegym", "massager"
    ]

    for kw in device_keywords:
        if kw in n:
            return "device"

    return "skincare"


def classify_sub_category(name: str) -> str:
    """
    서브 카테고리:
    cleanser, toner, serum, cream, mask, other
    """
    if not isinstance(name, str):
        return "other"

    n = name.lower()

    # 순서는 중요 (겹치는 키워드 고려)
    if any(kw in n for kw in ["cleanser", "face wash", "cleansing foam", "cleansing oil", "cleansing balm"]):
        return "cleanser"

    if any(kw in n for kw in ["toner", "essence", "softener", "skin lotion"]):
        return "toner"

    if any(kw in n for kw in ["serum", "ampoule", "ampoule", "booster"]):
        return "serum"

    if any(kw in n for kw in ["cream", "moisturizer", "moisturiser", "gel cream", "lotion", "emulsion", "milk"]):
        return "cream"

    if "mask" in n or "sheet mask" in n or "sleeping pack" in n:
        return "mask"

    return "other"


df_products["category"] = df_products["product_name"].apply(classify_main_category)
df_products["sub_category"] = df_products["product_name"].apply(classify_sub_category)

print("[INFO] 카테고리 분포:")
print(df_products["category"].value_counts())
print("[INFO] 서브 카테고리 분포:")
print(df_products["sub_category"].value_counts())

# --------------------------------------------------------
# 5. 예측 함수 (성분 → 4개 피부타입 점수)
# --------------------------------------------------------
@torch.no_grad()
def predict_scores(text: str) -> np.ndarray:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits.squeeze(0).cpu().numpy()
    logits = logits.clip(0.0, 1.0)
    return logits

# --------------------------------------------------------
# 6. 모든 제품에 대해 점수 계산
# --------------------------------------------------------
results = []

print("[INFO] 제품별 피부타입 점수 계산 시작...")

for idx, row in df_products.iterrows():
    ing = row["ingredients"]
    scores = predict_scores(ing)

    record = {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "brand_name": row["brand_name"],
        "ingredients": ing,
        "category": row["category"],
        "sub_category": row["sub_category"],
    }
    for i, skin in enumerate(SKIN_LABELS):
        record[skin] = float(scores[i])

    results.append(record)

    if idx % 50 == 0:
        print(f"[INFO] {idx} / {len(df_products)} products processed...")

# --------------------------------------------------------
# 7. 저장
# --------------------------------------------------------
df_out = pd.DataFrame(results)
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

print("[OK] product_scores 저장 완료:", CSV_OUT)
print("[INFO] 최종 제품 수:", len(df_out))
print("[INFO] 컬럼:", df_out.columns.tolist())
