# make_ingredient_dataset.py
import pandas as pd
from pathlib import Path

# 1. 경로 설정 (VSCode / 로컬 PC)
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "processed" / "reviews_products_merged_clean.csv"
OUT_PATH = ROOT / "processed" / "ingredients_by_skin_type.csv"

print(f"[INFO] CSV 로딩: {CSV_PATH}")

# 2. 데이터 로드
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# 3. 기본 필터링
valid_skin = ["combination", "dry", "normal", "oily"]

df = df.copy()

# 문자열 처리 오류 fix: .strip() → .str.strip()
df["ingredients"] = df["ingredients"].fillna("").astype(str).str.strip()
df = df[df["ingredients"] != ""]

df["user_skin_type"] = df["user_skin_type"].astype(str).str.lower()
df = df[df["user_skin_type"].isin(valid_skin)]

# is_recommended 확인
if "is_recommended" not in df.columns:
    raise ValueError("is_recommended 컬럼이 필요합니다.")

# 4. product_id + user_skin_type 기준 집계
group_cols = ["product_id", "user_skin_type"]

agg = df.groupby(group_cols).agg(
    n_reviews=("is_recommended", "size"),
    rec_rate=("is_recommended", "mean"),
    rating_mean=("rating_review", "mean"),
    ingredients=("ingredients", "first"),
    product_name=("product_name", "first"),
    brand_name=("brand_name", "first"),
).reset_index()

# 5. 최소 리뷰 수 필터
min_reviews = 5
agg = agg[agg["n_reviews"] >= min_reviews].reset_index(drop=True)

print("집계된 행 수:", len(agg))
print(agg.head())

# 6. 저장
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
agg.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"[INFO] 저장 완료: {OUT_PATH}")
