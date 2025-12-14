# api_app.py
# 피부타입 + 피부고민 기반 스킨케어 추천 FastAPI 서버
#
# 핵심 원칙
# 1) product_scores.csv : 전체 제품 + 성분 + 피부타입 점수
# 2) product_meta.csv   : 서비스 가능한 제품만 (url, image)
# 3) 행(row) 기준      : product_meta (inner join)
# 4) 컬럼(column) 기준 : product_scores의 ingredients 포함 유지
# 5) ingredients 컬럼 표준화 + suffix 안전 처리

from pathlib import Path
from typing import Dict, List
from fastapi import Query


import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# 1. 설정
# -----------------------------

BASE_DIR = Path(__file__).parent

MODEL_DIR = BASE_DIR / "out" / "skin_ingredient_distilbert_multi_v2"
PRODUCT_SCORES_PATH = BASE_DIR / "out" / "product_scores.csv"
PRODUCT_META_PATH = BASE_DIR / "data" / "product_meta.csv"
REVIEWS_PATH = BASE_DIR / "data" / "reviews_clean.csv"


SKIN_LABELS = ["combination", "dry", "normal", "oily"]

CATEGORY_KEYS = [
    "all",
    "cleanser",
    "toner",
    "serum",
    "cream",
    "mask",
    "sunscreen",
    "device",
    "other",
]

# -----------------------------
# 2. 피부 고민 성분 규칙
# -----------------------------

CONCERN_RULES = {
    "acne": {
        "positive": ["salicylic acid", "bha", "niacinamide", "zinc", "tea tree", "centella"],
        "negative": ["isopropyl myristate", "coconut oil"],
    },
    "sensitive": {
        "positive": ["ceramide", "panthenol", "allantoin", "centella", "madecassoside"],
        "negative": ["fragrance", "parfum", "essential oil", "alcohol denat"],
    },
    "brightening": {
        "positive": ["niacinamide", "ascorbic acid", "vitamin c", "arbutin", "licorice"],
        "negative": [],
    },
    "wrinkle": {
        "positive": ["retinol", "retinal", "peptide", "adenosine", "bakuchiol"],
        "negative": [],
    },
    "pore_sebum": {
        "positive": ["niacinamide", "zinc", "bha", "charcoal", "clay"],
        "negative": [],
    },
}

CONCERN_KEYS = sorted(CONCERN_RULES.keys())

# -----------------------------
# 3. FastAPI 앱
# -----------------------------

app = FastAPI(
    title="Skincare Recommender API",
    description="피부타입 + 피부고민 기반 스킨케어 추천 API",
    version="3.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 4. 유틸: ingredients 컬럼 표준화
# -----------------------------

def standardize_ingredients_column(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "ingredients", "Ingredients", "INGREDIENTS",
        "ingredient", "Ingredient",
        "ingredient_list", "ingredients_list",
        "inci", "inci_list",
        "raw_ingredients",
    ]
    for c in candidates:
        if c in df.columns:
            if c != "ingredients":
                df = df.rename(columns={c: "ingredients"})
            break
    return df

# -----------------------------
# 5. 모델 / 데이터 로딩
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()

scores_df = pd.read_csv(PRODUCT_SCORES_PATH, encoding="utf-8-sig")
meta_df = pd.read_csv(PRODUCT_META_PATH, encoding="utf-8-sig")

# 컬럼 표준화
scores_df = standardize_ingredients_column(scores_df)
meta_df = standardize_ingredients_column(meta_df)

# 필수 컬럼 체크
for col in ["product_id", "product_name", "brand_name"] + SKIN_LABELS:
    if col not in scores_df.columns:
        raise RuntimeError(f"product_scores.csv 에 '{col}' 컬럼이 없습니다.")

for col in ["product_id", "sephora_url", "image_url"]:
    if col not in meta_df.columns:
        raise RuntimeError(f"product_meta.csv 에 '{col}' 컬럼이 없습니다.")

# 타입/결측 처리
scores_df["product_id"] = scores_df["product_id"].astype(str).str.strip()
meta_df["product_id"] = meta_df["product_id"].astype(str).str.strip()

scores_df["ingredients"] = scores_df.get("ingredients", "").fillna("").astype(str)

# -----------------------------
# 리뷰 데이터 로딩
# -----------------------------

_reviews_df = None

def load_reviews_df() -> pd.DataFrame:
    global _reviews_df
    if _reviews_df is None:
        df = pd.read_csv(REVIEWS_PATH, encoding="utf-8-sig")

        df["product_id"] = df["product_id"].astype(str).str.strip()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "rating_review" in df.columns:
            df["rating_review"] = pd.to_numeric(df["rating_review"], errors="coerce")

        if "helpfulness" in df.columns:
            df["helpfulness"] = pd.to_numeric(df["helpfulness"], errors="coerce")

        if "is_recommended" in df.columns:
            df["is_recommended"] = (
                df["is_recommended"]
                .astype(str)
                .str.lower()
                .isin(["1", "true", "yes", "y"])
            )

        _reviews_df = df

        print("[INFO] reviews_df rows:", len(_reviews_df))
        print("[INFO] reviews_df columns:", _reviews_df.columns.tolist())

    return _reviews_df


# -----------------------------
# 6. merge (행 기준: meta)
# -----------------------------

full_df = scores_df.merge(
    meta_df[["product_id", "sephora_url", "image_url"]],
    on="product_id",
    how="inner",
)

# ingredients suffix 안전 처리
if "ingredients" not in full_df.columns:
    if "ingredients_x" in full_df.columns:
        full_df["ingredients"] = full_df["ingredients_x"]
    else:
        full_df["ingredients"] = ""

full_df["ingredients"] = full_df["ingredients"].fillna("").astype(str)

# 안전장치: 이미지 없는 제품 제거
full_df = full_df[full_df["image_url"].astype(str).str.strip() != ""].copy()

print("[INFO] scores_df rows:", len(scores_df))
print("[INFO] meta_df rows:", len(meta_df))
print("[INFO] full_df rows (service):", len(full_df))
print("[INFO] full_df columns:", full_df.columns.tolist())

# -----------------------------
# 7. 스키마
# -----------------------------

class RecommendRequest(BaseModel):
    skin_type: str
    category: str = "all"
    top_k: int = 10
    concerns: List[str] = []

class RecommendItem(BaseModel):
    product_id: str
    product_name: str
    brand_name: str
    score: float
    sephora_url: str
    image_url: str

class RecommendResponse(BaseModel):
    skin_type: str
    category: str
    concerns: List[str]
    items: List[RecommendItem]

class ProductDetailResponse(BaseModel):
    product_id: str
    product_name: str
    brand_name: str
    ingredients: str = ""
    category: str = ""
    sub_category: str = ""
    sephora_url: str = ""
    image_url: str = ""
    scores: Dict[str, float]

# -----------------------------
# 8. 카테고리 필터
# -----------------------------

def filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    if category == "all":
        return df

    name = df["product_name"].str.lower()

    if category == "cleanser":
        return df[name.str.contains("cleanser|cleansing|foam|wash")]
    if category == "toner":
        return df[name.str.contains("toner|pad|essence")]
    if category == "serum":
        return df[name.str.contains("serum|ampoule")]
    if category == "cream":
        return df[name.str.contains("cream|moisturizer|lotion")]
    if category == "mask":
        return df[name.str.contains("mask|pack")]
    if category == "sunscreen":
        return df[name.str.contains("sunscreen|sun")]
    if category == "device":
        return df[name.str.contains("device|tool|brush")]

    return df

# -----------------------------
# 9. 피부 고민 점수
# -----------------------------

def calc_concern_score(ingredients: str, concerns: List[str]) -> float:
    if not concerns:
        return 0.0

    text = ingredients.lower()
    score = 0.0

    for c in concerns:
        rule = CONCERN_RULES.get(c)
        if not rule:
            continue

        for kw in rule["positive"]:
            if kw in text:
                score += 1.0
        for kw in rule["negative"]:
            if kw in text:
                score -= 1.0

    return score

# -----------------------------
# 10. 엔드포인트
# -----------------------------

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if req.skin_type not in SKIN_LABELS:
        raise HTTPException(status_code=400, detail="잘못된 피부타입")

    unknown = [c for c in req.concerns if c not in CONCERN_RULES]
    if unknown:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 고민: {unknown}")

    df = filter_by_category(full_df.copy(), req.category)
    if df.empty:
        return RecommendResponse(
            skin_type=req.skin_type,
            category=req.category,
            concerns=req.concerns,
            items=[],
        )

    df["base_score"] = df[req.skin_type].astype(float)
    df["concern_score"] = df["ingredients"].apply(lambda x: calc_concern_score(x, req.concerns))
    df["final_score"] = df["base_score"] * 0.8 + df["concern_score"] * 0.2

    df = df.sort_values("final_score", ascending=False).head(req.top_k)

    items = [
        RecommendItem(
            product_id=row["product_id"],
            product_name=row["product_name"],
            brand_name=row["brand_name"],
            score=float(row["final_score"]),
            sephora_url=row["sephora_url"],
            image_url=row["image_url"],
        )
        for _, row in df.iterrows()
    ]

    return RecommendResponse(
        skin_type=req.skin_type,
        category=req.category,
        concerns=req.concerns,
        items=items,
    )

@app.get("/product/{product_id}", response_model=ProductDetailResponse)
def get_product(product_id: str):
    if "product_id" not in full_df.columns:
        raise HTTPException(status_code=500, detail="full_df에 product_id 컬럼이 없습니다.")

    row_df = full_df[full_df["product_id"].astype(str) == str(product_id)]
    if row_df.empty:
        raise HTTPException(status_code=404, detail="Not Found")

    row = row_df.iloc[0]

    scores = {
        "combination": float(row.get("combination", 0.0)),
        "dry": float(row.get("dry", 0.0)),
        "normal": float(row.get("normal", 0.0)),
        "oily": float(row.get("oily", 0.0)),
    }

    return ProductDetailResponse(
        product_id=str(row.get("product_id", "")),
        product_name=str(row.get("product_name", "")),
        brand_name=str(row.get("brand_name", "")),
        ingredients=str(row.get("ingredients", "") if pd.notna(row.get("ingredients", "")) else ""),
        category=str(row.get("category", "") if pd.notna(row.get("category", "")) else ""),
        sub_category=str(row.get("sub_category", "") if pd.notna(row.get("sub_category", "")) else ""),
        sephora_url=str(row.get("sephora_url", "") if pd.notna(row.get("sephora_url", "")) else ""),
        image_url=str(row.get("image_url", "") if pd.notna(row.get("image_url", "")) else ""),
        scores=scores,
    )

@app.get("/product/{product_id}/reviews")
def get_product_reviews(
    product_id: str,
    sort: str = Query("recent", pattern="^(recent|helpful|rating|recommended)$"),
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
):
    reviews_df = load_reviews_df()
    pid = str(product_id)

    sub = reviews_df[reviews_df["product_id"] == pid].copy()
    total = len(sub)

    if total == 0:
        return {
            "product_id": pid,
            "total": 0,
            "limit": limit,
            "offset": offset,
            "items": [],
        }

    # 정렬
    if sort == "recent":
        sub = sub.sort_values("date", ascending=False, na_position="last")
    elif sort == "helpful":
        if "helpfulness" in sub.columns:
            sub = sub.sort_values(
                ["helpfulness", "date"],
                ascending=[False, False],
                na_position="last",
            )
    elif sort == "rating":
        if "rating_review" in sub.columns:
            sub = sub.sort_values(
                ["rating_review", "date"],
                ascending=[False, False],
                na_position="last",
            )
    elif sort == "recommended":
        if "is_recommended" in sub.columns:
            sub = sub.sort_values(
                ["is_recommended", "date"],
                ascending=[False, False],
                na_position="last",
            )

    page = sub.iloc[offset : offset + limit]

    items = page[
        [
            c for c in [
                "review_id",
                "rating_review",
                "is_recommended",
                "helpfulness",
                "date",
                "review_text",
                "user_skin_type",
            ]
            if c in page.columns
        ]
    ].copy()

    if "date" in items.columns:
        items["date"] = items["date"].astype("string")

    return {
        "product_id": pid,
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items.to_dict(orient="records"),
    }


# -----------------------------
# 11. 실행
# -----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_app:app", host="0.0.0.0", port=8000, reload=True)
