import pandas as pd

# 경로
PRODUCT_META_PATH = "./data/product_meta.csv"
REVIEWS_MERGED_PATH = "./processed/reviews_products_merged_clean.csv"
OUTPUT_REVIEWS_PATH = "./data/reviews_clean.csv"

# 로드
products = pd.read_csv(PRODUCT_META_PATH)
reviews = pd.read_csv(REVIEWS_MERGED_PATH)

# product_id 문자열 통일
products["product_id"] = products["product_id"].astype(str)
reviews["product_id"] = reviews["product_id"].astype(str)

# 1) product_meta 기준으로 리뷰 필터링
reviews = reviews[reviews["product_id"].isin(products["product_id"])].copy()

# 2) 날짜 파싱
if "date" in reviews.columns:
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

# 3) 추천 여부 정리
if "is_recommended" in reviews.columns:
    reviews["is_recommended"] = (
        reviews["is_recommended"]
        .astype(str)
        .str.lower()
        .isin(["1", "true", "yes", "y"])
    )

# 4) helpfulness 정리
if "helpfulness" in reviews.columns:
    reviews["helpfulness"] = pd.to_numeric(reviews["helpfulness"], errors="coerce")

# 5) review_id 생성
reviews = reviews.reset_index(drop=True)
reviews["review_id"] = reviews.index + 1

# 6) 리뷰용 컬럼만 선택
review_cols = [
    "review_id",
    "product_id",
    "rating_review",
    "is_recommended",
    "helpfulness",
    "date",
    "review_text",
    "user_skin_type",
]

reviews_clean = reviews[review_cols]

# 저장
reviews_clean.to_csv(OUTPUT_REVIEWS_PATH, index=False, encoding="utf-8-sig")

print("리뷰 정리 완료")
print(f"- 총 리뷰 수: {len(reviews_clean)}")
print(f"- 저장 위치: {OUTPUT_REVIEWS_PATH}")
