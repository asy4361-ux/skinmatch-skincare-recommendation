# create_meta_from_scores.py
#
# out/product_scores.csv 를 기반으로
# 서비스에서 사용할 메타 파일 data/product_meta.csv 를 만드는 스크립트입니다.

import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

SCORES_PATH = BASE / "out" / "product_scores.csv"
META_PATH = BASE / "data" / "product_meta.csv"

def main():
    print(f"[INFO] product_scores 로딩: {SCORES_PATH}")
    df = pd.read_csv(SCORES_PATH, encoding="utf-8-sig")

    # 필수 컬럼 체크
    required_cols = ["product_id", "product_name", "brand_name"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"product_scores.csv 에 '{col}' 컬럼이 없습니다.")

    # 결측 방지 및 문자열 통일
    df["product_id"] = df["product_id"].astype(str)
    df["product_name"] = df["product_name"].fillna("").astype(str)
    df["brand_name"] = df["brand_name"].fillna("").astype(str)

    # 우리가 메타 파일에서 사용할 최소 컬럼만 남기기
    meta = df[["product_id", "brand_name", "product_name"]].copy()

    # 아직 비어 있는 메타데이터 컬럼 추가
    meta["sephora_url"] = ""      # 세포라 제품 상세 URL
    meta["image_url"] = ""        # 대표 이미지 URL

    # 저장
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(META_PATH, index=False, encoding="utf-8-sig")

    print(f"[INFO] 메타 파일 저장 완료: {META_PATH}")
    print("[INFO] 행 수:", len(meta))
    print("[INFO] 컬럼:", meta.columns.tolist())


if __name__ == "__main__":
    main()
