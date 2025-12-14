# remove_no_image_products.py
#
# data/product_meta.csv에서 image_url이 비어 있는 행을 삭제합니다.
# 원본은 자동으로 백업 파일로 저장합니다.

from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
META_PATH = BASE / "data" / "product_meta.csv"

def is_blank(x) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in ["none", "nan", "null"]

def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"모든 인코딩 시도 실패: {last_error}"
    )

def main():
    if not META_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {META_PATH}")

    # 인코딩 안전 로드
    df = read_csv_safe(META_PATH)

    if "image_url" not in df.columns:
        raise ValueError("product_meta.csv에 'image_url' 컬럼이 없습니다.")

    before = len(df)

    # image_url이 비어 있는 행 삭제
    keep_mask = ~df["image_url"].apply(is_blank)
    removed_df = df[~keep_mask].copy()
    df2 = df[keep_mask].copy()

    after = len(df2)
    removed = before - after

    # 백업 저장
    backup_path = META_PATH.with_name(
        "product_meta_backup_before_remove_no_image.csv"
    )
    df.to_csv(backup_path, index=False, encoding="utf-8-sig")

    # 삭제 후 저장
    df2.to_csv(META_PATH, index=False, encoding="utf-8-sig")

    print("[완료]")
    print("원본 행 수:", before)
    print("삭제된 행 수(image_url 비어있음):", removed)
    print("남은 행 수:", after)
    print("백업 파일:", backup_path)

    if removed > 0:
        cols = [
            c for c in
            ["product_id", "brand_name", "product_name", "sephora_url", "image_url"]
            if c in removed_df.columns
        ]
        print("\n[삭제된 제품 예시(최대 10개)]")
        print(removed_df[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
