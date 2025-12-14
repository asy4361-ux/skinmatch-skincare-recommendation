import os
import re
import csv
from pathlib import Path
from typing import List, Optional, Dict, Iterable
from collections import Counter
import pandas as pd

# =========================
# 0) 경로/파라미터
# =========================
def resolve_data_dir() -> Path:
    env_dir = os.getenv("DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    mnt = Path("/mnt/data")
    if mnt.exists():
        return mnt
    return Path(__file__).resolve().parent / "data"

ROOT = Path(__file__).resolve().parent
DATA = resolve_data_dir()
PROCESSED = ROOT / "processed"
OUT = ROOT / "out"
PROCESSED.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)

PRODUCTS_PATH = DATA / "product_info.csv"
REVIEW_FILES = sorted(DATA.glob("reviews_*.csv")) or ([DATA / "reviews_1.csv"] if (DATA / "reviews_1.csv").exists() else [])

HELPFULNESS_MIN = float(os.getenv("HELPFULNESS_MIN", 0.3))
POS_RATING_MIN  = float(os.getenv("POS_RATING_MIN", 4))
CHUNKSIZE       = int(os.getenv("CSV_CHUNKSIZE", "0"))  # 0이면 일반 로딩, 그 외엔 청크로

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROCESSED = ROOT / "processed"
OUT = ROOT / "out"
PROCESSED.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

PRODUCTS_PATH = DATA / "product_info.csv"
REVIEW_PATTERNS = [DATA / f"reviews_{i}.csv" for i in range(1, 6)]

# =========================
# 1) 유틸: 안전 로딩/정규화/보조
# =========================
def read_csv_smart(path: Path, usecols: List[str] = None, chunksize: int = None) -> pd.DataFrame:
    encs = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    last_err = None
    for enc in encs:
        # 1) 먼저 C 엔진(기본)으로 시도: low_memory/engine/on_bad_lines 모두 생략
        try:
            kw = dict(encoding=enc)
            if usecols is not None:
                kw["usecols"] = usecols
            if chunksize:
                chunks = []
                for ch in pd.read_csv(path, chunksize=chunksize, **kw):
                    chunks.append(ch)
                return pd.concat(chunks, ignore_index=True)
            return pd.read_csv(path, **kw)
        except Exception as e:
            last_err = e
            # 2) 실패 시 python 엔진으로 재시도: on_bad_lines만 추가, low_memory 절대 X
            try:
                kw = dict(encoding=enc, engine="python", on_bad_lines="skip")
                if usecols is not None:
                    kw["usecols"] = usecols
                if chunksize:
                    chunks = []
                    for ch in pd.read_csv(path, chunksize=chunksize, **kw):
                        chunks.append(ch)
                    return pd.concat(chunks, ignore_index=True)
                return pd.read_csv(path, **kw)
            except Exception as e2:
                last_err = e2
                continue
    raise last_err

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return df

def choose_col(df: pd.DataFrame, aliases: Iterable[str], required=False, note="") -> Optional[str]:
    for a in aliases:
        if a in df.columns:
            return a
    if required:
        raise KeyError(f"[스키마 매핑 실패] {note} 후보={list(aliases)}")
    return None

def safe_dropna(df: pd.DataFrame, cols_existing: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in cols_existing if c in df.columns]
    return df.dropna(subset=cols) if cols else df

def safe_len_filter(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        return df[df[col].astype(str).str.strip().str.len() > 0]
    return df

def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================
# 2) 스킨케어 필터 키워드
# =========================
ALLOW_ANY = [
    "cleanser","cleansing","makeup remover","micellar",
    "toner","essence","serum","ampoule",
    "moisturizer","moisturiser","cream","lotion","gel-cream",
    "eye cream","eye serum","eye treatment",
    "treatment","acne treatment","spot treatment",
    "sunscreen","sun screen","spf",
    "mask","face mask","sheet mask","wash-off mask",
    "exfoliator","peel","peeling","aha","bha","pha",
    "face wash","face oil","mist","hydrator","emulsion","balm","cleansing oil","cleansing balm"
]
DENY_ANY = [
    "lip","lip balm","lip mask","lip treatment","lipstick","tint","gloss",
    "brow","lash","mascara","eyeliner","eyeshadow","blush","highlighter","bronzer",
    "foundation","concealer","primer","palette","setting spray","setting powder",
    "nail","polish",
    "fragrance","perfume","cologne",
    "hair","shampoo","conditioner","scalp",
    "body","hand cream","foot","deodorant",
    "tool","brush","roller","gua sha","device",
    "candle","home","gift","set","mini","kit","value","bundle","sampler","travel size","trial"
]

def contains_any(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in keywords)

# =========================
# 3) 리뷰 로드/정제
# =========================
def load_reviews() -> pd.DataFrame:
    paths = [p for p in REVIEW_PATTERNS if p.exists()]
    if not paths:
        raise FileNotFoundError("[ERROR] reviews_*.csv 파일을 찾지 못했습니다.")

    dfs = []
    for p in paths:
        df = read_csv_smart(p, chunksize=CHUNKSIZE or None)
        df = normalize_columns(df)

        # 스키마 표준화
        rename_map = {
            "rating": "rating_review",
            "rating_value": "rating_review",
            "review": "review_text",
            "text": "review_text",
            "content": "review_text",
            "helpful_ratio": "helpfulness",
            "helpful_score": "helpfulness",
            "submission_time": "date",
        }
        hit = {k: v for k, v in rename_map.items() if k in df.columns}
        if hit:
            df = df.rename(columns=hit)

        # 보존 컬럼(있으면 유지)
        KEEP_REVIEW_COLS = [
            "product_id",
            "rating_review", "is_recommended", "helpfulness",
            "total_feedback_count",  # 필요 없으면 이 줄을 주석처리/삭제
            "date", "review_text",
            "skin_type"              # 검증용 선택 컬럼
        ]
        cols = [c for c in KEEP_REVIEW_COLS if c in df.columns]
        df = df[cols].copy()

        # 텍스트/타입 정리
        if "review_text" in df.columns:
            # 1) NaN 제거
            df = df[df["review_text"].notna()]

            # 2) 문자열 클리닝
            df["review_text"] = (
                df["review_text"]
                .astype(str)
                .str.replace(r"<[^>]+>", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

            # 3) 공백 문자열 제거
            df = df[df["review_text"].str.len() > 0]

        for numc in ["rating_review","helpfulness","total_feedback_count"]:
            if numc in df.columns:
                df[numc] = pd.to_numeric(df[numc], errors="coerce")

        # 개인정보/중복 메타 제거
        for dropc in ["author_id","total_neg_feedback_count","total_pos_feedback_count","review_title","eye_color","hair_color","skin_tone","product_name","brand_name","price_usd"]:
            if dropc in df.columns:
                df.drop(columns=dropc, inplace=True)

        dfs.append(df)

    reviews = pd.concat(dfs, ignore_index=True)

    # 컬럼명 정리(선택)
    if "skin_type" in reviews.columns:
        reviews = reviews.rename(columns={"skin_type": "user_skin_type"})

    # 결측/중복 제거
    reviews = reviews.dropna(subset=["product_id","review_text"])
    reviews["product_id"] = reviews["product_id"].astype(str).str.strip()
    keep_keys = [c for c in ["product_id","review_text"] if c in reviews.columns]
    if keep_keys:
        reviews = reviews.drop_duplicates(subset=keep_keys)

    reviews.to_csv(PROCESSED / "reviews_selected.csv", index=False, encoding="utf-8-sig")
    return reviews

# =========================
# 4) 제품 로드/정제
# =========================
def load_products() -> pd.DataFrame:
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"제품 파일이 없습니다: {PRODUCTS_PATH}")

    pr = read_csv_smart(PRODUCTS_PATH, chunksize=CHUNKSIZE or None)
    pr = normalize_columns(pr)

    # 표준 이름 매핑
    rename_map = {
        "id": "product_id",
        "sku": "product_id",
        "name": "product_name",
        "title": "product_name",
        "brand": "brand_name",
        "brandname": "brand_name",
        "ingredient_list": "ingredients",
        "inci": "ingredients",
        "avg_rating": "rating",
        "rating_value": "rating",
    }
    hit = {k: v for k, v in rename_map.items() if k in pr.columns}
    if hit:
        pr = pr.rename(columns=hit)

    # 보존 컬럼(있으면 유지)
    KEEP_PRODUCT_COLS = [
        "product_id", "product_name", "brand_name", "ingredients",
        "rating", "loves_count", "reviews", "price_usd",
        "primary_category", "secondary_category", "tertiary_category"
    ]
    cols = [c for c in KEEP_PRODUCT_COLS if c in pr.columns]
    products = pr[cols].copy()

    # 후처리
    for c in ["product_id","product_name","brand_name"]:
        if c in products.columns:
            products[c] = products[c].astype(str).str.strip()
    if "ingredients" in products.columns:
        products["ingredients"] = products["ingredients"].fillna("").astype(str)

    # category 단일화(secondary→tertiary 우선)
    if "secondary_category" in products.columns or "tertiary_category" in products.columns:
        sec = products.get("secondary_category", "")
        ter = products.get("tertiary_category", "")
        products["category"] = sec.mask(sec.eq("") | sec.isna(), ter)

    # 제품 평균 평점 컬럼명 통일
    if "rating" in products.columns:
        products = products.rename(columns={"rating": "rating_product"})

    products.to_csv(PROCESSED / "products_selected.csv", index=False, encoding="utf-8-sig")
    return products

# =========================
# 5) 병합 및 클린
# =========================
def join_and_clean(reviews: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    keep_for_join = ["product_id","product_name","brand_name","ingredients","category","rating_product","price_usd"]
    keep_for_join = [c for c in keep_for_join if c in products.columns]
    merged = pd.merge(reviews, products[keep_for_join], on="product_id", how="inner")

    # 1) 카테고리 기반 필터링: 립/미니/툴/셀프태너/웰니스/기프트세트 제거
    if "category" in merged.columns:
        cat = merged["category"].astype(str).str.lower()

        deny_patterns = [
            r"lip balm",
            r"lip balms",
            r"lip balms? & treatments",
            r"lip care",

            r"mini size",
            r"\bmini\b",

            r"tool",
            r"tools",

            r"self tanner",
            r"self tanners",

            r"wellness",

            r"value & gift set",
            r"value & gift sets",
            r"gift set",
            r"gift sets",
        ]
        deny_regex = "|".join(deny_patterns)
        mask_deny = cat.str.contains(deny_regex, na=False)
        merged = merged[~mask_deny]

    # 2) 기본 결측/텍스트 필터
    merged = safe_dropna(merged, ["review_text","product_id"])
    merged = safe_len_filter(merged, "review_text")

    # 3) 불필요 컬럼 제거
    merged = merged.drop(
        columns=[c for c in ["author_id","total_neg_feedback_count","total_pos_feedback_count","review_title"]
                 if c in merged.columns],
        errors="ignore",
    )
    merged = merged.loc[:, ~merged.columns.str.contains(r"^unnamed[:\s_]*0$", case=False)]

    # 4) 필수 컬럼 보정 (없으면 Na로 만들어 두기)
    for c in [
        "product_name","brand_name","category","ingredients","rating_product","price_usd",
        "rating_review","is_recommended","helpfulness","date","review_text","user_skin_type"
    ]:
        if c not in merged.columns:
            merged[c] = pd.NA

    # 5) 컬럼 순서 정리
    front = [
        "product_id","product_name","brand_name","category","ingredients","rating_product","price_usd",
        "rating_review","is_recommended","helpfulness","date","review_text","user_skin_type",
        "total_feedback_count"
    ]
    front = [c for c in front if c in merged.columns]
    back  = [c for c in merged.columns if c not in front]
    merged = merged[front + back]

    merged.reset_index(drop=True, inplace=True)

    # 6) 중복 리뷰 정리 (최신/도움됨/평점 우선)
    merged = dedupe_reviews(merged)

    # 7) 여기서 결측치 후처리

    # 7-1) review_text / product_id 결측/공백 행 최종 제거
    merged = merged[merged["product_id"].notna()].copy()
    merged = merged[merged["review_text"].notna()].copy()
    merged = merged[merged["review_text"].astype(str).str.strip() != ""].copy()

    # 7-2) ingredients 결측 → 빈 문자열로 채우기 (성분 정보 없음)
    if "ingredients" in merged.columns:
        merged["ingredients"] = (
            merged["ingredients"]
            .astype("string")
            .fillna("")
            .replace(["nan", "NaN", "None"], "")
        )

    # 7-3) user_skin_type 결측 → "unknown"으로 채우기
    if "user_skin_type" in merged.columns:
        merged["user_skin_type"] = merged["user_skin_type"].fillna("unknown")

    # 7-4) is_recommended 결측 → 0으로 채우기 (추천 여부 정보 없음 = 0)
    if "is_recommended" in merged.columns:
        merged["is_recommended"] = merged["is_recommended"].fillna(0)

    return merged


def dedupe_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    review_text + product_id 기준 중복 리뷰를 제거하되,
    (date 최신 → helpfulness 높음 → rating_review 높음) 순으로 우선순위를 두고 가장 좋은 리뷰 1개만 남긴다.
    """
    # 날짜 정제 (문자 → datetime)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    # helpfulness 정제
    if "helpfulness" in df.columns:
        df["helpfulness"] = pd.to_numeric(df["helpfulness"], errors="coerce").fillna(0)
    else:
        df["helpfulness"] = 0

    # rating_review 정제
    if "rating_review" in df.columns:
        df["rating_review"] = pd.to_numeric(df["rating_review"], errors="coerce").fillna(0)
    else:
        df["rating_review"] = 0

    # 최신 리뷰 + 품질 우선순위 정렬
    df = df.sort_values(
        by=["date", "helpfulness", "rating_review"],
        ascending=[False, False, False]
    )

    # review_text + product_id 중복 제거
    df = df.drop_duplicates(subset=["product_id", "review_text"], keep="first")

    return df

# =========================
# 6) 문장 분할/긍정 선택/키워드 태깅 및 스코어
# =========================
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
NEG_TRIGGERS = [
    r"\bnot work(s|ed)?\b", r"didn'?t work", r"\bno result(s)?\b", r"\bwaste\b", r"\bdisappoint(ed|ing)\b",
    r"\bbroke me out\b", r"\bbreakout(s)?\b", r"\bclog(ged|s)? pore(s)?\b", r"\bcomedogenic\b",
    r"\birritat(e|ed|ing|ion)\b", r"\bsting(ing)?\b", r"\bburn(ing)?\b", r"\bitch(ing)?\b", r"\brash\b",
    r"\bgreasy\b", r"\boily\b", r"\bsticky\b", r"\btacky\b",
]
NEG_RE = re.compile("|".join(NEG_TRIGGERS), flags=re.I)

SKIN_TYPE_PATTERNS = {
    "dry":        [r"\bdry\b", r"\bvery dry\b", r"\bdehydrated\b"],
    "oily":       [r"\boily\b", r"\boil(?:y|iness)\b", r"\bgreasy\b", r"\bshiny\b"],
    "combination":[r"\bcombination\b", r"\bcombo\b", r"\boily t-?zone\b"],
    "normal":     [r"\bnormal\b", r"\bbalanced\b"],
    "sensitive":  [r"\bsensitive\b", r"\birritation-?prone\b", r"\breactive\b"],
}
CONCERN_PATTERNS = {
    "acne":      [r"\bacne\b", r"\bbreakout(s)?\b", r"\bblemish(es)?\b", r"\bwhitehead(s)?\b", r"\bblackhead(s)?\b"],
    "redness":   [r"\bredness\b", r"\brosea(?:cea)?\b", r"\bflushed\b"],
    "wrinkles":  [r"\bwrinkle(s)?\b", r"\bfine line(s)?\b", r"\baging\b", r"\banti-?aging\b"],
    "pores":     [r"\b(pore|pores)\b", r"\bclogged\b", r"\benlarged pores\b"],
    "dryness":   [r"\bdryness\b", r"\bdehydration\b"],
    "oiliness":  [r"\boiliness\b", r"\btoo oily\b", r"\bshiny\b"],
    "sensitivity":[r"\bsensitivity\b", r"\birritation\b", r"\bstinging\b", r"\bburning\b"],
    "dullness":  [r"\bdullness\b", r"\bdull\b", r"\black of glow\b", r"\bbrighten(ing)?\b"],
    "texture":   [r"\btexture\b", r"\brough\b", r"\bbumpy\b"],
    "hyperpigmentation":[r"\bdark spot(s)?\b", r"\bpigmentation\b", r"\bmelasma\b", r"\buneven tone\b"],
}

def has_any(patterns, text: str) -> bool:
    return any(re.search(p, text) for p in patterns)

def which_keys(pattern_dict: Dict[str, List[str]], text: str) -> List[str]:
    found = []
    for k, pats in pattern_dict.items():
        if has_any(pats, text):
            found.append(k)
    return found

def is_positive_sentence(sent: str, rating: Optional[float], min_rating: float = POS_RATING_MIN) -> bool:
    if rating is None or rating < min_rating:
        return False
    return not bool(NEG_RE.search(sent))

def sentence_and_score(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    # 문장 분할
    merged["sentences"] = merged["review_text"].astype(str).map(lambda s: [x.strip() for x in SENT_SPLIT_RE.split(s) if x.strip()])
    # 긍정 문장
    merged["positive_sentences"] = merged.apply(
        lambda r: [s for s in r["sentences"] if is_positive_sentence(s.lower(), pd.to_numeric(r.get("rating_review"), errors="coerce"))],
        axis=1
    )

    # 제품×피부타입/고민 카운트
    skin_counter, concern_counter = Counter(), Counter()
    for _, row in merged.iterrows():
        pid = row["product_id"]
        for s in row["positive_sentences"]:
            s_l = s.lower()
            for st in set(which_keys(SKIN_TYPE_PATTERNS, s_l)):
                skin_counter[(pid, st)] += 1
            for cc in set(which_keys(CONCERN_PATTERNS, s_l)):
                concern_counter[(pid, cc)] += 1

    # 점수화(Laplace smoothing)
    ALPHA, BETA = 1.0, 1.0
    def counter_to_df(counter, key_name):
        rows = []
        for (pid, key), pos in counter.items():
            score = (pos + ALPHA) / (pos + ALPHA + BETA)  # 0.5~1.0
            rows.append({"product_id": pid, key_name: key, "pos_count": int(pos), "score": float(score)})
        df = pd.DataFrame(rows).sort_values(["score","pos_count"], ascending=[False, False])
        meta_cols = [c for c in ["product_name","brand_name","category"] if c in merged.columns]
        if meta_cols:
            df = df.merge(merged[["product_id"]+meta_cols].drop_duplicates("product_id"), on="product_id", how="left")
        return df

    skin_df = counter_to_df(skin_counter, "skin_type")
    concern_df = counter_to_df(concern_counter, "concern")

    skin_df.to_csv(OUT / "product_skin_scores.csv", index=False, encoding="utf-8-sig")
    concern_df.to_csv(OUT / "product_concern_scores.csv", index=False, encoding="utf-8-sig")

    # 간단 요약: 피부타입별 상위 고민 예시
    top_rows = []
    for st in sorted(set(skin_df["skin_type"])) if len(skin_df) else []:
        sub = concern_df.sort_values(["score","pos_count"], ascending=[False, False]).head(50)
        top_rows.append({"skin_type": st, "top_concerns_example": ", ".join(sub["concern"].head(5).tolist())})
    pd.DataFrame(top_rows).to_csv(OUT / "product_top_concerns_by_skin.csv", index=False, encoding="utf-8-sig")

    return merged


# =========================
# 7) 저장
# =========================
def save_all(df: pd.DataFrame, fname: str):
    out_path = PROCESSED / fname
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 저장: {out_path}")

# =========================
# main
# =========================
def main():
    print(f"[INFO] DATA_DIR = {DATA}")
    print("[STEP] 1. 리뷰 로드/정제")
    reviews = load_reviews()
    print(f"[INFO] 리뷰 수(정제 후): {len(reviews):,}")

    print("[STEP] 2. 제품 로드/정제")
    products = load_products()
    print(f"[INFO] 제품 수(스킨케어 필터 후): {len(products):,}")

    print("[STEP] 3. 병합/정리")
    merged = join_and_clean(reviews, products)
    print(f"[INFO] 병합 후 행 수: {len(merged):,}")
    save_all(merged, "reviews_products_merged_clean.csv")

    print("[STEP] 4. 문장 분석/점수 산출")
    merged2 = sentence_and_score(merged)

if __name__ == "__main__":
    main()
