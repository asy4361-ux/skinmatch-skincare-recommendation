\# SkinMatch: Skincare Recommendation System



본 프로젝트는 스킨케어 제품 추천을 위한 웹 기반 추천 시스템으로,

제품 성분 텍스트와 사용자 리뷰 데이터를 기반으로 피부타입별 적합도를 산출한다.



---



\## 1. 프로젝트 개요



\- 목적: 사용자 피부타입 및 피부 고민에 맞는 스킨케어 제품 추천

\- 데이터: Kaggle의 \*Sephora Products and Skincare Reviews\* 데이터셋

\- 모델: DistilBERT 기반 성분 텍스트 임베딩 + 다중 출력 회귀

\- 서비스: FastAPI 기반 추천 API + HTML/JS 웹 인터페이스



---



\## 2. 데이터 구성



본 프로젝트에서는 다음 두 종류의 데이터를 사용하였다.



1\. 제품 데이터  

&nbsp;  - 제품명, 브랜드, 성분 텍스트, 제품 URL, 이미지 URL



2\. 사용자 리뷰 데이터  

&nbsp;  - 리뷰 텍스트 및 평점 정보



원본 데이터는 Kaggle 공개 데이터셋을 기반으로 하였으며,

제출용 저장소에는 원본 데이터 파일은 포함하지 않았다.



---



\## 3. 모델 설명



\- DistilBERT (distilbert-base-uncased)를 사용하여 성분 텍스트를 임베딩

\- 하나의 입력에 대해 4개 피부타입(복합성, 건성, 중성, 지성)에 대한 점수를 동시에 예측하는 다중 출력 회귀 구조

\- 평가 지표: MAE (Mean Absolute Error)



---



\## 4. 시스템 구조



1\. 데이터 전처리

2\. 성분 텍스트 기반 점수 산출

3\. 제품별 피부타입 점수 저장

4\. 제품 메타데이터와 결합

5\. FastAPI를 통한 추천 API 제공

6\. 웹 화면에서 추천 결과 출력



---



\## 5. 실행 환경



\- Python 3.10+

\- FastAPI

\- HuggingFace Transformers

\- scikit-learn

\- Google Colab (모델 학습)



---



\## 6. 비고



\- 모델 가중치 및 대용량 데이터 파일은 저장소에 포함하지 않음

\- 제출 및 코드 검토 목적의 경량 저장소 구성

\-백엔드 실행:

\-uvicorn api_app2:app --reload

\-프론트:

\-index2.html을 브라우저로 열기


