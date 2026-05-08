# 🧠 Employee Mental Health Prediction & Care System
우울 일기(데이터 수집), DNN(위험도 분석), 상담 챗봇(심리 케어)이 통합된 사내 맞춤형 EAP 플랫폼
<br>

1. 프로젝트 개요

작업 기간: 2026.04.06 ~ 2026.04.10 (1인 프로젝트)

임직원이 작성한 '우울 일기' 데이터를 기반으로 DNN 모델이 우울 위험도를 예측하고, 결과에 따라 로컬 LLM 상담 챗봇이 맞춤형 위로와 처방을 제공하는 통합 멘탈케어 시스템입니다.

주요 역할: 감정 일기 인터페이스 설계, DNN 분류 모델링, 로컬 LLM 기반 상담 파이프라인 구축 (기여도 100%)
<br>
<br>
<br>
2. 사용 기술 (Tech Stack)
<br>
<br>
Language: Python

Deep Learning: TensorFlow(Keras), Scikit-learn

Natural Language Processing: LangChain, Ollama (Gemma2), Chroma DB

Frontend & UI: Streamlit (일기 및 챗봇 인터페이스 구현)

Backend & DB: Supabase, PostgreSQL, Pandas
<br>
<br>
<br>
3. 핵심 구현 내용 (Key Features)
<br>
✍️ 감정 기록 시스템: 우울 일기 (Depression Diary)<br>
사용자가 일상에서 느끼는 감정을 텍스트로 기록할 수 있는 인터페이스 구현.

기록된 일기 데이터는 단순 저장에 그치지 않고, 딥러닝 모델의 분석을 위한 비정형 데이터 소스로 활용.
<br>
📊 심층 신경망(DNN) 기반 실시간 위험도 분석<br>
일기 내용과 임직원 활동 데이터를 전처리하여 DNN 모델에 입력, 현재 우울 위험도를 실시간 분류.

수치형 데이터와 텍스트 기반 피처를 결합하여 다각적인 위험도 판단 로직 설계.
<br>
💬 맞춤형 AI 상담 챗봇 (Counseling Chatbot)<br>
로컬 LLM(Ollama)을 활용하여 분석된 위험도 등급에 최적화된 상담 시나리오 제공.

RAG(검색 증강 생성) 기법을 적용하여 사용자의 일기 내용을 바탕으로 공감대를 형성하고, 검증된 심리 처방 가이드를 답변에 반영.

보안을 위해 모든 상담 내용은 외부 유출 없이 사내망 내에서만 처리되도록 설계.
<br>
<br>
<br>
4. 기술적 의사결정 (Technical Decision)
<br>
'일기-분석-상담' 파이프라인 구축

이유: 단순 예측 서비스는 사용자에게 불안감만 줄 수 있다고 판단함. '일기를 통한 감정 배설 - 정확한 상태 진단 - 챗봇을 통한 즉각적인 케어'라는 선순환 구조를 통해 실질적인 솔루션을 제공하고자 함.
<br>
<br>
로컬 LLM (Gemma2) 기반 RAG 시스템

이유: 일기 내용은 매우 민감한 개인 정보이므로 클라우드 API 대신 로컬 환경에서 구동되는 모델을 사용함. RAG를 통해 모델이 사용자의 이전 일기 내용을 기억하고 대화의 맥락을 유지하도록 구현.
<br>
<br>
https://github.com/user-attachments/assets/0da87d21-dd89-4e5d-b373-2d448d7d01f5 로그인 후 시연 영상

https://github.com/user-attachments/assets/29908ae0-c6b8-484e-9bd8-3488bfb2d827 LLM 맞춤 처방전 시연 영상


https://github.com/user-attachments/assets/341cbe4c-6f4e-4757-9cc5-3f7d055ce458 심리 상담사 챗봇 시연 영상


https://github.com/user-attachments/assets/17d1b19e-d814-4c1a-b092-cdd92ba8c797 우울증 예측 일기장 시연 영상


<br>
<br>
5. 트러블슈팅 및 배운 점 (Troubleshooting & Lessons Learned)<br>
💡 텍스트 데이터와 수치 데이터의 결합 (Data Fusion)<br>
이슈: 텍스트(일기)의 감정 점수와 정형 데이터(활동 지표)를 어떻게 하나의 모델에서 효과적으로 처리할지에 대한 고민 발생.

해결: 일기 텍스트를 감정 분석 라이브러리를 통해 수치화하여 DNN의 가중치에 반영하는 방식을 실험함. 이 과정을 통해 다양한 형태의 데이터를 정제하고 융합하는 데이터 엔지니어링 역량을 확보함.
<br>
<br>
⚙️ 대화 맥락 유지와 시스템 부하 관리<br>
이슈: 상담 챗봇이 일기 내용을 모두 기억하게 하면 시스템 부하가 커지는 문제 발생.

교훈: LangChain의 메모리 관리 기능을 활용하여 대화의 핵심 요약본만 참조하게 함으로써, 성능 저하 없이 매끄러운 상담 흐름을 유지하는 최적화 기법을 습득함.
