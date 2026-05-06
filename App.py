import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from supabase import create_client
import datetime
import plotly.graph_objects as go
import plotly.express as px
import shap
import re
import os
import time

# ✨ 새롭게 추가된 Langchain, Ollama, Chroma 라이브러리
# 수정 후 (이걸로 교체하세요!)
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------------------------------------
# ★ 로컬 LLM (Ollama) 세팅 ★
# --------------------------------------------------------
# 내 컴퓨터에서 돌고 있는 Gemma2 모델을 연결합니다. (API 키 불필요!)
llm_model = ChatOllama(model="gemma2:2b", temperature=0.7)

# 한국어 특화 임베딩 모델 (텍스트를 벡터로 변환)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --------------------------------------------------------
# 1. 초기 세션 및 화면 설정
# --------------------------------------------------------
st.set_page_config(page_title="Deep-Mind EAP Solution", layout="wide")

if 'user_name' not in st.session_state: st.session_state['user_name'] = ""
if 'page_mode' not in st.session_state: st.session_state['page_mode'] = 'test'
if 'view_record' not in st.session_state: st.session_state['view_record'] = None

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-title { font-size: calc(1.5rem + 1.2vw) !important; font-weight: 800; color: #0f172a; margin-bottom: 1rem; }
    .stAlert p { font-size: calc(0.8rem + 0.3vw) !important; word-break: keep-all; line-height: 1.5; }
    .stButton>button { width: 100%; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# 2. Supabase 연결
# --------------------------------------------------------
URL = "https://ghkgcmdyxbibgichoksf.supabase.co"
KEY = "sb_publishable_VvLOj0sru5rc05PmStMVJg_8LxC-cwo"
try:
    supabase = create_client(URL, KEY)
except:
    supabase = None

# --------------------------------------------------------
# 3. 리소스 로드 (DNN 모델 및 ✨ Chroma DB 생성)
# --------------------------------------------------------
@st.cache_resource
def load_resources():
    # 1. DNN 예측용 데이터 (이 부분은 기존과 동일)
    dnn_model = tf.keras.models.load_model('depression_model.h5')
    df = pd.read_csv('Depression.csv')
    df.columns = ['성별', '나이', '업무_압박감', '직업_만족도', '수면_시간', '식습관', '자해_충동_여부', '근무_시간', '재정적_스트레스', '정신질환_가족력', '우울증']
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns: df[col] = le.fit_transform(df[col])
    X = df.drop('우울증', axis=1); y = df['우울증']
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler(); scaler.fit(X_train)
    cause_df = pd.read_csv('Cause_Solutions.csv')
    background_data = scaler.transform(X_train)[:100]
    
    # ========================================================
    # ✨ App.py 다이어트 완료! (여기서부터 바뀌었습니다)
    # 데이터를 읽고 번역(from_texts)하는 무거운 과정을 싹 지우고,
    # 이미 만들어진 폴더를 '연결'만 해줍니다.
    # ========================================================
    
    # 2. 이미 만들어진 챗봇용 Chroma DB 불러오기
    chat_db = Chroma(
        persist_directory="./chroma_chat_db",
        embedding_function=embeddings
    )

    # 3. 이미 만들어진 감정 분석용 Chroma DB 불러오기
    emotion_db = Chroma(
        persist_directory="./chroma_emotion_db",
        embedding_function=embeddings
    )
    
    return dnn_model, scaler, X_train.columns, cause_df, background_data, chat_db, emotion_db

# 로딩 바 표시 (Chroma 변환 시 시간이 조금 걸립니다)
with st.spinner("🚀 AI 모델 및 벡터 데이터베이스(Chroma) 초기화 중..."):
    model, scaler, feature_cols, cause_df, background_data, chat_db, emotion_db = load_resources()

sleep_ml_map = {'5~6시간':0, '7~8시간':1, '5시간 미만':2, '8시간 이상':3}
diet_ml_map = {'건강함':0, '보통':1, '불건강함':2}
sleep_radar_map = {'5시간 미만': 5, '5~6시간': 4, '7~8시간': 2, '8시간 이상': 1}
diet_radar_map = {'불건강함': 5, '보통': 3, '건강함': 1}

# --------------------------------------------------------
# 4. 사내 시스템 로그인
# --------------------------------------------------------
if not st.session_state['user_name']:
    st.markdown("<h1 class='main-title'>🏢 Deep-Mind: 임직원 정신건강 관리 시스템</h1>", unsafe_allow_html=True)
    name_input = st.text_input("성함을 입력해주세요 (직원용)", placeholder="예: 홍길동")
    if st.button("진단 시스템 접속"):
        if name_input and re.match(r'^[가-힣a-zA-Z\s]+$', name_input):
            st.session_state['user_name'] = name_input; st.rerun()
        else: st.error("🚨 올바른 이름을 입력해주세요.")
    st.stop()

# --------------------------------------------------------
# 5. 사이드바 메뉴 및 히스토리 조회
# --------------------------------------------------------
with st.sidebar:
    st.title("📂 EAP 통합 메뉴")
    app_mode = st.radio("이동할 페이지", [
        "🔍 임직원 정신건강 자가검진", 
        "📝 오늘의 감정 일기", 
        "💬 AI 심리 상담소", 
        "📊 날짜별 우울증 변화"
    ])
    st.divider()
    st.write(f"👤 접속자: **{st.session_state['user_name']}**")
    if st.button("로그아웃"): st.session_state['user_name'] = ""; st.rerun()
    
    if app_mode == "🔍 임직원 정신건강 자가검진":
        st.header("📅 개인별 진단 히스토리")
        selected_date = st.date_input("조회 날짜 선택", datetime.date.today())
        if supabase:
            start_dt = f"{selected_date}T00:00:00"; end_dt = f"{selected_date}T23:59:59"
            res = supabase.table("user_logs").select("*").eq("user_name", st.session_state['user_name']).filter("created_at", "gte", start_dt).filter("created_at", "lte", end_dt).order("created_at", desc=True).execute()
            if res.data:
                time_options = {pd.to_datetime(r['created_at']).tz_convert('Asia/Seoul').strftime('%H시 %M분'): r for r in res.data}
                selected_time = st.selectbox("진단 기록 선택", list(time_options.keys()))
                if st.button("🔍 상세 리포트 불러오기"):
                    st.session_state['view_record'] = time_options[selected_time]
                    st.session_state['page_mode'] = 'history'
                    st.rerun()

# ========================================================
# ✨ [순수 LLM 방식] 프롬프트 최적화 리포트 생성 함수
# ========================================================
def generate_eap_report(age, gender, status, top_reason, sleep, diet, work_p):
    prompt = f"""
    당신은 기업 EAP의 15년 차 수석 정신건강의학과 전문의이자 헬스트레이너, 영양사입니다.
    현재 내담자의 정보는 다음과 같습니다:
    - 나이/성별: {age}세 {gender}
    - AI 진단 위험도: {status}
    - 가장 큰 스트레스 원인: {top_reason} (업무 압박감 수준: {work_p}점)
    - 수면 상태: {sleep} / 식습관: {diet}

    위 정보를 바탕으로 따뜻하고 전문적인 처방을 작성해주세요.
    
    [중요 규칙 - 반드시 지킬 것]
    1. 서론 인사말과 결론 맺음말은 절대 쓰지 마세요. 시간 단축을 위해 바로 본론만 말하세요.
    2. 위험한 의학적 처방(약물 권유 등)을 지어내지 말고, 보편적이고 안전한 생활 습관 위주로 조언하세요.
    3. 각 항목당 반드시 2~3줄 이내의 짧은 개조식(- 기호 사용)으로만 핵심만 답변하세요.
    
    ### 💡 전문의 종합 소견
    (작성)
    ### 🏃 맞춤 운동 처방
    (작성)
    ### 🥗 수면 및 식단 생활 지침
    (작성)
    """
    try:
        response = llm_model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"로컬 LLM 호출 중 에러가 발생했습니다: {e}"

# --------------------------------------------------------
# 6. [페이지 1] 자가검진 및 리포트 (기존 기능 100% 유지)
# --------------------------------------------------------
if app_mode == "🔍 임직원 정신건강 자가검진":
    if st.session_state['page_mode'] == 'history' and st.session_state['view_record']:
        record = st.session_state['view_record']
        st.markdown(f"## 📊 {selected_date} 정밀 분석 리포트")
        st.info("💡 과거 기록의 상세 차트가 표시됩니다.")
        if st.button("⬅️ 진단 화면으로 돌아가기"): 
            st.session_state['page_mode'] = 'test'; st.session_state['view_record'] = None; st.rerun()

    else:
        st.subheader("📝 실시간 직무 스트레스 자가검진")
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("성별", ["남성", "여성"]); age = st.number_input("나이", 18, 65, 30)
            work_p = st.selectbox("업무 압박감 (1-5)", [1,2,3,4,5], index=2); job_s = st.selectbox("직업 만족도 (1-5)", [1,2,3,4,5], index=2)
            sleep = st.selectbox("전일 수면 시간", ['5시간 미만', '5~6시간', '7~8시간', '8시간 이상'])
        with c2:
            diet = st.selectbox("식습관 상태", ['건강함', '보통', '불건강함']); suicide = st.selectbox("자해 충동 여부", ["아니오", "예"])
            work_h = st.number_input("평균 근무 시간", 0, 24, 8); finance = st.selectbox("재정 스트레스 (1-5)", [1,2,3,4,5], index=2)
            family = st.selectbox("정신건강 가족력", ["없음", "있음"])

        if st.button("AI 정밀 분석 실행", type="primary"):
            raw_data = [1 if gender=="남성" else 0, age, work_p, job_s, sleep_ml_map[sleep], diet_ml_map[diet], 1 if suicide=="예" else 0, work_h, finance, 1 if family=="있음" else 0]
            scaled_data = scaler.transform(pd.DataFrame([raw_data], columns=feature_cols))
            prob = float(model.predict(scaled_data)[0][0]); status = "심각한 우울 (위험)" if prob >= 0.7 else ("가벼운 우울 (주의)" if prob >= 0.4 else "정상")
            
            explainer = shap.DeepExplainer(model, background_data)
            user_shap = np.array(explainer.shap_values(scaled_data)[0]).flatten()
            max_idx = [2,3,4,5,6,7,8][np.argmax([user_shap[i] for i in [2,3,4,5,6,7,8]])]
            top_reason = feature_cols[max_idx]; cause_row = cause_df[cause_df['위험_요인'] == top_reason].iloc[0]
            
            cg1, cg2 = st.columns(2)
            with cg1:
                fg = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text':"위험도(%)"}, gauge={'axis':{'range':[0,100]}, 'steps':[{'range':[0,40],'color':"#A7F3D0"},{'range':[70,100],'color':"#FECACA"}]}))
                st.plotly_chart(fg, use_container_width=True)
            with cg2:
                v_r = [work_p, 6-job_s, finance, sleep_radar_map[sleep], diet_radar_map[diet]]
                fr = go.Figure(data=go.Scatterpolar(r=v_r+[v_r[0]], theta=['업무압박','직무만족','재정스트레스','수면부족','식단불량','업무압박'], fill='toself'))
                st.plotly_chart(fr, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🤖 로컬 LLM 맞춤형 EAP 통합 처방전")
            with st.spinner("Gemma2 모델이 맞춤형 솔루션을 생성 중입니다..."):
                llm_report = generate_eap_report(age, gender, status, top_reason, sleep, diet, work_p)
                st.write(llm_report)

            if supabase:
                supabase.table("user_logs").insert({
                    "user_name": st.session_state['user_name'], "score": prob, "severity": status,
                    "work_pressure": work_p, "job_satisfaction": job_s, "sleep_time": sleep, "diet_status": diet, "financial_stress": finance
                }).execute()

# --------------------------------------------------------
# 7. [페이지 2] 오늘의 감정 일기
# --------------------------------------------------------
elif app_mode == "📝 오늘의 감정 일기":
    st.markdown("<h2 class='main-title'>📖 로컬 LLM 감정 일기 및 분석</h2>", unsafe_allow_html=True)
    
    # ✅ 수정 1: 버튼 먹통 해결! '콜백(Callback)' 함수로 강제 삭제 명령을 내립니다.
    def delete_diary_log(log_id):
        if supabase:
            supabase.table("diary_logs").delete().eq("id", log_id).execute()

    # ✅ 수정 2: 창을 줄여도 삭제 버튼이 안 찌그러지게 '최소 너비(min-width)'를 강제 방어합니다.
    st.markdown("""
        <style>
        div[data-testid="column"]:nth-child(2) .stButton > button {
            min-width: 80px !important; 
            padding: 0.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    diary_tab1, diary_tab2 = st.tabs(["✍️ 오늘의 감정 일기 쓰기 ☁️", "📅 나의 일기장 & 감정 통계 📈"])

    with diary_tab1:
        with st.expander("💡 감정 점수(1~10점) 가이드라인 보기", expanded=False):
            st.markdown("* 🔴 1-3점: 심각 / 🟡 4-5점: 지침 / 🟢 6-7점: 보통 / 🔵 8-10점: 행복")

        if "diary_input" not in st.session_state:
            st.session_state.diary_input = ""

        user_diary = st.text_area(
            "오늘 하루, 어떤 감정들이 스쳐 지나갔나요? 사소한 것도 좋으니 편하게 털어놓아 주세요. 🍵", 
            height=150, 
            key="diary_input"
        )
        
        if st.button("✨ 내 감정 분석 및 일기장 저장하기 ✨", type="primary"):
            if user_diary:
                with st.spinner("Chroma DB를 검색하고 Gemma2가 분석 중입니다..."):
                    docs = emotion_db.similarity_search(user_diary, k=1)
                    emotion_label = docs[0].metadata['label'] if docs else 1
                    emotion_txt = "긍정/안정" if emotion_label == 1 else "부정/우울/스트레스"
                    
                    prompt = f"""
                    당신은 15년차 정신건강의학과 전문의이자 심리 분석가입니다.
                    말이 잘 통하는 따뜻한 상담가이기도 합니다.
                    emotion_large_dataset 데이터를 참고만 하고, 오늘 사용자가 쓴 일기 텍스트 '{user_diary}'의 감정을 분석해 주세요.
                    과거 분석 결과, 이 일기는 '{emotion_txt}' 감정에 가깝다고 판단되었습니다.
                    결과를 바탕으로 사용자가 이해하기 쉽게 다음 3가지 형식을 정확히 지켜서 답변해주세요.
                    
                    [위로와 분석]: (공감하는 따뜻한 위로 4문장)
                    [감정 이모지]: (오늘의 대표 이모지 1개)
                    [감정 점수]: (1~10 사이의 숫자만 입력)
                    """
                    response = llm_model.invoke(prompt)
                    res_text = response.content
                    
                    score_match = re.search(r'감정\s*점수.*?(\d+)', res_text)
                    if score_match:
                        score_val = max(1, min(10, int(score_match.group(1)))) 
                    else:
                        score_val = 5 
                    
                    st.divider()
                    st.markdown("### 🤖 Gemma2 감정 분석 결과")
                    st.write(res_text)

                    if supabase:
                        try:
                            supabase.table("diary_logs").insert({
                                "user_name": st.session_state['user_name'],
                                "diary_text": user_diary, 
                                "ai_comment": res_text, 
                                "emotion_score": score_val
                            }).execute()
                            
                            st.toast("✅ 감정일기장에 성공적으로 저장되었습니다!")
                            time.sleep(1.5)
                            st.session_state.diary_input = "" 
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f" 저장 중 에러 발생: {e}")
            else:
                st.warning("일기 내용을 입력해 주세요!")

    with diary_tab2:
        st.subheader("📅 과거 일기 조회 및 감정 통계")
        if supabase:
            res = supabase.table("diary_logs").select("*").eq("user_name", st.session_state['user_name']).order("created_at", desc=False).execute()
            if res.data:
                df_diary = pd.DataFrame(res.data)
                
                # ✅ 수정 3: 영국 시간(UTC)을 명시하고, 한국 시간(Asia/Seoul)으로 완벽하게 +9시간 변환합니다.
                df_diary['created_at'] = pd.to_datetime(df_diary['created_at'], utc=True).dt.tz_convert('Asia/Seoul')
                df_diary['date_only'] = df_diary['created_at'].dt.date

                fig_diary = px.line(df_diary, x='created_at', y='emotion_score', markers=True, color_discrete_sequence=['#2563EB'])
                fig_diary.update_layout(yaxis=dict(range=[0, 11]))
                st.plotly_chart(fig_diary, use_container_width=True)

                st.divider()
                selected_date = st.date_input("조회할 날짜를 선택하세요", value=df_diary['date_only'].iloc[-1], min_value=df_diary['date_only'].min(), max_value=df_diary['date_only'].max())
                day_data = df_diary[df_diary['date_only'] == selected_date]

                if not day_data.empty:
                    for idx, row in day_data.iterrows():
                        # 버튼이 숨을 쉴 수 있도록 컬럼 비율을 [0.8, 0.2]로 넉넉하게 주었습니다.
                        col_text, col_btn = st.columns([0.8, 0.2])
                        with col_text:
                            st.markdown(f"**🕒 {row['created_at'].strftime('%H:%M')} | ⭐ {row['emotion_score']}점**")
                        with col_btn:
                            # ✅ 수정 1 반영: 버튼을 누르는 즉시(on_click) 삭제 함수가 100% 실행되도록 보장합니다!
                            st.button("🗑️ 삭제", key=f"del_{row['id']}", on_click=delete_diary_log, args=(row['id'],))
                                
                        st.info(f"**📝 나의 일기:**\n{row['diary_text']}")
                        st.success(f"**🤖 AI 분석:**\n{row['ai_comment']}")
                        st.markdown("---")
            else: st.info("아직 저장된 일기가 없습니다.")

# --------------------------------------------------------
# 8. [페이지 3] AI 심리 상담소 (✨ Ollama + Chroma RAG)
# --------------------------------------------------------
elif app_mode == "💬 AI 심리 상담소":
    st.markdown("<h2 class='main-title'>🩺 로컬 AI 심리 상담사</h2>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"안녕하세요 {st.session_state['user_name']}님. 외부로 대화가 유출되지 않는 안전한 로컬 상담소입니다. 편하게 말씀해 주세요."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    user_chat = st.chat_input("고민을 입력하세요...")
    if user_chat:
        st.session_state.messages.append({"role": "user", "content": user_chat})
        with st.chat_message("user"): st.markdown(user_chat)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            docs = chat_db.similarity_search(user_chat, k=1)
            best_answer = docs[0].metadata.get('answer', "위로와 공감을 전해주세요.") if docs else "위로와 공감을 전해주세요."
            
            prompt = f"""
            당신은 기업 EAP 소속의 15년 차 수석 정신건강의학과 전문의입니다.
            사용자의 말: "{user_chat}"
            우수 상담 데이터 가이드: "{best_answer}"
            
            위 가이드를 참고하되, 단순히 동조하고 위로만 하는 것을 넘어 전문의로서의 신뢰감 있는 분석과 실질적인 심리 처방(행동 지침)을 제안해주세요. 
            말투는 따뜻하면서도 지적이고 차분한 '의사 선생님'의 톤(예: "~하셨군요.", "~해 보시는 것을 권해드립니다.")을 완벽하게 유지해주세요.
            """
            response = llm_model.invoke(prompt)
            full_response = response.content
            
            displayed_text = ""
            for chunk in full_response.split(" "):
                displayed_text += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(displayed_text + "▌")
            message_placeholder.markdown(displayed_text)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --------------------------------------------------------
# 9. [페이지 4] 히스토리 차트
# --------------------------------------------------------
elif app_mode == "📊 날짜별 우울증 변화":
    st.markdown("<h2 class='main-title'>📈 날짜별 우울증 변화 그래프</h2>", unsafe_allow_html=True)
    if supabase:
        res = supabase.table("user_logs").select("*").eq("user_name", st.session_state['user_name']).order("created_at", desc=False).execute()
        if res.data:
            df_l = pd.DataFrame(res.data)
            df_l['created_at'] = pd.to_datetime(df_l['created_at']).dt.tz_convert('Asia/Seoul')
            df_l['score_pct'] = df_l['score'] * 100
            fig_t = px.line(df_l, x='created_at', y='score_pct', markers=True, title="기간별 우울 지수 변동 추이", labels={'created_at': '진단 시간', 'score_pct': '위험도 (%)'})
            st.plotly_chart(fig_t, use_container_width=True)
        else: 
            # ✅ 수정 3: 페이지가 안 뜬다면 데이터가 비어있을 확률이 큽니다. 명확하게 알려주도록 수정!
            st.info("아직 저장된 자가검진 기록이 없습니다. '임직원 정신건강 자가검진' 탭에서 진단을 먼저 진행해 주세요!")
    else:
        st.error("데이터베이스 연결에 문제가 있습니다.")