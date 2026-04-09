import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from supabase import create_client
import datetime
import plotly.graph_objects as go
import plotly.express as px
import shap
import re
import os

# 텐서플로우 로그 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
# 3. 리소스 로드 및 공통 맵핑 변수 설정
# --------------------------------------------------------
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('depression_model.h5')
    df = pd.read_csv('Depression.csv')
    df.columns = ['성별', '나이', '업무_압박감', '직업_만족도', '수면_시간', '식습관', '자해_충동_여부', '근무_시간', '재정적_스트레스', '정신질환_가족력', '우울증']
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns: df[col] = le.fit_transform(df[col])
    X = df.drop('우울증', axis=1)
    y = df['우울증']
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    recom_df = pd.read_csv('Recommendations.csv')
    cause_df = pd.read_csv('Cause_Solutions.csv')
    exer_df = pd.read_csv('Exercise_Solutions.csv')
    background_data = scaler.transform(X_train)[:100]
    
    return model, scaler, X_train.columns, recom_df, cause_df, exer_df, background_data

model, scaler, feature_cols, recom_df, cause_df, exer_df, background_data = load_resources()

# ML 모델용 데이터 변환 맵핑
sleep_ml_map = {'5~6시간':0, '7~8시간':1, '5시간 미만':2, '8시간 이상':3}
diet_ml_map = {'건강함':0, '보통':1, '불건강함':2}

# 다각형 차트 수치화 맵핑
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
            st.session_state['user_name'] = name_input
            st.rerun()
        else: st.error("🚨 올바른 이름을 입력해주세요.")
    st.stop()

# --------------------------------------------------------
# 5. 사이드바 메뉴 및 히스토리 조회
# --------------------------------------------------------
with st.sidebar:
    st.title("📂 검진 메뉴")
    app_mode = st.radio("이동할 페이지", ["🔍 임직원 정신건강 자가검진", "📊 날짜별 우울증 변화"])
    st.divider()
    st.write(f"👤 접속자: **{st.session_state['user_name']}**")
    if st.button("로그아웃"):
        st.session_state['user_name'] = ""
        st.rerun()
    
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

# --------------------------------------------------------
# 6. [페이지 1] 자가검진 및 리포트
# --------------------------------------------------------
if app_mode == "🔍 임직원 정신건강 자가검진":
    if st.session_state['page_mode'] == 'history' and st.session_state['view_record']:
        record = st.session_state['view_record']
        st.markdown(f"## 📊 {selected_date} 정밀 분석 리포트")
        score_pct = record['score'] * 100
        status = record['severity']
        
        # 차트 레이아웃
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=score_pct, title={'text':"위험도 지수 (%)"},
                gauge={'axis':{'range':[0,100]}, 'steps':[{'range':[0,40],'color':"#A7F3D0"},{'range':[40,70],'color':"#FDE047"},{'range':[70,100],'color':"#FECACA"}]}))
            st.plotly_chart(fig_g, use_container_width=True)
        with ch2:
            v_radar = [
                record.get('work_pressure', 3), 
                6 - record.get('job_satisfaction', 3), 
                record.get('financial_stress', 3), 
                sleep_radar_map.get(record.get('sleep_time', '7~8시간'), 3), 
                diet_radar_map.get(record.get('diet_status', '보통'), 3)
            ]
            fig_r = go.Figure(data=go.Scatterpolar(r=v_radar+[v_radar[0]], theta=['업무압박','직무만족','재정스트레스','수면부족','식단불량','업무압박'], fill='toself'))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])), title="당시 스트레스 분포")
            st.plotly_chart(fig_r, use_container_width=True)

        # ★ 1. 과거 기록용 원인 분석(SHAP) 재계산 ★
        raw_hist = [1, 30, record.get('work_pressure', 3), record.get('job_satisfaction', 3), 
                    sleep_ml_map.get(record.get('sleep_time', '7~8시간'), 1), diet_ml_map.get(record.get('diet_status', '보통'), 1), 
                    0, 8, record.get('financial_stress', 3), 0]
        scaled_hist = scaler.transform(pd.DataFrame([raw_hist], columns=feature_cols))
        explainer_hist = shap.DeepExplainer(model, background_data)
        shap_hist = np.array(explainer_hist.shap_values(scaled_hist)[0]).flatten()
        max_idx_h = [2,3,4,5,6,7,8][np.argmax([shap_hist[i] for i in [2,3,4,5,6,7,8]])]
        top_reason_h = feature_cols[max_idx_h]
        cause_row_h = cause_df[cause_df['위험_요인'] == top_reason_h].iloc[0]

        # ★ 2. 상세 리포트 내용 풍성하게 복구 ★
        row = recom_df[recom_df['진단'] == status].iloc[0]
        ex_row = exer_df[exer_df['진단'] == status].iloc[0]
        
        st.markdown("### 📋 맞춤형 EAP 관리 리포트")
        t1, t2, t3 = st.tabs(["💡 진단 및 원인 소견", "🏃 맞춤 운동 처방", "🛠️ 필수 생활 지침"])
        
        with t1: 
            st.info(f"**상태 요약:** {row['상태_설명']}")
            st.error(f"**🤖 주요 스트레스 원인 ({top_reason_h.replace('_', ' ')}):**\n\n{cause_row_h['원인_분석']}\n\n**💡 맞춤 솔루션:**\n\n{cause_row_h['맞춤_솔루션']}")
        with t2: 
            st.success(f"**추천 운동:** {ex_row['운동_종류']} ({ex_row['운동_강도']})\n\n**실행 가이드:** {ex_row['실행_가이드']}\n\n**기대 효과:** {ex_row['기대_효과']}")
        with t3: 
            st.warning(f"**행동 지침:** {row['해결_방안']}")

        # ★ 3. 데이터 삭제 버튼 복구 ★
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ 진단 화면으로 돌아가기", use_container_width=True): 
                st.session_state['page_mode'] = 'test'
                st.rerun()
        with col_btn2:
            if st.button("🗑️ 이 기록 영구 삭제하기", type="secondary", use_container_width=True):
                if supabase and 'id' in record:
                    supabase.table("user_logs").delete().eq("id", record['id']).execute()
                    st.toast("✅ 기록이 성공적으로 삭제되었습니다.")
                    st.session_state['view_record'] = None
                    st.session_state['page_mode'] = 'test'
                    st.rerun()

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

            # ★ 실시간 진단 시에도 상세 리포트 출력 ★
            row = recom_df[recom_df['진단'] == status].iloc[0]
            ex_row = exer_df[exer_df['진단'] == status].iloc[0]
            
            st.markdown("### 📋 맞춤형 EAP 관리 리포트")
            t1, t2, t3 = st.tabs(["💡 진단 및 원인 소견", "🏃 맞춤 운동 처방", "🛠️ 필수 생활 지침"])
            with t1: 
                st.info(f"**상태 요약:** {row['상태_설명']}")
                st.error(f"**🤖 주요 스트레스 원인 ({top_reason.replace('_', ' ')}):**\n\n{cause_row['원인_분석']}\n\n**💡 맞춤 솔루션:**\n\n{cause_row['맞춤_솔루션']}")
            with t2: 
                st.success(f"**추천 운동:** {ex_row['운동_종류']} ({ex_row['운동_강도']})\n\n**실행 가이드:** {ex_row['실행_가이드']}\n\n**기대 효과:** {ex_row['기대_효과']}")
            with t3: 
                st.warning(f"**행동 지침:** {row['해결_방안']}")

            if supabase:
                supabase.table("user_logs").insert({
                    "user_name": st.session_state['user_name'], "score": prob, "severity": status,
                    "work_pressure": work_p, "job_satisfaction": job_s, "sleep_time": sleep, "diet_status": diet, "financial_stress": finance
                }).execute()
                st.toast("✅ 분석 데이터가 저장되었습니다.")

# --------------------------------------------------------
# 7. [페이지 2] 날짜별 우울증 변화
# --------------------------------------------------------
elif app_mode == "📊 날짜별 우울증 변화":
    st.markdown("<h2 class='main-title'>📈 날짜별 우울증 변화 그래프</h2>", unsafe_allow_html=True)
    if supabase:
        res = supabase.table("user_logs").select("*").eq("user_name", st.session_state['user_name']).order("created_at", desc=False).execute()
        if res.data:
            df_l = pd.DataFrame(res.data)
            df_l['created_at'] = pd.to_datetime(df_l['created_at']).dt.tz_convert('Asia/Seoul')
            df_l['score_pct'] = df_l['score'] * 100
            
            st.subheader("🗓️ 기간별 우울 지수 변동 추이")
            fig_t = px.line(df_l, x='created_at', y='score_pct', markers=True, title="기간별 우울 지수 변동 추이", labels={'created_at': '진단 시간', 'score_pct': '위험도 (%)'}, line_shape='spline', color_discrete_sequence=['#4F46E5'])
            fig_t.update_layout(plot_bgcolor='white', yaxis=dict(range=[0, 105]))
            st.plotly_chart(fig_t, use_container_width=True)
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("누적 데이터 수", f"{len(df_l)}건")
            with m2: st.metric("평균 위험 수치", f"{df_l['score_pct'].mean():.1f}%")
            with m3:
                last_val = df_l['score_pct'].iloc[-1]
                delta_val = last_val - df_l['score_pct'].iloc[-2] if len(df_l) > 1 else 0
                st.metric("최근 변동폭", f"{last_val:.1f}%", delta=f"{delta_val:.1f}%", delta_color="inverse")
            
            st.subheader("📋 전체 진단 이력 데이터")
            st.dataframe(df_l[['created_at', 'score_pct', 'severity']].rename(columns={'created_at':'진단 시간', 'score_pct':'위험도(%)', 'severity':'상태'}), use_container_width=True)
        else: st.info("기록이 없습니다.")