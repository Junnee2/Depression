import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

print("🚀 데이터 임베딩 공장 가동 시작! (청크 단위 안정적 처리)")

# 1. 임베딩 엔진 준비
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 💡 한 번에 소화할 덩어리(청크) 크기 설정
CHUNK_SIZE = 100 

# ========================================================
# [1] 챗봇 데이터 청크 처리
# ========================================================
print("\n💬 [1/2] 챗봇 데이터 읽는 중...")
# 원하시는 전체 개수를 설정하세요. (예: 2000개)
df_chat = pd.read_csv("chatbot_large_dataset.csv").dropna().head(100000) 

chat_texts = df_chat['Q'].tolist()
chat_metas = [{"answer": a} for a in df_chat['A'].tolist()]

chat_db = None
print(f"총 {len(chat_texts)}개의 챗봇 데이터를 {CHUNK_SIZE}개씩 나누어 저장합니다.")

for i in range(0, len(chat_texts), CHUNK_SIZE):
    batch_texts = chat_texts[i : i + CHUNK_SIZE]
    batch_metas = chat_metas[i : i + CHUNK_SIZE]
    
    if chat_db is None:
        # 첫 번째 덩어리: DB 폴더를 새로 만듭니다.
        chat_db = Chroma.from_texts(
            texts=batch_texts, 
            embedding=embeddings, 
            metadatas=batch_metas, 
            persist_directory="./chroma_chat_db"
        )
    else:
        # 두 번째 덩어리부터: 만들어진 DB에 '추가(add_texts)' 합니다.
        chat_db.add_texts(texts=batch_texts, metadatas=batch_metas)
        
    print(f"   ▶ 진행률: {min(i + CHUNK_SIZE, len(chat_texts))} / {len(chat_texts)} 완료")

print("✅ 챗봇 데이터 DB 굽기 완료!")

# ========================================================
# [2] 감정 일기 데이터 청크 처리
# ========================================================
print("\n📖 [2/2] 감정 일기 데이터 읽는 중...")
# 원하시는 전체 개수를 설정하세요. (예: 2000개)
df_emotion = pd.read_csv("emotion_large_dataset.csv").dropna().head(2000)

emotion_texts = df_emotion['일기_텍스트'].tolist()
emotion_metas = [{"label": l} for l in df_emotion['감정_라벨'].tolist()]

emotion_db = None
print(f"총 {len(emotion_texts)}개의 일기 데이터를 {CHUNK_SIZE}개씩 나누어 저장합니다.")

for i in range(0, len(emotion_texts), CHUNK_SIZE):
    batch_texts = emotion_texts[i : i + CHUNK_SIZE]
    batch_metas = emotion_metas[i : i + CHUNK_SIZE]
    
    if emotion_db is None:
        emotion_db = Chroma.from_texts(
            texts=batch_texts, 
            embedding=embeddings, 
            metadatas=batch_metas, 
            persist_directory="./chroma_emotion_db"
        )
    else:
        emotion_db.add_texts(texts=batch_texts, metadatas=batch_metas)
        
    print(f"   ▶ 진행률: {min(i + CHUNK_SIZE, len(emotion_texts))} / {len(emotion_texts)} 완료")

print("✅ 일기 데이터 DB 굽기 완료!")

print("\n🎉 모든 데이터베이스 구축이 안전하게 완료되었습니다!")