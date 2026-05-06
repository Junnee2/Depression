import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

print("🏃‍♂️ 데이터 임베딩 이어달리기 가동! (10000번 ~ 13000번)")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
CHUNK_SIZE = 100 

# ========================================================
# [2] 감정 일기 데이터 이어달리기 (5000 ~ 10000)
# ========================================================
print("\n📖 [2/2] 감정 일기 데이터 10000번부터 읽는 중...")
# ✨ 핵심: 앞에서 5000개를 건너뛰고, 다음 5000개를 가져옵니다.
df_emotion = pd.read_csv("emotion_large_dataset.csv").dropna().iloc[5000:10000]

emotion_texts = df_emotion['일기_텍스트'].tolist()
emotion_metas = [{"label": l} for l in df_emotion['감정_라벨'].tolist()]

emotion_db = Chroma(persist_directory="./chroma_emotion_db", embedding_function=embeddings)

for i in range(0, len(emotion_texts), CHUNK_SIZE):
    batch_texts = emotion_texts[i : i + CHUNK_SIZE]
    batch_metas = emotion_metas[i : i + CHUNK_SIZE]
    emotion_db.add_texts(texts=batch_texts, metadatas=batch_metas)
    print(f"   ▶ 일기 추가 진행률: {min(i + CHUNK_SIZE, len(emotion_texts))} / {len(emotion_texts)} 완료")

print("\n🎉 이어달리기 구축 완료!")
print(f"📊 최종 일기 DB 총 데이터 수: {emotion_db._collection.count()}개")