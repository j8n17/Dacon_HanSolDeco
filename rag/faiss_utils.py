import os
import logging
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 로깅 설정 (필요에 따라 파일 출력이나 포맷 변경 가능)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_vector_store(
        parent_dir="./", 
        less_general=False,
        source_col='reason',
        faiss_name="faiss_index",
        train_csv_path="../data/train.csv",
        cos_sim_matrix_path="../data/cos_sim_matrix.npy",
        embedding_model_name="upskyy/bge-m3-korean",
        logging_mode=True
    ):
    """
    FAISS 인덱스가 저장된 디렉토리가 있으면 로드하고,
    없으면 train_csv_path에서 데이터를 읽어 새 인덱스를 생성 후 저장합니다.
        
    Returns:
        vector_store: FAISS 벡터스토어 인스턴스.
    """
    if less_general:
        faiss_name = "less_general_" + source_col + "_" + faiss_name
    else:
        faiss_name = 'general_' + source_col + "_" + faiss_name
    logging.info("faiss loading 시작: %s", faiss_name)
    faiss_path = os.path.join(parent_dir, faiss_name)
    if not logging_mode:
        logging.disable(logging.INFO)

    if os.path.exists(faiss_path):
        logging.info("저장된 FAISS 인덱스를 불러옵니다.")
        # 인덱스를 불러오기 위해 임베딩 모델 객체가 필요합니다.
        embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_store = FAISS.load_local(faiss_path, embedding, allow_dangerous_deserialization=True)
    else:
        logging.info("새로운 FAISS 인덱스를 생성합니다.")
        # CSV 데이터를 불러오고 질문과 답변(메타데이터) 추출
        train_df = pd.read_csv(train_csv_path, encoding='utf-8-sig')
        cos_sim_matrix = np.load(cos_sim_matrix_path)[11380, train_df.id.tolist()]  # 현장 관리 철저와 작업자 안전교육 실시를 통한 재발 방지 대책 및 향후 조치 계획.

        ground_truths = train_df['ground_truth'].tolist()
        correct_descriptions = train_df['correct_description'].tolist()
        ids = train_df['id'].tolist()

        if less_general:
            sources = []
            metadatas = []
            for i, gt in enumerate(ground_truths):
                score = cos_sim_matrix[i]
                if 0.55 <= score <= 0.8:
                    sources.append(train_df.loc[i, source_col])
                    metadatas.append({"gt": gt, 'correct_description': correct_descriptions[i], 'id': ids[i]})
        else:
            sources = train_df[source_col].tolist()
            metadatas = [{"gt": gt, 'correct_description': correct_descriptions[i], 'id': ids[i]} for i, gt in enumerate(ground_truths)]

        # 임베딩 모델 생성
        embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
        # FAISS 인덱스 생성: 질문 텍스트와 메타데이터를 함께 인덱싱합니다.
        vector_store = FAISS.from_texts(sources, embedding, metadatas=metadatas)
        # 생성된 인덱스를 저장합니다.
        vector_store.save_local(faiss_path)
        logging.info("인덱스가 저장되었습니다: %s", faiss_path)
    logging.info('vector_store 로드 완료')
    return vector_store

def search_query(query, vector_store, top_k=5, logging_mode=True):
    if not logging_mode:
        logging.disable(logging.INFO)
        
    logging.info("search_query 함수 실행")
    # 초기 FAISS 검색 (최대 initial_k개 검색)
    rag_results = vector_store.similarity_search_with_score(query, k=top_k)
    
    for i, (source, rag_score) in enumerate(rag_results):
        logging.info(f"검색 결과 {i+1}: {source.page_content}, 유사도: {rag_score}")
    
    return rag_results

def rerank_search(query, vector_store, reranker, top_k=25, logging_mode=True):
    rag_results = search_query(query, vector_store, top_k=top_k)
    rag_sources = [source.page_content for source, rag_score in rag_results]
    rag_gts = [source.metadata.get('gt') for source, rag_score in rag_results]
    rag_ids = [source.metadata.get('id') for source, rag_score in rag_results]

    pairs = [(query, source) for source in rag_sources]
    # HuggingFace CrossEncoder로 유사도 점수 계산
    rerank_scores = reranker.predict(pairs)
    
    # 각 결과와 점수를 함께 묶은 후 점수가 높은 순으로 정렬하여 상위 final_top_k개 선택 후 순서 바꿔서 저장
    ranked_rag_results = sorted(zip(rag_results, rerank_scores), key=lambda x: x[1], reverse=True)

    rerank_queries = [source.page_content for (source, rag_score), rerank_score in ranked_rag_results][:top_k][::-1]
    rerank_gts = [source.metadata.get('gt') for (source, rag_score), rerank_score in ranked_rag_results][:top_k][::-1]
    rerank_scores = [rerank_score for (source, rag_score), rerank_score in ranked_rag_results][:top_k][::-1]
    
    logging.info(f"Query: {query}")
    logging.info("Reranking 후 최종 결과 (상위 %d개):", top_k)
    for i, gt in enumerate(rerank_gts):
        logging.info(f"Reranked Result {i+1}: source: %s, grount_truth: %s, 유사도: %s", rerank_queries[i], gt, rerank_scores[i])
    
    return rag_ids, rerank_queries, rerank_gts, rerank_scores

def rerank_search_with_metadata(query, metadata_name, vector_store, reranker, top_k=5, rag_scale=5, logging_mode=True):
    rag_results = search_query(query, vector_store, top_k=rag_scale*top_k)
    rag_sources = [source.page_content for source, rag_score in rag_results]
    rag_gts = [source.metadata.get('gt') for source, rag_score in rag_results]
    rag_ids = [source.metadata.get('id') for source, rag_score in rag_results]

    if metadata_name == 'gt':
        gt_query = '작업전 안전교육 강화 및 작업장 위험요소 점검을 통한 재발 방지와 안전관리 교육 철저를 통한 향후 조치 계획.'
        pairs = [(gt_query, source.metadata.get(metadata_name)) for source, rag_score in rag_results]
        top_k = rag_scale * top_k
    else:
        pairs = [(query, source.metadata.get(metadata_name)) for source, rag_score in rag_results]
    # HuggingFace CrossEncoder로 유사도 점수 계산
    rerank_scores = reranker.predict(pairs)
    
    # 각 결과와 점수를 함께 묶은 후 점수가 높은 순으로 정렬하여 상위 final_top_k개 선택 후 순서 바꿔서 저장
    ranked_rag_results = sorted(zip(rag_results, rerank_scores), key=lambda x: x[1], reverse=True)
    
    rerank_sources = [source for (source, rag_score), rerank_score in ranked_rag_results][:top_k][::-1]
    rerank_queries = [source.page_content for (source, rag_score), rerank_score in ranked_rag_results][:top_k][::-1]
    rerank_gts = [source.metadata.get('gt') for (source, rag_score), rerank_score in ranked_rag_results][:top_k][::-1]
    rerank_scores = [rerank_score for (source, rag_score), rerank_score in ranked_rag_results][:top_k][::-1]
    
    logging.info(f"Query: {query}")
    logging.info("Reranking 후 최종 결과 (상위 %d개):", top_k)
    for i, source in enumerate(rerank_sources):
        logging.info(f"Reranked Result {i+1}: 질문: {source.page_content}, {metadata_name}: {source.metadata.get(metadata_name)}, grount_truth: {source.metadata.get('gt')}, 유사도: {rerank_scores[i]}")
    
    return rag_ids, rerank_queries, rerank_gts, rerank_scores
