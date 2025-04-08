import concurrent.futures
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
import random
from tqdm import tqdm
tqdm.pandas()

def load_ollama(model="gemma3:4b", temperature=0.0, top_p=0.9, num_predict=256):
    llm = ChatOllama(model=model, temperature=temperature, num_predict=num_predict, top_p=top_p)
    prompt = ChatPromptTemplate.from_template("{message}")
    chat_model = prompt | llm | StrOutputParser()
    return chat_model

def run_ollama(message, chat_model):
    answer = chat_model.invoke({"message": message})
    return answer

def prompt_test(df, col, chat_model, idx=-1):
    if idx == -1:
        idx = random.randint(0, len(df) - 1)
    print(f"idx: {idx}")
    
    prompt = df.loc[idx, col]
    print(prompt)
    answer = run_ollama(prompt, chat_model)
    print(answer)

# 함수 정의: 부분 데이터프레임에 대해 체인을 적용합니다.
def process_sub_df_with_progress_bar(sub_df, prompt_col, chain):
    result = sub_df[prompt_col].progress_apply(lambda prompt: run_ollama(prompt, chain))
    return result

def process_sub_df(sub_df, prompt_col, chain):
    result = sub_df[prompt_col].apply(lambda prompt: run_ollama(prompt, chain))
    return result

def parallel_process_n(df, prompt_col, output_col, chain1, chain2):
    # 데이터프레임을 4개로 분할합니다.
    quarter = len(df) // 4
    df1 = df.iloc[:quarter].copy()
    df2 = df.iloc[quarter:quarter*2].copy()
    df3 = df.iloc[quarter*2:quarter*3].copy()
    df4 = df.iloc[quarter*3:].copy()

    # 동시에 처리하도록 ThreadPoolExecutor(또는 상황에 따라 ProcessPoolExecutor)를 사용합니다.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(process_sub_df, df1, prompt_col, chain1)
        future2 = executor.submit(process_sub_df, df2, prompt_col, chain2)
        future3 = executor.submit(process_sub_df, df3, prompt_col, chain1)
        future4 = executor.submit(process_sub_df, df4, prompt_col, chain2)

        result1 = future1.result()
        result2 = future2.result()
        result3 = future3.result()
        result4 = future4.result()

    # 결과를 원본 데이터프레임에 다시 할당합니다.
    df.loc[df1.index, output_col] = result1
    df.loc[df2.index, output_col] = result2
    df.loc[df3.index, output_col] = result3
    df.loc[df4.index, output_col] = result4

def process_dataframe_in_chunks(df, prompt_col, output_cols, chain1, chain2, chunk_size=30):
    # output_col이 str인 경우 리스트로 변환합니다.
    if isinstance(output_cols, str):
        output_cols = [output_cols]

    # output_cols에 포함된 컬럼이 없으면 새로 생성합니다.
    for col in output_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 모든 output_cols가 결측치인 행만 선택합니다.
    missing_mask = df[output_cols].isna().any(axis=1)
    missing_df = df[missing_mask]
    
    print(f"처리할 행의 개수: {len(missing_df)}, 출력 컬럼 개수: {len(output_cols)}, 한번에 연산되는 양: {chunk_size}개")
    
    # 결측치 행들을 청크 단위로 처리합니다.
    for start in tqdm(range(0, len(missing_df), chunk_size)):
        # 현재 청크에 해당하는 인덱스 추출
        chunk_indices = missing_df.index[start:start + chunk_size]
        # 원본 df에서 해당 인덱스의 행을 선택 (원본과 연결된 view)
        chunk = df.loc[chunk_indices]
        
        # 해당 청크를 처리할 때 KeyboardInterrupt가 아닌 다른 에러가 발생하면 재시도
        while True:
            try:
                # parallel_process_n은 output_col을 그대로 사용합니다.
                for output_col in output_cols:
                    parallel_process_n(chunk, prompt_col, output_col, chain1, chain2)
                # 처리된 청크의 결과를 원본 df에 반영합니다.
                df.loc[chunk_indices, output_cols] = chunk[output_cols]
                break  # 성공하면 반복문 탈출
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"에러 발생: {e}. 해당 청크를 다시 실행합니다.")
    
    return df

if __name__ == '__main__':
    chat_model = load_ollama()
    print(run_ollama("안녕하세요", chat_model))
