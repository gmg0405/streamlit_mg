from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
import pandas as pd
import re


import streamlit as st
import pandas as pd
from kiwipiepy import Kiwi

def main():
    st.set_page_config(layout="wide")  # Wide 모드로 설정

    st.title("어휘 연어 관계 분석(콘코던스)")

    # 파일 업로드
    uploaded_file =st.file_uploader("Excel 파일 업로드", type=["xlsx","csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        # if df is not None:
        #     df = preprocess_data(df)
        show_data(df)

def load_data(uploaded_file):
    kiwi = Kiwi()
    st.write("데이터의 형태소를 분석 중입니다...")
    with st.spinner("로딩 중..."):
        df=pd.read_csv('jinkyeong_sentence.csv',encoding='utf-8')
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                #df['content_pos'] = df['content'].apply(lambda x: kiwi.tokenize(x))
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                #df['content_pos'] = df['content'].apply(lambda x: kiwi.tokenize(x))
            return df
        except Exception as e:
            st.error(f"파일을 로드하는 중 오류가 발생했습니다: {e}")
            return None
        st.success("로딩 완료!")  # 작업 완료 후 메시지 표시
    return df

# def preprocess_data(df):
#     # 형태소 분석기 초기화
#     kiwi = Kiwi()
#     #kiwi.prepare()

#     # 'content' 열을 형태소 분석하여 새로운 열에 저장
#     df['content_pos'] = df['content'].apply(lambda x: tokenize(x, kiwi))
#     print(df)
#     return df

def show_data(df):
    st.write("### 데이터 미리보기")
    st.write(df)
    length=len(df)
    st.write(f'전체 데이터 건수: {length}')


    search_col1, search_col2 = st.columns(2)

    with search_col1:
        # 왼쪽 검색 기능
        #left_search_term_pos = st.text_input("왼쪽 품사 입력 (명사, 동사 등):")
        left_search_term = st.text_input("왼쪽 검색어 입력:")
    
    with search_col2:
        # 오른쪽 검색 기능
        #right_search_term_pos = st.text_input("오른쪽 품사 입력 (명사, 동사 등):")
        right_search_term = st.text_input("오른쪽 검색어 입력:")

    # 화면 분할
    col1, col2 = st.columns(2)

    # 왼쪽 열에 검색 결과 표시
    with col1:
        if left_search_term:
            # 형태소 분석기 초기화
        
            kiwi = Kiwi(model_type='sbg')
            #kiwi.prepare()

            

            # 검색어 형태소 분석
            left_search_tokens = tokenize(left_search_term, kiwi)
            st.write(left_search_tokens)

            # 형태소 분석 결과를 활용하여 검색하기
            search_results_left = df[df['content'].str.contains(left_search_term, case=False)][['content','url','category']]#search_with_pos(df, left_search_term, left_search_term_pos)#
            st.write("검색 건 수:",len(search_results_left))
            st.write(f"### {left_search_term}에 대한 검색 결과")
            #st.write(search_results_left)

            #검색 결과 표에 하이라이트 적용
            search_results_html_left = highlight_search_results(search_results_left, left_search_term)
            st.write(search_results_html_left, unsafe_allow_html=True)

    # 오른쪽 열에 검색 결과 표시
    with col2:
        if right_search_term:
            # 형태소 분석기 초기화
            kiwi = Kiwi(model_type='sbg')
            #kiwi.prepare()

            # 검색어 형태소 분석
            right_search_tokens = tokenize(right_search_term, kiwi)
            st.write(right_search_tokens)

            # 형태소 분석 결과를 활용하여 검색하기
            search_results_right = df[df['content'].str.contains(right_search_term, case=False)][['content','url','category']] #search_with_pos(df, right_search_tokens, 'content_pos')#d
            st.write("검색 건 수:",len(right_search_term))
            st.write(f"### {right_search_term}에 대한 검색 결과")
            #st.write(search_results_right)

            #검색 결과 표에 하이라이트 적용
            search_results_html_right = highlight_search_results(search_results_right, right_search_term)
            st.write(search_results_html_right, unsafe_allow_html=True)

def tokenize(text, kiwi):
    tokens = kiwi.analyze(text)
    token_list = []
    for token in tokens:
        for morph in token:
            token_list.append(morph)
    return token_list



def search_with_pos(df, search_term, search_term_pos):
    idx=[]
    for idx_num,row in enumerate(df.content_pos.tolist()):
        for token in row:
            if token.tag == search_term_pos and token.form == search_term:
                #print(idx_num)
                idx.append(idx_num)
    return df[['content','url','category']].loc[idx]

def highlight_text(text, keyword):
    # 검색어 하이라이트 처리
    highlighted_text = re.sub(re.escape(keyword), f"<span style='background-color: #ffff00'>{keyword}</span>", text, flags=re.IGNORECASE)
    return highlighted_text

def highlight_search_results(df, keyword):
    # 검색 결과 표에 검색어 하이라이트 적용
    df_html = df.to_html(escape=False)
    highlighted_html = re.sub(re.escape(keyword), f"<span style='background-color: #ffff00'>{keyword}</span>", df_html, flags=re.IGNORECASE)
    return highlighted_html

if __name__ == "__main__":
    main()


# def main():
#     st.set_page_config(layout="wide")  # Wide 모드로 설정

#     st.title("어휘 연어 관계 분석(콘코던스)")

#     # 파일 업로드
#     uploaded_file = st.file_uploader("Excel 파일 업로드", type=["xlsx","csv"])

#     if uploaded_file is not None:
#         df = load_data(uploaded_file)
#         if df is not None:
#             show_data(df)

# def load_data(uploaded_file):
#     try:
#         if uploaded_file.name.endswith('.xlsx'):
#             df = pd.read_excel(uploaded_file)
#         elif uploaded_file.name.endswith('.csv'):
#             df = pd.read_csv(uploaded_file)
#         return df
#     except Exception as e:
#         st.error(f"파일을 로드하는 중 오류가 발생했습니다: {e}")
#         return None

# def show_data(df):
#     st.write("### 데이터 미리보기")
#     st.write(df)
#     length=len(df)
#     st.write(f'전체 데이터 건수: {length}')

#     # 왼쪽 검색 기능
#     left_search_term = st.text_input("왼쪽 검색어 입력:")
#     # 오른쪽 검색 기능
#     right_search_term = st.text_input("오른쪽 검색어 입력:")

#     # 화면 분할
#     col1, col2 = st.columns(2)

#     # 왼쪽 열에 검색 결과 표시
#     with col1:
#         if left_search_term:
#             # '이름' 칼럼에서만 검색하기
#             search_results_left = df[df['content'].str.contains(left_search_term, case=False)]
#             st.write(f"### {left_search_term}에 대한 검색 결과")
#             # 검색 결과 표에 하이라이트 적용
#             search_results_html_left = highlight_search_results(search_results_left, left_search_term)
#             st.write(search_results_html_left, unsafe_allow_html=True)

#     # 오른쪽 열에 검색 결과 표시
#     with col2:
#         if right_search_term:
#             search_results_right = df[df['content'].str.contains(right_search_term, case=False)]
#             st.write(f"### {right_search_term}에 대한 검색 결과")
#             # 검색 결과 표에 하이라이트 적용
#             search_results_html_right = highlight_search_results(search_results_right, right_search_term)
#             st.write(search_results_html_right, unsafe_allow_html=True)




# def highlight_text(text, keyword):
#     # 검색어 하이라이트 처리
#     highlighted_text = re.sub(re.escape(keyword), f"<span style='background-color: #ffff00'>{keyword}</span>", text, flags=re.IGNORECASE)
#     return highlighted_text

# def highlight_search_results(df, keyword):
#     # 검색 결과 표에 검색어 하이라이트 적용
#     df_html = df.to_html(escape=False)
#     highlighted_html = re.sub(re.escape(keyword), f"<span style='background-color: #ffff00'>{keyword}</span>", df_html, flags=re.IGNORECASE)
#     return highlighted_html

# if __name__ == "__main__":
#     main()




# def semantic_search(df, search_term):
#     # TF-IDF 벡터화
#     tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

#     # 검색어를 TF-IDF 벡터로 변환
#     search_vector = tfidf_vectorizer.transform([search_term])

#     # 코사인 유사도 계산
#     cosine_similarities = cosine_similarity(search_vector, tfidf_matrix).flatten()

#     # 유사도가 높은 순으로 정렬하여 결과 반환
#     search_results = df.iloc[cosine_similarities.argsort()[::-1]]
#     return search_results   

        

        # # 통계 기능
        # st.write("### 통계")
        # st.write(get_keyword_statistics(search_results, search_term))

# def get_keyword_statistics(df, keyword):
#     keyword_stats = {}
#     keyword_regex = re.compile(r'\b{}\b'.format(re.escape(keyword)), re.IGNORECASE)

#     for index, row in df.iterrows():
#         text = row['이름']
#         matches = keyword_regex.findall(text)
#         for match in matches:
#             prev_word = get_previous_word(text, match)
#             next_word = get_next_word(text, match)
#             if match not in keyword_stats:
#                 keyword_stats[match] = {"total_occurrences": 1, "prev_words": {prev_word: 1}, "next_words": {next_word: 1}}
#             else:
#                 keyword_stats[match]["total_occurrences"] += 1
#                 keyword_stats[match]["prev_words"][prev_word] = keyword_stats[match]["prev_words"].get(prev_word, 0) + 1
#                 keyword_stats[match]["next_words"][next_word] = keyword_stats[match]["next_words"].get(next_word, 0) + 1

#     return keyword_stats

# def get_previous_word(text, keyword):
#     start_index = text.lower().index(keyword.lower())
#     text_before_keyword = text[:start_index].split()[-1]
#     return text_before_keyword.strip("',.!?-\"")

# def get_next_word(text, keyword):
#     start_index = text.lower().index(keyword.lower()) + len(keyword)
#     text_after_keyword = text[start_index:].split()[0]
#     return text_after_keyword.strip("',.!?-\"")
# import streamlit as st
# import pandas as pd
# import re

# def main():
#     st.title("Excel 데이터 검색, 정렬 및 통계")

#     # 파일 업로드
#     uploaded_file = st.file_uploader("Excel 파일 업로드", type=["xlsx", "csv"])

#     if uploaded_file is not None:
#         df = load_data(uploaded_file)
#         if df is not None:
#             show_data(df)

# def load_data(uploaded_file):
#     try:
#         if uploaded_file.name.endswith('.xlsx'):
#             df = pd.read_excel(uploaded_file)
#         elif uploaded_file.name.endswith('.csv'):
#             df = pd.read_csv(uploaded_file)
#         return df
#     except Exception as e:
#         st.error(f"파일을 로드하는 중 오류가 발생했습니다: {e}")
#         return None

# def show_data(df):
#     st.write("### 데이터 미리보기")
#     st.write(df)

#     # 왼쪽 검색 기능
#     left_search_term = st.text_input("왼쪽 검색어 입력:")
#     # 오른쪽 검색 기능
#     right_search_term = st.text_input("오른쪽 검색어 입력:")

#     # 검색 결과
#     if left_search_term or right_search_term:
#         search_results = search_data(df, left_search_term, right_search_term)
#         st.write("### 검색 결과")
#         st.write(search_results)

# def search_data(df, left_search_term, right_search_term):
#     left_results = pd.DataFrame()
#     right_results = pd.DataFrame()

#     if left_search_term:
#         left_results = df[df['content'].str.contains(left_search_term, case=False)]
#     if right_search_term:
#         right_results = df[df['content'].str.contains(right_search_term, case=False)]

#     # 두 검색어의 교집합을 찾습니다.
#     search_results = pd.concat([left_results, right_results], ignore_index=True)
#     return search_results

# if __name__ == "__main__":
#     main()
