import streamlit as st  # 웹 애플리케이션 구축을 위한 라이브러리
from langchain_community.document_loaders import PyPDFLoader  # PDF 파일을 로드하고 텍스트를 추출하는 도구
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 긴 텍스트를 작은 청크로 분할하는 도구
from langchain_community.embeddings import OllamaEmbeddings  # Ollama를 사용한 텍스트 임베딩 생성
from langchain_community.vectorstores import Chroma  # 벡터 데이터베이스 저장소
from langchain_community.llms import Ollama  # Ollama LLM(대규모 언어 모델) 인터페이스
from langchain.chains import RetrievalQA  # langchain의 RetrievalQA 체인
from langchain.prompts import PromptTemplate  # 프롬프트 템플릿
from langchain.chains import LLMChain  # LLM 체인
import tempfile  # 임시 파일 처리
import os  # 파일 시스템 작업
import time  # 시간 측정을 위한 모듈 추가
import subprocess  # 터미널 명령 실행

# Streamlit 페이지 기본 설정
st.set_page_config(
  page_title="myLLM 플랫폼",
  page_icon="📚",
  layout="wide"
)

# 메인 페이지 제목
st.title("📚 myLLM 플랫폼 with Ollama")


# 1️⃣ 로컬에 설치된 Ollama 모델 목록 가져오기
def get_ollama_models():
  try:
    result = subprocess.run("ollama list | sort -k1", capture_output=True, text=True, shell=True)
    models = [line.split()[0] for line in result.stdout.splitlines() if line]
    return models[1:]
  except Exception as e:
    st.error(f"Ollama 모델 목록을 불러오는 데 실패했습니다: {e}")
    return []


# 사용자가 보유한 Ollama 모델 리스트 불러오기
available_models = get_ollama_models()

# 모델 선택 UI (기본값: deepseek-r1:14b)
selected_model = st.sidebar.selectbox(
  "추론(Reasoning)에 사용할 Ollama 모델을 선택하세요:",
  available_models,
  index=available_models.index("deepseek-r1:14b") if "deepseek-r1:14b" in available_models else 0,
  key="reasoning_model"
)

# 답변 생성에 사용할 모델 선택 UI
answer_model = st.sidebar.selectbox(
  "답변 생성에 사용할 Ollama 모델을 선택하세요:",
  available_models,
  index=available_models.index("exaone3.5:latest") if "exaone3.5:latest" in available_models else 0,
  key="answer_model"
)

# PDF 파일 업로드 위젯
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])

# 2️⃣ 파일이 업로드되면 실행되는 로직
if uploaded_file is not None:
  # 업로드된 파일을 임시 파일로 저장
  with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    temp_path = tmp_file.name  # 임시 파일 경로 저장

  try:
    # PDF 파일 로드 및 텍스트 추출
    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    # 텍스트를 작은 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)

    # 문서 처리가 완료되면 임베딩 및 질문 처리 진행
    if 'splits' in locals():
      # 3️⃣ 선택한 모델을 기반으로 텍스트 임베딩 설정
      embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
      )

      # 벡터 저장소 설정 및 문서 저장
      persist_directory = "./.chroma"

      # 벡터 저장소 설정
      vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)

      # 4️⃣ 선택한 모델을 기반으로 LLM 설정
      reasoning_llm = Ollama(
        model=selected_model,  # 사용자가 선택한 추론 모델 반영
        stop=["</think>"],
        base_url="http://localhost:11434"
      )

      answer_llm = Ollama(
        model=answer_model,  # 사용자가 선택한 답변 생성 모델 반영
        temperature=0,
        base_url="http://localhost:11434"
      )

      # RetrievalQA 체인 설정 (추론용)
      retrieval_chain = RetrievalQA.from_chain_type(
        llm=reasoning_llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
      )

      # 답변 생성을 위한 프롬프트 템플릿 정의
      answer_template = """
          당신은 학술 문서를 분석하고 질문에 답변하는 AI 어시스턴트입니다.

          다음은 질문에 대한 초기 분석 결과입니다:
          {reasoning_result}

          위 분석 결과를 바탕으로, 다음 질문에 대해 명확하고 간결하게 답변해주세요:
          {question}

          문서의 내용에 근거하여 답변하되, 학술적이고 정확한 표현을 사용하세요.
          """

      answer_prompt = PromptTemplate(
        template=answer_template,
        input_variables=["reasoning_result", "question"]
      )

      answer_chain = LLMChain(
        llm=answer_llm,
        prompt=answer_prompt
      )

      # 사용자 질문 입력 받기
      user_question = st.text_input("문서에 대해 질문해보세요:")

      # 질문이 입력되면 실행
      if user_question:
        start_time = time.time()

        with st.spinner('답변을 생성하고 있습니다...'):
          progress_placeholder = st.empty()
          progress_placeholder.text("1/2 단계: 관련 정보 검색 및 추론 중...")

          # 1단계: 추론 모델로 관련 정보 검색 및 분석
          reasoning_result = retrieval_chain({"query": user_question})
          reasoning_output = reasoning_result["result"]

          progress_placeholder.text("2/2 단계: 최종 답변 생성 중...")

          # 2단계: 답변 생성 모델로 최종 답변 만들기
          final_answer = answer_chain.run(
            reasoning_result=reasoning_output,
            question=user_question
          )

          end_time = time.time()
          elapsed_time = round(end_time - start_time, 2)

          progress_placeholder.empty()

          # 답변 출력
          st.write("### 🤖 AI Answer:")
          st.write(final_answer)

          # 상세 정보 확인 옵션 (접이식)
          with st.expander("추론 과정 상세 보기"):
            st.markdown("#### 추론 모델의 분석 결과:")
            st.write(reasoning_output)

          st.info(f"⏱️ 답변 생성 소요 시간: {elapsed_time}초")

  finally:
    # 임시 파일 삭제
    os.unlink(temp_path)

# 사이드바에 사용 설명 추가
with st.sidebar:
  st.header("사용 방법")
  st.write("""
    1. 추론(Reasoning)에 사용할 모델과 답변 생성에 사용할 모델을 선택하세요.
    2. PDF 파일을 업로드하세요.
    3. 문서에 대해 궁금한 점을 질문하세요.
    4. AI가 두 단계로 문서를 분석하여 답변을 제공합니다.
    """)

  st.markdown("---")

  st.header("두 단계 처리 과정")
  st.write("""
    **추론 단계 (Reasoning)**
    - 문서에서 질문과 관련된 정보를 검색합니다.
    - 검색된 정보를 바탕으로 초기 분석을 수행합니다.

    **답변 생성 단계 (Answer Generation)**
    - 추론 단계의 결과물을 바탕으로 더 정제된 최종 답변을 생성합니다.
    - 보다 자연스럽고 명확한 답변을 제공합니다.
    """)
