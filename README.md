# LangChain RAG Study Project

이 프로젝트는 LangChain을 사용한 RAG(Retrieval-Augmented Generation) 시스템의 다양한 구현 방법과 고급 기법을 학습하기 위한 프로젝트입니다.

## 프로젝트 구조

```
langchain_rag/
├── 1.langchain_llm_test.ipynb              # LangChain LLM 기본 테스트
├── 2.rag_with_chroma.ipynb                 # ChromaDB를 사용한 기본 RAG 구현
├── 2.1.rag_with_chroma+upstage.ipynb       # ChromaDB + Upstage 모델을 사용한 RAG
├── 3.rag_without_langchain_with_chroma.ipynb # LangChain 없이 ChromaDB 직접 사용
├── 4.rag_with_pinecone.ipynb               # Pinecone을 사용한 RAG 구현
├── 4.1.rag_with_pinecone_modified_ask.ipynb # Pinecone + 쿼리 강화 기법 적용
├── ollama_langchain.ipynb                  # Ollama 로컬 LLM을 사용한 RAG
├── tax.docx                                # 소득세 관련 문서 (원본)
├── tax-markdown.docx                       # 마크다운 변환된 문서
├── chroma/                                 # ChromaDB 저장소
└── README.md
```

## 주요 학습 내용

### 1. 다양한 벡터 데이터베이스 활용
- **ChromaDB**: 로컬 벡터 데이터베이스로 빠른 프로토타이핑
- **Pinecone**: 클라우드 기반 벡터 데이터베이스로 확장성 있는 솔루션

### 2. 다양한 LLM 모델 사용
- **OpenAI GPT**: GPT-4o-mini를 사용한 고품질 답변 생성
- **Upstage Solar**: 한국어 특화 모델로 정확한 답변 생성
- **Ollama Llama3**: 로컬 LLM으로 프라이버시 보장

### 3. 문서 전처리 및 임베딩
- **문서 로딩**: DOCX 파일을 텍스트로 변환
- **텍스트 분할**: CharacterTextSplitter를 사용한 청크 생성
- **임베딩 모델**: OpenAI, Upstage 임베딩 모델 활용

### 4. RAG 쿼리 강화 기법
- **질문 변환**: 사용자 질문을 더 적절한 형태로 변환
- **체인 조합**: LangChain을 사용한 복합 체인 구성
- **프롬프트 엔지니어링**: 효과적인 프롬프트 설계

### 5. 이미지 → 마크다운 데이터 전처리
- 문서를 마크다운 형식으로 변환하여 구조화된 데이터 활용

## 설치 방법

### 1. 가상환경 생성 및 활성화

```bash
# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
venv\Scripts\activate
```

### 2. 필요한 패키지 설치

```bash
# 기본 LangChain 패키지
pip install langchain langchain-community langchain-openai

# 벡터 데이터베이스
pip install langchain-chroma chromadb
pip install langchain-pinecone pinecone-client

# 한국어 특화 모델
pip install langchain-upstage

# 로컬 LLM
pip install langchain-ollama ollama

# 문서 처리
pip install docx2txt python-dotenv

# 추가 유틸리티
pip install langchain-text-splitters langchainhub
```

### 3. 환경변수 설정

```bash
# .env 파일 생성 후 다음 키들 설정
OPENAI_API_KEY=your_openai_api_key
UPSTAGE_API_KEY=your_upstage_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## 사용 방법

### Jupyter 노트북 실행

```bash
jupyter notebook
# 또는
jupyter lab
```

### 각 노트북별 설명

1. **1.langchain_llm_test.ipynb**: LangChain 기본 사용법 학습
2. **2.rag_with_chroma.ipynb**: ChromaDB를 사용한 기본 RAG 구현
3. **2.1.rag_with_chroma+upstage.ipynb**: 한국어 특화 모델 활용
4. **3.rag_without_langchain_with_chroma.ipynb**: LangChain 없이 직접 구현
5. **4.rag_with_pinecone.ipynb**: Pinecone 클라우드 벡터 DB 활용
6. **4.1.rag_with_pinecone_modified_ask.ipynb**: 쿼리 강화 기법 적용
7. **ollama_langchain.ipynb**: 로컬 LLM을 사용한 RAG

## 주요 라이브러리

### 핵심 프레임워크
- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **LangChain Community**: 다양한 통합 도구들

### LLM 모델
- **OpenAI**: GPT-4o-mini, text-embedding-3-large
- **Upstage**: Solar 모델, solar-embedding-1-large
- **Ollama**: Llama3 로컬 모델

### 벡터 데이터베이스
- **ChromaDB**: 로컬 벡터 데이터베이스
- **Pinecone**: 클라우드 벡터 데이터베이스

### 문서 처리
- **docx2txt**: DOCX 파일 텍스트 추출
- **langchain-text-splitters**: 텍스트 청킹

## 학습 목표

1. **RAG 시스템의 기본 원리 이해**
   - 문서 로딩, 청킹, 임베딩, 검색, 생성 과정 학습

2. **다양한 벡터 데이터베이스 활용**
   - 로컬 vs 클라우드 벡터 DB의 장단점 비교
   - 각 DB의 특성에 맞는 활용 방법

3. **LLM 모델 비교 및 선택**
   - 상용 모델 vs 로컬 모델의 트레이드오프
   - 한국어 특화 모델의 활용

4. **고급 RAG 기법**
   - 쿼리 변환 및 강화
   - 체인 조합을 통한 복잡한 워크플로우 구성
   - 프롬프트 엔지니어링

5. **실제 응용**
   - 소득세 관련 문서를 활용한 실무 RAG 시스템
   - 이미지 → 마크다운 전처리 파이프라인

## 향후 계획

- [ ] Streamlit을 활용한 챗봇 웹 애플리케이션 구현
- [ ] 다양한 문서 형식 지원 (PDF, 이미지, 웹페이지)
- [ ] 멀티모달 RAG 시스템 구현
- [ ] 성능 최적화 및 벤치마킹

## 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [OpenAI API 문서](https://platform.openai.com/docs)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [Pinecone 문서](https://docs.pinecone.io/)
- [Upstage API 문서](https://console.upstage.ai/)
- [Ollama 문서](https://ollama.ai/)



