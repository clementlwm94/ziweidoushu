from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient

from astro_chart import full_chart_generation
from phoenix.otel import register

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "zhiwei_DAG")
EMBEDDING_MODEL_HANDLE = os.getenv("QDRANT_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-zh")
SPARSE_MODEL_HANDLE = os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/BM25")
STORE_PATH = os.getenv("QDRANT_STORE_PATH", "./store_location")
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = float(os.getenv("QDRANT_LLM_TEMPERATURE", "0.3"))
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", PHOENIX_COLLECTOR_ENDPOINT)


embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_HANDLE)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_HANDLE)

client = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
vectorstore = QdrantVectorStore(
    embedding=embeddings,
    client=client,
    collection_name=COLLECTION_NAME,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
)

fs = LocalFileStore(STORE_PATH)
doc_store = create_kv_docstore(fs)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=doc_store,
    child_splitter=child_splitter,
    search_kwargs={"k": 10},
    id_key="source_id"
)

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor,
)

query_prompt_text = """
角色: 你是一个专业的紫微斗数检索查询生成器，用于 RAG(检索增强生成)。

目标: 基于用户的命盘关键信息与具体问题，生成若干高质量、可直接用于检索的查询字符串，覆盖问题最核心的宫位与关键星曜组合，并按相关性排序。

约束:
- 仅使用用户的命盘中出现的宫位、主星、辅星、煞星；不要编造。
- 若用户的命盘未提供四化，不要构造四化查询；若提供，至少包含一条带四化的查询。
- 使用标准中文术语与命名；短语之间用空格分隔；避免冗余词。
- 每条查询尽量精炼但信息完整；生成 1-3 条，按相关性从高到低排序。
- 仅输出 JSON 字符串数组；不要任何解释、标题或 Markdown 代码块。

步骤:
1) 解析问题与领域（如事业财运、感情婚姻、健康、人际等），确定最核心宫位。
2) 从命盘中提取该核心宫位及其三方四正的关键主星、**如果宮位沒有核心（空宮）主星，用對面宮位的主星來代替**, 重要辅星（左辅、右弼、文昌、文曲）、煞星（擎羊、陀罗、火星、铃星、地空、地劫。
3) 组合查询，覆盖核心宫位、关键星曜影响，去重并按相关性排序。

查询模板（仅使用以下几类，其一或多条）:
- 模式A: 宫位-星曜-主题
  形式: [宫位] [星曜组合] [问题领域] 解释
  例: 夫妻宫 廉贞七杀 感情婚姻 解释
- 模式B: 星曜-四化-宫位
  形式: [星曜] 化[禄/权/科/忌]入 [宫位] 对[问题领域]的影响
  例: 太阳化忌入父母宫 对学业的影响
- 模式C: 宫位-煞/辅
  形式: [宫位] 遇 [煞星/辅星] 作用
  例: 命宫 擎羊同宫 影响
- 模式D: 宫位关系/格局
  形式: [宫位A] [宫位B] [关系类型] 影响
      或 [星曜组合] [宫位] [格局名称] 格局
  例: 命宫 迁移宫 对照 影响
      紫微破军在丑未宫 紫府朝垣格

输入:
- 用户命盘: {user_chart}
- 用户问题: {user_question}

输出（严格遵守）:
- 仅输出 JSON 字符串数组；不含任何解释、前后缀、注释或代码块标记。
"""

summary_prompt_text = """
角色: 你是一个紫微斗数命盘分析助手，需要根据检索到的文档段落，总结与用户问题最相关的命盘要点。

输入:
- 用户命盘: {user_chart}
- 用户问题: {user_question}
- 检索文段: {retrieved_passages}

指令:
1. 标出与问题高度相关的紫微斗数要点，匹配用户命盘。
2. 明确指明信息是否来自命盘还是检索文段。
3. 若检索文段无关，指出信息缺口。
4. 仅使用中文，结构化输出。
5. 不做结论或建议；不要编造命盘或文段外的信息。

输出格式:
关键信息:
- … 
- …
命盘关联:
- …
- …
信息缺口:
- …
"""

answer_prompt_text = """
角色: 你是专业紫微斗数顾问，需要结合摘要要点与命盘回答用户问题。

输入:
- 用户命盘: {user_chart}
- 用户问题: {user_question}
- 摘要要点: {summary_text}

指令:
1. 基于摘要要点,结合命盘结构说明关键影响,如果摘要的信息没办法回答请用你多年紫薇斗数的经验来回答问题
2. 明确回答用户问题，分条阐述理由。
3. 请尽量给出谨慎、可执行的建议
4. 全文使用中文，条理清晰。

输出格式:
最终结论: …
关键理由:
- …
- …
建议:
- …
"""

question_translation_prompt_text = """
You are a multilingual assistant. Detect the primary language of the user's question and provide a fluent Simplified Chinese translation.

Return a valid JSON object with keys:
- "language": ISO-639-1 lower-case language code of the input (e.g. "zh", "en", "fr"). Use "zh" if the question is already Chinese.
- "translation": the question rewritten in Simplified Chinese. If the input is already Chinese, repeat it.

Question: {text}
"""

answer_translation_prompt_text = """
You are a professional translator. Translate the following Simplified Chinese response into {target_language} while preserving meaning, tone, and formatting.
Only output the translated text with no explanations.

中文原文:
{text}
"""

query_prompt = PromptTemplate(
    input_variables=["user_question", "user_chart"],
    template=query_prompt_text,
)
summary_prompt = PromptTemplate(
    input_variables=["user_question", "user_chart", "retrieved_passages"],
    template=summary_prompt_text,
)
answer_prompt = PromptTemplate(
    input_variables=["user_question", "user_chart", "summary_text"],
    template=answer_prompt_text,
)
question_translation_prompt = PromptTemplate(
    input_variables=["text"],
    template=question_translation_prompt_text,
)
answer_translation_prompt = PromptTemplate(
    input_variables=["text", "target_language"],
    template=answer_translation_prompt_text,
)

# configure the Phoenix tracer
tracer_provider = register(
  project_name="ziweidoushu", # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed OI dependencies
)
tracer = tracer_provider.get_tracer(__name__)


def _create_llm(api_key: Optional[str], llm_model: Optional[str]) -> ChatOpenAI:
    target_model = llm_model or LLM_MODEL
    return ChatOpenAI(model=target_model, temperature=LLM_TEMPERATURE, api_key=api_key)


def pretty_print_docs(docs: Iterable[Any]) -> str:
    """Return a formatted string of retrieved documents and optionally persist it."""
    docs = list(docs)
    if not docs:
        return ""

    separator = "\n{}\n".format("-" * 100)
    doc_text = separator.join(
        [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )
    return doc_text


def parse_json_list(raw_text: str) -> List[str]:
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return []


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def generate_queries(
    question: str,
    user_chart: Dict[str, Any],
    top_n: Optional[int] = 3,
    llm: Optional[ChatOpenAI] = None,
) -> List[str]:
    if llm is None:
        raise ValueError("LLM client not provided")
    chain = query_prompt | llm
    ai_message = chain.invoke({"user_question": question, "user_chart": user_chart})
    queries = parse_json_list(ai_message.content)
    return queries[:top_n] if top_n else queries


def retrieve_documents(queries: Iterable[str]) -> List[Any]:
    seen_doc_ids = set()
    aggregated = []
    for query in queries:
        docs = compression_retriever.invoke(query)
        for doc in docs:
            doc_id = (
                doc.metadata.get("doc_id")
                or doc.metadata.get("source")
                or doc.metadata.get("id")
            )
            dedupe_key = doc_id or id(doc)
            if dedupe_key in seen_doc_ids:
                continue
            seen_doc_ids.add(dedupe_key)
            aggregated.append(doc)
    return aggregated


def summarize_passages(
    question: str,
    user_chart: Dict[str, Any],
    passages: str,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    if llm is None:
        raise ValueError("LLM client not provided")
    chain = summary_prompt | llm
    payload = {
        "user_question": question,
        "user_chart": user_chart,
        "retrieved_passages": passages,
    }
    ai_message = chain.invoke(payload)
    return ai_message.content


def answer_question(
    question: str,
    user_chart: Dict[str, Any],
    summary_text: str,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    if llm is None:
        raise ValueError("LLM client not provided")
    chain = answer_prompt | llm
    payload = {
        "user_question": question,
        "user_chart": user_chart,
        "summary_text": summary_text,
    }
    ai_message = chain.invoke(payload)
    return ai_message.content


def translate_question_to_chinese(
    question: str,
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, str]:
    if llm is None:
        raise ValueError("LLM client not provided")
    chain = question_translation_prompt | llm
    ai_message = chain.invoke({"text": question})
    data = parse_json_object(ai_message.content)
    detected = (data.get("language") or "zh").strip().lower()
    if detected.startswith("zh"):
        detected = "zh"
    translated = data.get("translation") or question
    if not isinstance(translated, str):
        translated = str(translated)
    return {
        "question_language": detected,
        "translated_question": translated,
    }


def translate_answer_from_chinese(
    answer_text: str,
    target_language: Optional[str],
    llm: Optional[ChatOpenAI] = None,
) -> str:
    if not target_language:
        return answer_text

    normalized = target_language.strip().lower()
    if normalized in {"zh", "zh-cn", "zh-hans", "zh_tw", "zh-hant", "chinese"}:
        return answer_text

    if llm is None:
        raise ValueError("LLM client not provided")
    chain = answer_translation_prompt | llm
    ai_message = chain.invoke({"text": answer_text, "target_language": target_language})
    return ai_message.content.strip()


def _with_chart(inputs: Dict[str, Any]) -> Dict[str, Any]:
    chart = full_chart_generation(inputs["birth_date"], inputs["birth_hour"], inputs["gender"])
    return {**inputs, "user_chart": chart}


def _with_llm(inputs: Dict[str, Any]) -> Dict[str, Any]:
    llm = _create_llm(inputs.get("api_key"), inputs.get("llm_model"))
    retained = {k: v for k, v in inputs.items() if k != "api_key"}
    return {**retained, "llm": llm}


def _with_translated_question(inputs: Dict[str, Any]) -> Dict[str, Any]:
    translation = translate_question_to_chinese(
        inputs["question"],
        llm=inputs.get("llm"),
    )
    translated_question = translation.get("translated_question") or inputs["question"]
    question_language = translation.get("question_language", "zh")
    return {
        **inputs,
        "translated_question": translated_question,
        "question_language": question_language,
    }


def _with_queries(inputs: Dict[str, Any]) -> Dict[str, Any]:
    queries = generate_queries(
        inputs["translated_question"],
        inputs["user_chart"],
        top_n=inputs.get("top_n_queries", 3),
        llm=inputs.get("llm"),
    )
    return {**inputs, "queries": queries}


def _with_documents(inputs: Dict[str, Any]) -> Dict[str, Any]:
    documents = retrieve_documents(inputs["queries"])
    return {**inputs, "documents": documents}


def _with_documents_text(inputs: Dict[str, Any]) -> Dict[str, Any]:
    documents_text = pretty_print_docs(inputs["documents"])
    return {**inputs, "documents_text": documents_text}


def _with_summary(inputs: Dict[str, Any]) -> Dict[str, Any]:
    summary = summarize_passages(
        inputs["translated_question"],
        inputs["user_chart"],
        inputs["documents_text"],
        llm=inputs.get("llm"),
    )
    retained = {k: v for k, v in inputs.items() if k not in {"documents", "documents_text"}}
    return {**retained, "summary": summary}


def _with_answer(inputs: Dict[str, Any]) -> Dict[str, Any]:
    answer = answer_question(
        inputs["translated_question"],
        inputs["user_chart"],
        inputs["summary"],
        llm=inputs.get("llm"),
    )
    return {**inputs, "answer_chinese": answer}


def _with_localized_answer(inputs: Dict[str, Any]) -> Dict[str, Any]:
    localized_answer = translate_answer_from_chinese(
        inputs["answer_chinese"],
        target_language=inputs.get("question_language"),
        llm=inputs.get("llm"),
    )
    return {**inputs, "answer": localized_answer}


def _finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": payload["question"],
        "translated_question": payload.get("translated_question"),
        "user_chart": payload["user_chart"],
        "queries": payload["queries"],
        "summary": payload["summary"],
        "answer": payload["answer"],
        "llm_model": payload.get("llm_model"),
        "question_language": payload.get("question_language"),
        "answer_chinese": payload.get("answer_chinese"),
    }


workflow_chain = (
    RunnableLambda(_with_chart)
    | RunnableLambda(_with_llm)
    | RunnableLambda(_with_translated_question)
    | RunnableLambda(_with_queries)
    | RunnableLambda(_with_documents)
    | RunnableLambda(_with_documents_text)
    | RunnableLambda(_with_summary)
    | RunnableLambda(_with_answer)
    | RunnableLambda(_with_localized_answer)
    | RunnableLambda(_finalize)
)


def run_qdrant_rag_workflow(
    *,
    question: str,
    birth_date: str,
    birth_hour: int,
    gender: str,
    top_n_queries: int = 5,
    api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full Qdrant RAG workflow and return intermediate artifacts."""
    payload = {
        "question": question,
        "birth_date": birth_date,
        "birth_hour": birth_hour,
        "gender": gender,
        "top_n_queries": top_n_queries,
        "api_key": api_key,
        "llm_model": llm_model,
    }
    return workflow_chain.invoke(payload)


__all__ = [
    "generate_queries",
    "retrieve_documents",
    "summarize_passages",
    "answer_question",
    "run_qdrant_rag_workflow",
]
