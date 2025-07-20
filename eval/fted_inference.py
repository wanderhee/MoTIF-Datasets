import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from motif import model_init, mm_infer
from motif.utils import disable_torch_init

# RAG 模型和数据库路径
MODEL_PATH = 'VideoLLaMA2-tuned-highway-epoch5/videollama2qwen2_downstream_sft/finetune_siglip_tcv35_7b_16f_lora'
DB_FAISS_PATH = "/workspace/sdgs_test/VideoLLaMA2.1/videollama2/rag/db_faiss"  # 向量数据库路径

# VideoLLaMA2 模型路径
VIDEO_LLA_MA2_MODEL_PATH = 'VideoLLaMA2-tuned-highway-epoch5/videollama2qwen2_downstream_sft/finetune_siglip_tcv35_7b_16f_lora'
VIDEO_LLA_MA2_BASE = '../VideoLLaMA2.1-7B-16F'

def load_rag_model():
    """加载 RAG 模型和 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer

def load_vector_db():
    """加载 FAISS 向量数据库"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                       model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def create_rag_chain(model, tokenizer, db):
    """创建 RAG 链"""
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    num_return_sequences=1)
    llm = HuggingFacePipeline(pipeline=pipe)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def refine_context(retrieved_context, query, top_n=3):
    """根据问题对检索到的上下文进行二次筛选"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    documents = [doc.page_content for doc in retrieved_context]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents + [query])
    query_vector = tfidf_matrix[-1]
    document_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    refined_context = [retrieved_context[i] for i in top_indices]
    return refined_context

def generate_response(qa_chain, query):
    """生成回复，优先使用 RAG 检索结果"""
    response = qa_chain({"query": query})
    retrieved_context = response["source_documents"]
    rag_answer = response["result"]

    if not retrieved_context or "不知道答案" in rag_answer or "无法回答" in rag_answer:
        return "抱歉，未能找到相关答案。", retrieved_context
    else:
        combined_context = "\n".join([doc.page_content for doc in retrieved_context])
        return combined_context, retrieved_context

# 自定义 Prompt
prompt_template = """使用以下上下文来回答最后的问题。如果你不知道答案，就结合 deepseek 模型进行输出，不要试图编造答案。
上下文：{context}
问题：{question}
有用的回答："""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def inference():
    disable_torch_init()

    # 加载 RAG 模型和向量数据库
    rag_model, rag_tokenizer = load_rag_model()
    db = load_vector_db()
    rag_chain = create_rag_chain(rag_model, rag_tokenizer, db)

    # 视频推理
    modal = 'video'
    modal_path = 'assets/cat_and_chicken.mp4'
    instruct = 'What animals are in the video, what are they doing, and how does the video feel?'

    # 加载 VideoLLaMA2 模型
    model_path = VIDEO_LLA_MA2_MODEL_PATH
    model_base = VIDEO_LLA_MA2_BASE
    model, processor, tokenizer = model_init(model_path)

    # 使用 RAG 检索相关上下文
    rag_context, source_documents = generate_response(rag_chain, instruct)
    print("RAG 检索到的上下文：")
    print(rag_context)

    # 将 RAG 检索到的上下文作为额外信息传递给 VideoLLaMA2
    full_instruct = f"{rag_context}\n{instruct}"
    output = mm_infer(processor[modal](modal_path), full_instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

    print("VideoLLaMA2 推理结果：")
    print(output)

if __name__ == "__main__":
    inference()