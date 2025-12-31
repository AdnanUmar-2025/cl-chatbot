from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os
import re

# === Load environment ===
load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = "career-launcher"

# === Setup LLM & Vectorstore ===
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

# === Prompt Selector ===
def choose_prompt(inputs):
    subject = inputs.get("subject", "").lower()

    if "quant" in subject:
        template = (
            "You are a Quant expert. Format answers in this structure:\n"
            "\n"
            "### 🎯 Solve:\n"
            "### 🧮 Assumptions:\n"
            "- State known variables and values\n"
            "\n"
            "### 🧩 Steps:\n"
            "1. Break problem into math logic\n"
            "2. Use intermediate calculations\n"
            "3. Prefer symbolic math using LaTeX: ( ... ) inline\n"
            "\n"
            "### ✏️ Calculation:\n"
            "Use math formatting with final formula\n"
            "\n"
            "### ✅ Final Answer: <value>\n"
            "\n"
            "Context:\n{context}\n"
            "\n"
            "Question:\n{question}\n"
            "\n"
            "Answer:"
        )
    elif "lrdi" in subject:
        template = (
            "You are a Logical Reasoning & Data Interpretation expert.\n"
            "Use logic and deduction to solve the question.\n"
            "\n"
            "Instructions:\n"
            "- Identify any constraints and variables from the context\n"
            "- Use grid or table logic if needed\n"
            "- Use paragraph breaks for readability\n"
            "- Finish clearly with:\n"
            "✅ Final Answer: <value or choice>\n"
            "\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
    elif "varc" in subject:
        template = (
            "You are a VARC expert. Answer in simple language. For vocabulary, give the correct option and meaning in one sentence. For RC, the option with brief reason. For summaries or key points, extract from context concisely.\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
    else:
        template = (
            "You are a helpful exam tutor.\n"
            "Use clean formatting and end with:\n"
            "✅ Final Answer: <value>\n"
            "\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )

    return ChatPromptTemplate.from_template(template)

# === Updated Retriever with Namespace ===
def get_retriever(filters: dict, namespace: str, k: int = 3):
    return vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "filter": filters,
            "namespace": namespace
        }
    )

# Helper to extract exercise sort key
def get_exercise_key(exercise_str):
    match = re.search(r'(\d+)', exercise_str)
    if match:
        return int(match.group(1))
    return float('inf')  # Unknown or non-numeric go last

# Helper to get clean question snippet (question text only, no options, strip colon)
def get_clean_snippet(page_content):
    parts = page_content.split("\n\n", 1)
    if len(parts) > 1:
        question_part = parts[1]
    else:
        question_part = page_content
    # Remove number
    question_text = re.sub(r'^\d+\.\s*', '', question_part, count=1).strip()
    # Stop at first option or colon
    question_text = re.split(r'\(\d+\)|\:', question_text)[0].strip()
    # Truncate if long
    return (question_text[:80] + "...") if len(question_text) > 80 else question_text

# === Chain Builder ===
def build_chain(category, source_file, user_query):
    filters = {}
    if category:
        filters["subject"] = category
    if source_file:
        filters["source"] = source_file

    # Extract exercise first
    exercise_match = re.search(r"(Diagnostic Exercise\s*[-\u2013]\s*\d+|Exercise\s*[-\u2013]\s*\d+)", user_query, re.IGNORECASE)
    exercise_name = exercise_match.group(1).strip() if exercise_match else None
    if exercise_name:
        filters["exercise"] = exercise_name
        cleaned_query = user_query.replace(exercise_match.group(0), "").strip()
    else:
        cleaned_query = user_query

    # Detect summary-style request
    summary_request = bool(re.search(r"\b(summarize|important points?|key points?|main ideas?)\b", user_query, re.IGNORECASE))

    # Now extract question number from cleaned_query
    match = re.search(r"\b(\d{1,2})\b", cleaned_query)
    question_number = match.group(1) if match else None
    if question_number:
        filters["question_number"] = question_number

    # Set k higher if no question number (exercise-level queries)
    k = 3 if question_number else 50

    retriever = get_retriever(filters, namespace=category, k=k)
    docs = retriever.invoke(user_query)
    full_context = "\n\n".join(doc.page_content for doc in docs)

    # If summary-style and exercise found → summarization mode
    if summary_request and exercise_name:
        prompt = ChatPromptTemplate.from_template(
            "You are an assistant. Summarize the following exercise into exactly the number of key points requested in the question.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        return (
            {
                "context": lambda x: full_context,
                "question": lambda x: user_query,
                "subject": lambda x: category or ""
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    # 🧠 Check for duplicate question number contexts
    if question_number and len(docs) > 1:
        passage_tuples = [
            (doc.metadata.get("exercise", ""), doc.metadata.get("question_number", ""), get_clean_snippet(doc.page_content), doc)
            for doc in docs
        ]
        distinct_passages = set((t[0], t[1], t[2]) for t in passage_tuples)
        if len(distinct_passages) > 1:
            sorted_passages = sorted(distinct_passages, key=lambda p: get_exercise_key(p[0]))
            options = "\n".join(f"- ({p[0]}) {p[1]}. {p[2]}" for p in sorted_passages)
            first_exercise = sorted_passages[0][0]
            selected_docs = [t[3] for t in passage_tuples if t[0] == first_exercise]
            full_context = "\n\n".join(doc.page_content for doc in selected_docs)
            
            ambiguity_note = f"⚠️ There are multiple questions numbered {question_number} in this file. Answering from the first one ({first_exercise}) by default.\n" \
                             f"If this isn't the one you want, please specify the exercise.\n\n" \
                             f"Other options:\n{options}"
            
            prompt = choose_prompt({"subject": category, "question": user_query})
            supports_latex = False

            def clean_latex_output(answer: str):
                if not supports_latex:
                    return answer.replace(r"\%", "%").replace(r"\\", "").replace(r"\frac", "").replace(r"\text", "").replace(r"\left", "").replace(r"\right", "").replace(r"\begin{align*", "").replace(r"\end{align*", "")
                return answer

            chain = (
                {
                    "context": lambda x: full_context,
                    "question": lambda x: x["question"],
                    "subject": lambda x: category or ""
                }
                | prompt
                | llm
                | StrOutputParser()
                | RunnableLambda(lambda x: clean_latex_output(x))
            )
            answer = chain.invoke({"question": user_query})
            return ambiguity_note + "\n\n" + answer

    prompt = choose_prompt({"subject": category, "question": user_query})
    supports_latex = False

    def clean_latex_output(answer: str):
        if not supports_latex:
            return answer.replace(r"\%", "%").replace(r"\\", "").replace(r"\frac", "").replace(r"\text", "").replace(r"\left", "").replace(r"\right", "").replace(r"\begin{align*", "").replace(r"\end{align*", "")
        return answer

    return (
        {
            "context": lambda x: full_context,
            "question": lambda x: x["question"],
            "subject": lambda x: category or ""
        }
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda x: clean_latex_output(x))
    )
