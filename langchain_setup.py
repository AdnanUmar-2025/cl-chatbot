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
index_name = "career-launcher-index"

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
            "You are a VARC expert. Read the passage and answer the question with clear logic.\n"
            "- Be concise and direct\n"
            "- Justify your answer only from the context\n"
            "- Avoid over-explaining unless asked\n"
            "- Focus on identifying the best option if its MCQ\n"
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
def get_retriever(filters: dict, namespace: str):
    return vectorstore.as_retriever(
        search_kwargs={
            "k": 3,  # Increased from 2 to 3
            "filter": filters,
            "namespace": namespace
        }
    )

# === Chain Builder ===
def build_chain(category, source_file, user_query):
    match = re.search(r"\b(\d{1,2})\b", user_query)
    question_number = match.group(1) if match else None

    filters = {}
    if category:
        filters["subject"] = category
    if source_file:
        filters["source"] = source_file
    if question_number:
        filters["question_number"] = question_number

    retriever = get_retriever(filters, namespace=category)

    # Log retrieved context for debugging
    print("🧠 Retrieved Context:\n", retriever.invoke(user_query))

    prompt = choose_prompt({"subject": category, "question": user_query})

    supports_latex = False

    def clean_latex_output(answer: str):
        if not supports_latex:
            return answer.replace(r"\%", "%").replace(r"\\", "").replace(r"\frac", "").replace(r"\text", "").replace(r"\left", "").replace(r"\right", "").replace(r"\begin{align*", "").replace(r"\end{align*", "")
        return answer

    return (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "subject": lambda x: category or ""
        }
        | RunnableLambda(prompt.invoke)
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda x: clean_latex_output(x))
    )
