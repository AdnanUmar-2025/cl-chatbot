from flask import Flask, request, render_template, jsonify
from langchain_setup import build_chain, vectorstore
import os
import re
from langchain_core.runnables.base import Runnable

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_files")
def get_files():
    category = request.args.get("category")
    data_root = "data"
    folder_path = os.path.join(data_root, category)

    if not os.path.isdir(folder_path):
        return jsonify({"files": []})

    pdfs = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    return jsonify({"files": pdfs})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    source_file = data.get("pdfFile", "").strip()
    category = data.get("category", "").strip()

    if not query:
        return jsonify({"answer": "❗ Please ask a question."}), 400

    try:
        # Normal chain response (includes clarification if needed)
        chain = build_chain(category, source_file, query)

        # Only call invoke if chain is a Runnable
        if isinstance(chain, Runnable):
            answer = chain.invoke({"question": query})
        else:
            answer = str(chain)

        return jsonify({"answer": answer})

    except Exception as e:
        print("❌ Error during chain execution:", e)
        return jsonify({"answer": f"❌ Error: {str(e)}"}), 500
    
if __name__ == "__main__":
    app.run(debug=True)