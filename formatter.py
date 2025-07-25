# formatter.py
def format_quant_solution(question: str, solution_steps: list, final_answer: str) -> str:
    formatted = []

    formatted.append("## 🧮 Quant Question (Formatted)\n")
    formatted.append(f"**Q.**\n{question.strip()}\n")

    formatted.append("---\n")
    formatted.append("### ✅ Solution:\n")
    for step in solution_steps:
        formatted.append(step)

    formatted.append("\n---\n")
    formatted.append(f"### ✅ Final Answer: **{final_answer.strip()}**")

    return "\n".join(formatted)