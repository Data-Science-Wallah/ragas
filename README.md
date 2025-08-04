# 🧠 RAGAS Evaluation vs BLEU, ROUGE, BERTScore & MMI/TOIR

This project demonstrates how to evaluate **RAG (Retrieval-Augmented Generation)** pipelines using **RAGAS** — a modern, LLM-agnostic metric designed to analyze **answer relevancy**, **faithfulness**, and **hallucination detection** — compared to traditional NLP metrics like **BLEU**, **ROUGE**, **BERTScore**, and **MMI/TOIR**.

---

## 🎯 Why RAGAS?

Traditional metrics were not designed to evaluate answers from retrieval-based systems. They fail in these key areas:

| Metric     | Context Aware | Faithfulness | Hallucination Detection | Explanation |
|------------|----------------|---------------|---------------------------|-------------|
| BLEU       | ❌              | ❌             | ❌                         | Measures token-level overlap |
| ROUGE      | ❌              | ❌             | ❌                         | Based on recall; not contextual |
| BERTScore  | ✅              | ❌             | ❌                         | Uses embeddings but lacks grounding |
| MMI/TOIR   | ✅              | ⚠️              | ⚠️                         | Hard to scale; not explainable |
| **RAGAS**  | ✅              | ✅             | ✅                         | Multi-dimensional, LLM-optional, explainable |

---

## 🚀 Features

- ✅ Full RAGAS pipeline implementation in **Python + VSCode**
- 📂 CSV evaluation on `question`, `ground_truth`, `context`, `predicted_answer`
- ⚙️ Modular code with `pydantic`, `langchain`, and `transformers`
- 📊 Visualization via **Streamlit**
- 🔄 Compare RAGAS scores with BLEU, ROUGE, BERTScore, and MMI/TOIR

---

## 🧪 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
