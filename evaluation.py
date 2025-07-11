import json
from difflib import SequenceMatcher
  # assumes qa_chain is initialized in app.py
from app import exported_qa_chain as qa_chain


# def similarity_score(answer, expected):
#     return SequenceMatcher(None, answer.lower(), expected.lower()).ratio()


def load_test_cases(filename="test_questions.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

def similarity_score(answer, expected):
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    answer_emb = embeddings.embed_query(answer)
    expected_emb = embeddings.embed_query(expected)
    cosine_sim = np.dot(answer_emb, expected_emb) / (np.linalg.norm(answer_emb) * np.linalg.norm(expected_emb))
    return cosine_sim

def evaluate_qa_system():
    test_cases = load_test_cases()
    if not test_cases:
        print("❗ No test cases loaded. Check test_questions.json.")
        return
    total = len(test_cases)
    correct = 0
    threshold = 0.7  # Lowered for semantic similarity

    print(f"\n📊 Starting Evaluation with {total} test cases...\n")
    for i, test in enumerate(test_cases, 1):
        question = test["question"]
        expected = test["expected_answer"]
        try:
            predicted_result = qa_chain.invoke({"query": question})
            predicted = predicted_result["result"].strip()
            # source_docs = predicted_result["source_documents"]
            print(f"\nQ{i}: {question}")
            print(f"🔹 Expected: {expected}")
            print(f"🔸 Predicted: {predicted}")
            # print("\n📚 Retrieved Chunks:")
            # for doc in source_docs:
            #     print("•", doc.page_content[:300].replace("\n", " "), "\n")
            score = similarity_score(predicted, expected)
            is_correct = score >= threshold
            print(f"✅ Match Score: {round(score * 100, 2)}% — {'PASS' if is_correct else 'FAIL'}\n")
            if is_correct:
                correct += 1
        except Exception as e:
            print(f"❌ Error on question {i}: {e}")
            continue
         # 🔚 Final summary
    accuracy = (correct / total) * 100
    print("📈 Final Evaluation Summary")
    print(f"✔️ Correct Answers: {correct}/{total}")
    print(f"🎯 Overall Accuracy: {round(accuracy, 2)}%")

if __name__ == "__main__":
    evaluate_qa_system()
