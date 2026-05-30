import os
from dotenv import load_dotenv
from google import genai

from app.vector_store import load_vector_store

load_dotenv()


class RetrieverAgent:
    """
    This agent retrieves the most relevant document chunks from FAISS.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def retrieve(self, query: str):
        vector_store = load_vector_store()

        results = vector_store.similarity_search(
            query=query,
            k=self.top_k
        )

        retrieved_chunks = []

        for doc in results:
            retrieved_chunks.append({
                "chunk_text": doc.page_content,
                "metadata": doc.metadata
            })

        return retrieved_chunks


class AnswerAgent:
    """
    This agent uses retrieved document chunks and Gemini to generate
    a final answer.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")

        self.client = genai.Client(api_key=api_key)
        self.retriever = RetrieverAgent(top_k=3)

    def answer(self, question: str):
        retrieved_chunks = self.retriever.retrieve(question)

        context = "\n\n".join(
            [chunk["chunk_text"] for chunk in retrieved_chunks]
        )

        prompt = f"""
You are a document intelligence assistant.

Answer the user's question using ONLY the document context below.
If the answer is not present in the context, say:
"I could not find this information in the document."

Document Context:
{context}

User Question:
{question}

Final Answer:
"""

        try:
            response = self.client.models.generate_content(
                model="models/gemini-2.0-flash-lite",
                contents=prompt
            )

            final_answer = response.text

        except Exception:
            final_answer = (
                "Gemini API is currently unavailable or quota is exhausted. "
                "Below are the most relevant retrieved document chunks:\n\n"
                + context
            )

        return {
            "answer": final_answer,
            "source_chunks": retrieved_chunks
        }


class SummarizerAgent:
    """
    This agent summarizes the document.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")

        self.client = genai.Client(api_key=api_key)

    def summarize(self):
        vector_store = load_vector_store()

        docs = vector_store.similarity_search(
            query="main points summary important policy information",
            k=5
        )

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a document summarization agent.

Summarize the document context below in 5 clear bullet points.

Document Context:
{context}

Summary:
"""

        try:
            response = self.client.models.generate_content(
                model="models/gemini-2.0-flash-lite",
                contents=prompt
            )

            summary = response.text
            method = "gemini"

        except Exception:
            summary = self.local_extractive_summary(context)
            method = "local_fallback"

        return {
            "summary": summary,
            "method": method,
            "source_chunks": [
                {
                    "chunk_text": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }

    def local_extractive_summary(self, context: str):
        sentences = context.replace("\n", " ").split(".")

        important_keywords = [
            "policy",
            "data",
            "consent",
            "privacy",
            "security",
            "retention",
            "violation",
            "responsible",
            "encryption"
        ]

        selected_sentences = []

        for sentence in sentences:
            sentence_clean = sentence.strip()

            if not sentence_clean:
                continue

            lower_sentence = sentence_clean.lower()

            if any(keyword in lower_sentence for keyword in important_keywords):
                selected_sentences.append(sentence_clean)

        if not selected_sentences:
            selected_sentences = sentences[:3]

        bullet_points = []

        for sentence in selected_sentences[:5]:
            bullet_points.append(f"- {sentence.strip()}.")

        return "\n".join(bullet_points)


class ComplianceAgent:
    """
    This agent checks whether the document satisfies compliance requirements.

    Primary method: Gemini
    Fallback method: Local keyword-based validation
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")

        self.client = genai.Client(api_key=api_key)

    def check_compliance(self):
        vector_store = load_vector_store()

        docs = vector_store.similarity_search(
            query="privacy consent retention security encryption violation compliance policy",
            k=5
        )

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a compliance validation agent.

Check the document context below against this checklist:

1. User consent information
2. Data privacy policy
3. Data security or encryption clause
4. Data retention period
5. Violation or penalty clause

Document Context:
{context}

Return the answer in this format:
Passed Checks:
Failed Checks:
Missing Clauses:
Final Compliance Status:
"""

        try:
            response = self.client.models.generate_content(
                model="models/gemini-2.0-flash-lite",
                contents=prompt
            )

            report = response.text
            method = "gemini"

        except Exception:
            report = self.local_compliance_check(context)
            method = "local_fallback"

        return {
            "compliance_report": report,
            "method": method,
            "source_chunks": [
                {
                    "chunk_text": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }

    def local_compliance_check(self, context: str):
        lower_context = context.lower()

        checks = {
            "User consent information": ["consent"],
            "Data privacy policy": ["privacy", "data privacy"],
            "Data security or encryption clause": ["security", "securely", "encryption", "encrypted"],
            "Data retention period": ["retention", "2 years", "period"],
            "Violation or penalty clause": ["violation", "disciplinary", "penalty"]
        }

        passed_checks = []
        failed_checks = []

        for check_name, keywords in checks.items():
            if any(keyword in lower_context for keyword in keywords):
                passed_checks.append(check_name)
            else:
                failed_checks.append(check_name)

        if len(failed_checks) == 0:
            final_status = "Compliant"
        elif len(passed_checks) >= 3:
            final_status = "Partially Compliant"
        else:
            final_status = "Non-Compliant"

        report = "Passed Checks:\n"
        for item in passed_checks:
            report += f"- {item}\n"

        report += "\nFailed Checks:\n"
        if failed_checks:
            for item in failed_checks:
                report += f"- {item}\n"
        else:
            report += "- None\n"

        report += "\nMissing Clauses:\n"
        if failed_checks:
            for item in failed_checks:
                report += f"- {item}\n"
        else:
            report += "- None\n"

        report += f"\nFinal Compliance Status: {final_status}"

        return report