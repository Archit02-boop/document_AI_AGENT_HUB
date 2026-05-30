from fastapi import FastAPI
from pydantic import BaseModel

from app.ingestion import load_text_file, split_text_into_chunks
from app.vector_store import create_vector_store
from app.agents import RetrieverAgent, AnswerAgent, SummarizerAgent, ComplianceAgent
from app.database import (
    create_tables,
    save_document_metadata,
    save_query_log,
    save_compliance_report,
    get_documents
)

app = FastAPI(
    title="Multi-Agent Document Intelligence Platform",
    description="A platform for document ingestion, retrieval, summarization, and compliance checking.",
    version="1.0.0"
)


class IngestRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    create_tables()


@app.get("/")
def home():
    return {
        "message": "Multi-Agent Document Intelligence Platform is running successfully."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/ingest")
def ingest_document(request: IngestRequest):
    try:
        text = load_text_file(request.file_path)

        chunks = split_text_into_chunks(
            text=text,
            chunk_size=300,
            overlap=50
        )

        total_vectors = create_vector_store(chunks)

        document_id = save_document_metadata(
            file_path=request.file_path,
            total_chunks=len(chunks),
            total_vectors=total_vectors
        )

        return {
            "message": "Document loaded, chunked, embedded, stored in FAISS, and metadata saved in PostgreSQL.",
            "document_id": document_id,
            "file_path": request.file_path,
            "characters": len(text),
            "total_chunks": len(chunks),
            "total_vectors": total_vectors
        }

    except Exception as e:
        return {
            "message": "Error during ingestion.",
            "error": str(e)
        }


@app.post("/retrieve")
def retrieve_relevant_chunks(request: QueryRequest):
    try:
        retriever = RetrieverAgent(top_k=3)

        chunks = retriever.retrieve(request.question)

        return {
            "question": request.question,
            "retrieved_chunks": chunks
        }

    except Exception as e:
        return {
            "message": "Error during retrieval.",
            "error": str(e)
        }


@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        answer_agent = AnswerAgent()

        result = answer_agent.answer(request.question)

        log_id = save_query_log(
            question=request.question,
            answer=result["answer"]
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "query_log_id": log_id,
            "source_chunks": result["source_chunks"]
        }

    except Exception as e:
        return {
            "message": "Error while generating answer.",
            "error": str(e)
        }


@app.get("/summarize")
def summarize_document():
    try:
        summarizer = SummarizerAgent()

        result = summarizer.summarize()

        return {
            "summary": result["summary"],
            "method": result["method"],
            "source_chunks": result["source_chunks"]
        }

    except Exception as e:
        return {
            "message": "Error while summarizing document.",
            "error": str(e)
        }


@app.get("/compliance")
def check_compliance():
    try:
        compliance_agent = ComplianceAgent()

        result = compliance_agent.check_compliance()

        report_text = result["compliance_report"]

        if "Final Compliance Status: Compliant" in report_text:
            status = "Compliant"
        elif "Final Compliance Status: Partially Compliant" in report_text:
            status = "Partially Compliant"
        else:
            status = "Non-Compliant"

        report_id = save_compliance_report(
            status=status,
            method=result["method"],
            report=report_text
        )

        return {
            "compliance_report_id": report_id,
            "compliance_status": status,
            "compliance_report": result["compliance_report"],
            "method": result["method"],
            "source_chunks": result["source_chunks"]
        }

    except Exception as e:
        return {
            "message": "Error while checking compliance.",
            "error": str(e)
        }


@app.get("/documents")
def list_documents():
    try:
        documents = get_documents()

        return {
            "documents": documents
        }

    except Exception as e:
        return {
            "message": "Error while fetching documents.",
            "error": str(e)
        }