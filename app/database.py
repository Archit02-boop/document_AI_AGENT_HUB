import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """
    Creates a connection to PostgreSQL / Neon database.
    """

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL not found in .env file.")

    connection = psycopg2.connect(database_url)
    return connection


def create_tables():
    """
    Creates required database tables if they do not already exist.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            file_path TEXT NOT NULL,
            total_chunks INTEGER NOT NULL,
            total_vectors INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS compliance_reports (
            id SERIAL PRIMARY KEY,
            status TEXT NOT NULL,
            method TEXT NOT NULL,
            report TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    connection.commit()
    cursor.close()
    connection.close()


def save_document_metadata(file_path: str, total_chunks: int, total_vectors: int):
    """
    Saves document ingestion metadata into PostgreSQL.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO documents (file_path, total_chunks, total_vectors)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        (file_path, total_chunks, total_vectors)
    )

    document_id = cursor.fetchone()[0]

    connection.commit()
    cursor.close()
    connection.close()

    return document_id


def save_query_log(question: str, answer: str):
    """
    Saves user question and generated answer into PostgreSQL.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO query_logs (question, answer)
        VALUES (%s, %s)
        RETURNING id;
        """,
        (question, answer)
    )

    log_id = cursor.fetchone()[0]

    connection.commit()
    cursor.close()
    connection.close()

    return log_id


def save_compliance_report(status: str, method: str, report: str):
    """
    Saves compliance checking report into PostgreSQL.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO compliance_reports (status, method, report)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        (status, method, report)
    )

    report_id = cursor.fetchone()[0]

    connection.commit()
    cursor.close()
    connection.close()

    return report_id


def get_documents():
    """
    Returns all stored document metadata.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, file_path, total_chunks, total_vectors, created_at
        FROM documents
        ORDER BY created_at DESC;
    """)

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    documents = []

    for row in rows:
        documents.append({
            "id": row[0],
            "file_path": row[1],
            "total_chunks": row[2],
            "total_vectors": row[3],
            "created_at": str(row[4])
        })

    return documents