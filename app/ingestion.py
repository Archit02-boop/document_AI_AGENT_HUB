def load_text_file(file_path: str) -> str:
    """
    This function reads a text file and returns its content.
    """

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return text

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    except Exception as e:
        raise Exception(f"Error while reading file: {str(e)}")


def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50):
    """
    This function splits large text into smaller overlapping chunks.

    chunk_size = maximum number of characters in each chunk
    overlap = number of repeated characters between chunks
    """

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks