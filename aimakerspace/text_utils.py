import os
from typing import List
import PyPDF2

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )
    

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents

class PdFileLoader:
    """
    Class to load and extract text from PDF files. Supports loading from a single PDF file or all PDF files within a directory.
    """

    def __init__(self, path: str):
        self._documents = []
        self.path = path

    def load_pdf(self):
        """
        Load text from a single PDF file specified by the path.
        If the path is a directory, it will call the load_directory method.
        """
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            self._load_single_pdf(self.path)
        else:
            raise ValueError("Provided path is neither a valid directory nor a valid PDF file")

    def _load_single_pdf(self, file_path):
        """
        Helper method to load and extract text from a single PDF file.
        """
        with open(file_path, "rb") as file:
            try:
                reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for p in range(len(reader.pages)):
                    page = reader.pages[p]
                    text_parts.append(page.extract_text())
                
                document_text = "\n".join(text_parts) if text_parts else "<empty file>"
                self._documents.append(document_text)
            
            except PyPDF2.errors.PdfReadError as e:
                raise ValueError(f"Error reading the PDF file: {e}")
            except Exception as e:
                raise ValueError(f"An unexpected error occurred: {e}")

    def load_directory(self):
        """
        Load and extract text from all PDF files within the specified directory.
        """
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    self._load_single_pdf(file_path)

    def load_documents(self):
        """
        Load documents from the specified path (either a file or a directory) and return the extracted text.
        """
        self.load_pdf()
        return self._documents





class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    # loader = PdFileLoader("data/agent_paper.pdf")
    # loader.load_pdf()
    # splitter = CharacterTextSplitter()
    # chunks = splitter.split_texts(loader.documents)
    # print(len(chunks))
    # print(chunks[0])
    # print("--------")
    # print(chunks[1])
    # print("--------")
    # print(chunks[-2])
    # print("--------")
    # print(chunks[-1])
    print("test")
