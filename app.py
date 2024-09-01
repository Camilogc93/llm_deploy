import os
import numpy as np
from typing import List
from chainlit.types import AskFileResponse
from aimakerspace.text_utils import CharacterTextSplitter
from aimakerspace.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import chainlit as cl
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from PyPDF2 import PdfReader  # Use PyPDF2 for extracting text from PDFs

# Global variable for Qdrant client
client = None

# System Chat Prompts
system_template = "Use the following context to answer a user's question. If you cannot find the answer in the context, say you don't know the answer."
system_role_prompt = SystemRolePrompt(system_template)

user_prompt_template = "Context:\n{context}\nQuestion:\n{question}"
user_role_prompt = UserRolePrompt(user_prompt_template)

class QdrantRetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.embedding_model = EmbeddingModel()
        self.vector_db_retriever = None

    async def initialize_qdrant(self, texts: List[str]):
        global client
        client = QdrantClient(":memory:")
        if not client.collection_exists("my_collection"):
            client.create_collection(
                collection_name="my_collection",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
        embeddings = await self.embedding_model.async_get_embeddings(texts)
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            self._insert_into_qdrant(text, np.array(embedding), idx)
        
        self.vector_db_retriever = client

    def _insert_into_qdrant(self, text: str, vector: np.array, idx: int):
        point = PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"text": text}
        )
        client.upsert(
            collection_name="my_collection",
            points=[point]
        )

    async def arun_pipeline(self, user_query: str, k: int = 4):
        query_vector = self.embedding_model.get_embedding(user_query)
        context_list = self.vector_db_retriever.search(
            collection_name="my_collection",
            query_vector=query_vector, 
            limit=k
        )
        context_prompt = "\n".join(context.payload['text'] for context in context_list)

        formatted_system_prompt = system_role_prompt.create_message()
        formatted_user_prompt = user_role_prompt.create_message(question=user_query, context=context_prompt)

        async def generate_response():
            async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
                yield chunk

        return {"response": generate_response(), "context": context_list}

# Utility functions
text_splitter = CharacterTextSplitter()

def process_text_file(file: AskFileResponse) -> List[str]:
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file_path = temp_file.name

    with open(temp_file_path, "wb") as f:
        f.write(file.content)

    with open(temp_file_path, "r") as f:
        documents = [f.read()]
    texts = text_splitter.split_texts(documents)
    return texts

def process_pdf_file(file: AskFileResponse) -> List[str]:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name

    with open(temp_file_path, "wb") as f:
        f.write(file.content)

    # Extract text from the PDF
    reader = PdfReader(temp_file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    documents = [text]
    texts = text_splitter.split_texts(documents)
    return texts

# Chainlit event handlers
@cl.on_chat_start
async def on_chat_start():
    global client
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a Text or PDF file to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=2,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    if file.name.endswith('.pdf'):
        texts = process_pdf_file(file)
    else:
        texts = process_text_file(file)

    print(f"Processing {len(texts)} text chunks")
    
    chat_openai = ChatOpenAI()

    # Create the Qdrant pipeline and initialize the vector database with the provided texts
    qdrant_pipeline = QdrantRetrievalAugmentedQAPipeline(
        llm=chat_openai,
    )
    await qdrant_pipeline.initialize_qdrant(texts)

    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", qdrant_pipeline)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content)

    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()
