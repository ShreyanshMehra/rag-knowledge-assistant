import os
from dotenv import load_dotenv, find_dotenv
import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

_ = load_dotenv(find_dotenv())

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question using only the following context:
{context}
-------------------------------------------------------------
Answer this question based on the context above: {question}
"""

def chatbot_response():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The text to search for")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve more chunks with a lower threshold for completeness
    results = db.similarity_search_with_relevance_scores(query_text, k=10)  # k increased to 10

    # Filter results by a lower threshold (0.2)
    filtered = [(doc, score) for doc, score in results if score >= 0.2]
    if not filtered:
        print(f"Unable to find matching results for {query_text}")
        return

    # Concatenate context, keeping within LLM's input size (truncate if too long)
    max_context_chars = 2000  # Adjust for your LLM's context window
    context = ""
    for doc, score in filtered:
        if len(context) + len(doc.page_content) > max_context_chars:
            break
        context += doc.page_content + "\n\n---\n\n"

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context.strip(), question=query_text)
    print(prompt)

    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        device=0
    )
    model = HuggingFacePipeline(pipeline=hf_pipeline)
    response_text = model(prompt)

    sources = [doc.metadata.get("source", None) for doc, score in filtered]
    formatted_response = f"{response_text}\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    chatbot_response()
