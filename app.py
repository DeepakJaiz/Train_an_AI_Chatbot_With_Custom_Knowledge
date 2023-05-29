from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper,StorageContext, load_index_from_storage
from langchain import OpenAI
import gradio as gr
import sys
import os
os.environ["OPENAI_API_KEY"] = "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"

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit) 
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs)) 
    #storage_context = StorageContext.from_defaults(chunk_size_limit=512)
    documents = SimpleDirectoryReader(directory_path).load_data() 
    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper) 
    index.storage_context.persist(persist_dir="./index.json")
    return index


def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir="./index.json")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response

iface = gr.Interface(fn=chatbot,
inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
outputs="text",
title="Custom-trained AI Chatbot")
index = construct_index("documents")
iface.launch(share=True)