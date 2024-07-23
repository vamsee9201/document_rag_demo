#%%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS

#%%

#%%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",)
llm = OpenAI(model="gpt-3.5-turbo-instruct")

#%%
def get_vectorstore(pdf_content):
    documents = text_splitter.split_text(text=pdf_content)
    vectorstore = FAISS.from_texts(documents, embedding=embeddings)
    #vectorstore = None
    return vectorstore
#%%
def get_answer(question,pdf_content):
    answer = pdf_content + "\n" + question
    vectorstore = get_vectorstore(pdf_content=pdf_content)
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    docs = vectorstore.similarity_search(question)
    context = "context :"
    for doc in docs:
        context = context + doc.page_content + "\n"
    prompt = "answer the question below, if you don't find any answer in the context, reply you have no answer. "
    print(prompt)
    context2 = f"""context : {context} 
                question : {question}"""
    prompt_template = f""" {prompt} 
                        \n {context2}"""
    def answer_from_pdf(question):
        result = qa({"question":question, "chat_history":""})
        return result["answer"]
    answer = answer_from_pdf(question=question)
    return answer
#%%

