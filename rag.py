# %%
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader
# %%

openai_api_key="EMPTY"
openai_api_base="http://192.168.0.67:1234/v1/"
llm = ChatOpenAI(openai_api_key=openai_api_key,openai_api_base=openai_api_base)

embeddings_path="/data/lr/agent/models/bge-large-en-v1.5"
embeddings=HuggingFaceBgeEmbeddings(model_name=embeddings_path)

# %%
template= """
只根据一下文档回答问题：
{context}
问题：{question}
"""
prompt= ChatPromptTemplate.from_template(template)

vectorstore= FAISS.from_texts(["今天是星期日","今天需要上班"],embedding=embeddings)

retriver=vectorstore.as_retriever()

retriver.invoke("今天星期几")
# %%

outputParse=StrOutputParser()
setup_and_retrieval=RunnableParallel(
    {"context":retriver,"question":RunnablePassthrough()}
)

chain= setup_and_retrieval | prompt | llm | outputParse

# %%
chain.invoke("今天星期几")
# %%
print(chain)
loader=DirectoryLoader("./data/video")
docs=loader.load()

text_splitter= CharacterTextSplitter( chunk_size=10000, chunk_overlap=0)
docs_split= text_splitter.split_documents(docs)

vectorStoreDB= FAISS.from_documents(docs_split,embedding=embeddings)
# %%
retriver=vectorStoreDB.as_retriever(search_type="mmr",search_kwargs={"k":1})

retriver.get_relevant_documents("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models ")

print(retriver)

outputParse= StrOutputParser()

setup_and_retrieval=RunnableParallel({
    "context":retriver,
    "question": RunnablePassthrough()
})
# %%
chain= setup_and_retrieval | prompt | llm | outputParse
print(chain.invoke("please provide some information about the chain-of-thought technology"))

# %%

loader=PyPDFLoader("/data/lr/agent/data/video/2201.11903v6.pdf")
pages= loader.load()
# %%
docs=""

for item in pages:
    docs+=item.page_content

docs

# %%

template=" {context} 请总结以上论文内容 "
prompt=ChatPromptTemplate.from_template(template)
chain= prompt | llm | outputParse

chain.invoke({"context":docs[:4096]})
    
# %%
 