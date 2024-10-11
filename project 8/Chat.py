import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import google.generativeai as genai
#HUGGINGFACEHUB_API_TOKEN=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
load_dotenv()
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text(pdfs):
    text = ""
    # Read the PDF
    for pdf in pdfs:
        pdfreader = PdfReader(pdf)
        for page in pdfreader.pages:
            text += page.extract_text()
    return text 
def get_text_chunks(text):           

    # Split pages into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks
def get_vectore_store(chunks):    
    # Convert chunks into knowledge base
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    knowledge_base.save_local('faiss.index')

    return knowledge_base 


def get_conversation():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    chain=load_qa_chain(llm=llm,chain_type="stuff",prompt=prompt)
    return chain

def user_query(query):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    knowledgebase=FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True) 

    docs=knowledgebase.similarity_search(query=query) 
    chain=get_conversation()

    response=chain({"input_documents":docs,"question":query},return_only_outputs=True)

    st.write("Answer:",response["output_text"])

def main():
    st.set_page_config('Chat With Pdf')
    st.title('Welcome to Chat PDF App📑')

    query=st.text_input("Ask your question!")

    if query:
        user_query(query)

    with st.sidebar:    
         st.title("Menu:")
         pdfs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",accept_multiple_files=True)
         
         if st.button("submmit & process"):   
            with st.spinner("Processing..!"):
                raw_text=get_text(pdfs)
                chunks=get_text_chunks(raw_text)
                get_vectore_store(chunks)
                st.success("Done..!")
                


    

if __name__ == "__main__":
    main()







# import streamlit as st
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from langchain.text_splitter import CharacterTextSplitter
# import langchain
# import os
# from langchain.chains.question_answering import load_qa_chain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
# import google.generativeai as genai
# # from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# # from langchain.llms.huggingface_hub import HuggingFaceHub
# from langchain import FAISS
# #from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings

# load_dotenv()
# #HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)
# #os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_correct_token_here"


# def get_text(text_path):
#     # Read the PDF
#     pdf = PdfReader(text_path)

#     # Store PDF pages' text
#     text = ""
#     for page in pdf.pages:
#         text += page.extract_text()

#     #split pages into chunks
#     text_splitter=CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len)
#     chunks=text_splitter.split_text(text)

#     #convert chunks into knowledge base
#     model_name = "models/embedding-001"
#     # model_kwargs = {'device': 'cpu'}
#     # encode_kwargs = {'normalize_embeddings': False}
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model_name=model_name, 
#     )
   
#     knowledge_base=FAISS.from_texts(chunks,embeddings)
#     knowledge_base.save_local('faiss.index')

#     return knowledge_base    

# def main():
#     st.set_page_config('Chat With Pdf')
#     st.title('Welcome to Chat PDF App')

#     pdf= st.file_uploader("upload your pdf", type="pdf")

#     if pdf is not None:
#         knowledgebase=get_text(pdf)
#         #st.write(knowledgebase)

#         cancel_button=st.button('Cancel')
#         query=st.text_input('Ask Your Question....!')
         
#         if cancel_button:
#             st.stop()
#         if query:
#           embeddings = GoogleGenerativeAIEmbeddings(
#           model_name= "models/embedding-001")
#           knowledgebase=FAISS.load_local('faiss.index',embeddings)
#           docs= knowledgebase.similarity_search(query)

#           llm=ChatGoogleGenerativeAI(model="gemini-pro")    
#           chain=load_qa_chain(llm,chain_type="stuff")
#           response=chain.invoke(input={"question":query,"input_documents":docs})
#           st.write("the answer: ",response["output_text"])




# if __name__ == "__main__":
#     main()
