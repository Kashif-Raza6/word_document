# Import required libraries and modules
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant, Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain import VectorDBQA, OpenAI, Cohere
from langchain.chains import RetrievalQA
import os
from PIL import Image


# add the banner image
image = Image.open('Images/banner.jpg', 'r')

# Set the app title and header
#st.set_page_config(page_title="Chat with your Word Documents", page_icon=":speech_balloon:")
#st.title("Chat with your Word Documents")

# show the banner image
st.image(image, use_column_width=True)

st.markdown(
    """
    <footer style='text-align: center; padding-top: 30px;'>
        Created by Kashif Raza
    </footer>
    """,
    unsafe_allow_html=True,
)



# Function to perform QA using the given code and display the result
@st.cache_data(show_spinner=True)
def perform_qa(question):
    # Load and process documents
    
    loader = DirectoryLoader("data", glob="**/*.docx")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    API = st.secrets["API"]
    #os.environ["OPENAI_API_KEY"] = API
    embeddings = OpenAIEmbeddings(openai_api_key=API)
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=API, temp=0),
                                     chain_type="map_reduce", vectorstore=docsearch)
    ans = qa.run(question)
    return ans



# Add text input and submit button to get user's question
question = st.text_input("You:", value="", max_chars=None, key=None, type="default")
if st.button("Ask"):
    st.write("Loading and processing documents...")
    if question:
        # Call the function to perform QA
        st.write("Searching for answer...")
        answer = perform_qa(question)
        if answer:
            st.success("Answer: " + answer)
        else:
            st.error("Answer: No answer found")
    else:
        st.write("Please enter a question")

# Set page footer
footer = """
<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p> Example Queries </p>
<p>1. what is the primary religion practiced by the Yljiri?</p>
<p>2. Can you list the Big 6 companies that make up the Corpaco Collective with a brief description of each?</p>

<p>*********************************</p>
<p>Made by ❤️: Kashif Raza</p>
<p>View the code on <a href="https://github.com/">GitHub</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

