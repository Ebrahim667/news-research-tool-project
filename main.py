import os
import streamlit as st
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv


load_dotenv()  # load environment variables from .env (especially GROQ_API_KEY)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_path = "faiss_index"  # folder name, not a pickle file

main_placeholder = st.empty()

llm = ChatGroq(temperature=0.9, max_tokens=500, model="llama-3.3-70b-versatile")


# Helper so we use the SAME embedding model when saving and when loading.
# FAISS.load_local needs this object to interpret the saved vectors correctly.
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if process_url_clicked:
    # ─────────────────────────────────────────────────────────────
    # input validation — drop empty URL fields and stop
    # the run cleanly if the user clicked "Process" with nothing.
    # ─────────────────────────────────────────────────────────────
    valid_urls = [url.strip() for url in urls if url.strip()]

    if not valid_urls:
        st.error("Please enter at least one URL before processing.")
        st.stop()

    # ─────────────────────────────────────────────────────────────
    # error handling around URL loading.
    # If a URL is broken, unreachable, or returns garbage, we now
    # show a friendly message instead of crashing the whole app.
    # ─────────────────────────────────────────────────────────────
    try:
        loader = UnstructuredURLLoader(urls=valid_urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        if not data:
            st.error(
                "No content could be loaded from the provided URLs. "
                "Please check the URLs and try again."
            )
            st.stop()
    except Exception as e:
        st.error(f"Failed to load URLs: {e}")
        st.stop()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # ─────────────────────────────────────────────────────────────
    # error handling around embedding + index creation.
    # save using FAISS's native save_local() method
    # instead of pickle.dump(). Saves into a folder named
    # "faiss_index" containing index.faiss + index.pkl.
    # ─────────────────────────────────────────────────────────────
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        vectorstore.save_local(index_path)
    except Exception as e:
        st.error(f"Failed to build the vector store: {e}")
        st.stop()


query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(index_path):
        # ─────────────────────────────────────────────────────────
        # error handling around loading + answering.
        # load using FAISS.load_local() instead of
        # pickle.load(). The allow_dangerous_deserialization flag
        # is required because the small metadata file is still
        # pickled — we explicitly accept that residual risk.
        # ─────────────────────────────────────────────────────────
        try:
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=vectorstore.as_retriever()
            )
            result = chain.invoke({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)
        except Exception as e:
            st.error(f"Something went wrong while answering: {e}")
    else:
        st.info("No processed URLs yet. Please process some URLs first using the sidebar.")