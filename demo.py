import streamlit as st

from haystack.utils import convert_files_to_dicts, fetch_archive_from_http, clean_wiki_text
from haystack.nodes import Seq2SeqGenerator
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.pipelines import GenerativeQAPipeline

document_store = FAISSDocumentStore.load("haystack_got_faiss_1")

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Question Answering Webapp")
st.text("What would you like to know today?")

@st.cache(allow_output_mutation=True)

with st.spinner ('Loading Model into Memory....'):
    #retreiver = DensePassageRetriever.load(load_dir='/content/drive/My Drive/kiku/got_retreiver', document_store=document_store)
    retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki", passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",)
    generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
    pipe = GenerativeQAPipeline(generator, retriever)  

text = st.text_input('Enter your questions here....') # no input required
if text:
    st.write("Response:")
    with st.spinner('Searching for answers....'):
        prediction = pipe.run(query=text, params={"Retriever": {"top_k": 2}})
        st.write('answer: {}'.format(prediction[0]))
#        st.write('title: {}'.format(prediction[1]))
#        st.write('paragraph: {}'.format(prediction[2]))
    st.write("")
