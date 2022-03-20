# import streamlit as st

# from haystack.utils import convert_files_to_dicts, fetch_archive_from_http, clean_wiki_text
# from haystack.nodes import Seq2SeqGenerator
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import DensePassageRetriever
# from haystack.pipelines import GenerativeQAPipeline

# document_store = FAISSDocumentStore.load("haystack_got_faiss_1")

# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.title("Question Answering Webapp")
# st.text("What would you like to know today?")

# #@st.cache(allow_output_mutation=True)
# with st.spinner ('Loading Model into Memory....'):
#     retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki", passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",)
#     generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
#     pipe = GenerativeQAPipeline(generator, retriever)  

# text = st.text_input('Enter your questions here....') # no input required
# if text:
#     st.write("Response:")
#     with st.spinner('Searching for answers....'):
#         prediction = pipe.run(query=text, params={"Retriever": {"top_k": 3}})
#         st.write('answer: {}'.format(prediction[0]))
# #        st.write('title: {}'.format(prediction[1]))
# #        st.write('paragraph: {}'.format(prediction[2]))
#     st.write("")


# import streamlit as st

# from haystack.utils import convert_files_to_dicts, fetch_archive_from_http, clean_wiki_text
# #from haystack.nodes import Seq2SeqGenerator
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import DensePassageRetriever
# #from haystack.pipelines import GenerativeQAPipeline

# from haystack.utils import print_documents
# from haystack.pipelines import DocumentSearchPipeline

# #document_store = FAISSDocumentStore.load("haystack_got_faiss_1")

# document_store = FAISSDocumentStore(embedding_dim=128, faiss_index_factory_str="Flat")
# #document_store = FAISSDocumentStore.load(index_path="haystack_got_faiss", config_path="haystack_got_faiss_config")
# #document_store = FAISSDocumentStore.load("haystack_got_faiss_1")

# # Let's first get some files that we want to use
# doc_dir = "data/article_txt_got"
# s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
# fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# # Convert files to dicts
# dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# # Now, let's write the dicts containing documents to our DB.
# document_store.write_documents(dicts)



# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.title("Question Answering Webapp")
# st.text("What would you like to know today?")

# #@st.cache(allow_output_mutation=True)
# with st.spinner ('Loading Model into Memory....'):
#     retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki", passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",)
#     p_retrieval = DocumentSearchPipeline(retriever)
    
# text = st.text_input('Enter your questions here....') # no input required
# res = p_retrieval.run(query=text, params={"Retriever": {"top_k": 1}})

# if text:
#     st.write("Response:")
#     with st.spinner('Searching for answers....'):
#         prediction = print_documents(res, max_text_len=512)
#         st.write('answer: {}'.format(prediction))
# #        st.write('title: {}'.format(prediction[1]))
# #        st.write('paragraph: {}'.format(prediction[2]))
#     st.write("")    
    

    
import streamlit as st

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Welcome to our startup advisory!')
    st.text("On this page we will answer questions regarding your startups based on what we learnt from YC's Michael and Dalton")


    
with dataset:
    st.header("YC's YouTube videos")
    st.text("I found these videos very informative")


    
    
with features:
    st.header("The features I created")

    
    
with model_training:
    st.header("Time to train the model!")
    st.text("Here we'll play with hyperparamets")
    
    
    
    
    
    

