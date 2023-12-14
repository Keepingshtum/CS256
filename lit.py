import streamlit as st
import pickle
import os
import nltk
from nltk.tokenize import sent_tokenize
import tensorflow_hub as hub
from scipy.spatial import distance

#FOR T5
from transformers import AutoTokenizer, AutoModelWithLMHead


nltk.download('punkt')
    

def run_Word2Vec_model(model,articles_sent_tokenized,title):
    sentences_score = []
    model.train([title.lower().split()], total_examples=1, epochs=1)

    for sentence in articles_sent_tokenized:
        distance = model.wv.n_similarity(sentence.lower().split(), title.lower().split())
        sentences_score.append((distance, sentence))

    top_sentences = sorted(sentences_score)[-3:]
    summary = " ".join([sublist[1] for sublist in top_sentences])
    return top_sentences,summary

def run_t5_model(model,articles_sent_tokenized,title):
    sentences_score = []
    tokenizer=AutoTokenizer.from_pretrained('T5-base')
    inputs = tokenizer.encode("summarize: " + " ".join(articles_sent_tokenized), return_tensors='pt', max_length=512, truncation=True)
    output = model.generate(inputs, min_length=80, max_length=100, num_return_sequences=1)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary


def run_tfhub_model(model,articles_sent_tokenized,title):
    # Embed sentences using the Universal Sentence Encoder
    sentence_embeddings = model(articles_sent_tokenized)
    title_embedding = model([title])
    similarities = [1 - distance.cosine(title_embedding[0], sentence_embedding) for sentence_embedding in sentence_embeddings]
    top_sentences_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]  # Get top 3 indices
    summary_sentences = [articles_sent_tokenized[i] for i in top_sentences_indices]
    summary = " ".join(summary_sentences)
    print("Summary is :")
    print(summary)
    return summary_sentences,summary

@st.cache_resource
def getTFHub():
        st.write("Loading T5....")
        model=AutoModelWithLMHead.from_pretrained('T5-small', return_dict=True)
        return model

@st.cache_resource
def getT5():
        st.write("Loading TFhub....")
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        return model

def getmodel(selectedmodel):
    if selectedmodel == 'TFHub':
        model = getTFHub()
    elif selectedmodel == 'Word2Vec':
        st.write("Loading Word2Vec....")
        model = pickle.load(open('word2vec_model.pkl','rb'))
    elif selectedmodel == 'T5':
        model = getT5()
    # else: #Default to Word2Vec For Now 
    #     st.write("Loading Word2Vec....")
    #     model = pickle.load(open('word2vec_model.pkl','rb'))
    return model


def get_documents():
    corpus = []
    filenames = []

    # DO NOT CHANGE THIS PATH AS PROF has asked us to use this in the template.
    corpus_dir = 'politics'

    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(corpus_dir, filename)
            filenames.append(filename)
            with open(file_path, mode='rt', encoding='utf-8') as fp:
                lines = fp.read().splitlines()
                corpus.append([i for i in lines if i])

    # Map filenames to corpus elements
    file_corpus_mapping = {f"{i + 1:03d}.txt": corpus[i] for i in range(len(corpus))}

    return corpus, filenames, file_corpus_mapping

# Streamlit app setup
st.title('Document Summarizer')
_, _, filemappings = get_documents()  

# Dropdown to select document
selected_doc = st.selectbox('Select Document', filemappings.keys())

models = ["Feature Vector","Word2Vec","TFHub","T5"]

# Dropdown to select model
selected_model = st.selectbox('Select Model', models)
st.write(selected_model)

# Function to summarize and highlight
def summarize_and_highlight(text,model):
    # TODO: Pick title 
    title = text[0]
    sentences = " ".join(text[1:])  
    articles_sent_tokenized = sent_tokenize(sentences)
    if model == 'TFHub':
        st.write("Getting top sentences from TFHub")
        top_sentences,summary_from_TFHUB = run_tfhub_model(getmodel(model),articles_sent_tokenized,title)
    if model == 'Word2Vec':
        st.write("Getting top sentences from Word2Vec")
        top_sentences,summary_from_TFHUB = run_Word2Vec_model(getmodel(model),articles_sent_tokenized,title)
    if model == 'T5':
        st.write("Getting top sentences from T5")
        summary_from_TFHUB = run_t5_model(getmodel(model),articles_sent_tokenized,title)
        # st.write("Got from T5 : ",summary_from_TFHUB)
    # else: #Default to Word2Vec for now
    #     st.write("Getting top sentences from else")
    #     top_sentences,summary_from_TFHUB = run_Word2Vec_model(getmodel(model),articles_sent_tokenized,title)

    st.write(title)
    topG = summary_from_TFHUB
    for sentence in articles_sent_tokenized:
        if sentence in topG:
            highlight_text(sentence)
        else:
            st.write(sentence)
    st.write("Printing here the sumamry")
    st.write(summary_from_TFHUB)


def highlight_text(text, color='yellow'):
    highlighted_text = f'<mark style="background-color: {color};">{text}</mark>'
    st.markdown(highlighted_text, unsafe_allow_html=True)


# Button to trigger the process
if st.button('Process'):
    with st.spinner('Summarizing...'):
        document_text = filemappings[selected_doc]  # Fetch text from the selected document
        summarize_and_highlight(document_text,selected_model)  # Summarize and highlight