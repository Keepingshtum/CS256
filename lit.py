import streamlit as st
import pickle
import os
import nltk
from nltk.tokenize import sent_tokenize
from scipy.spatial import distance
import tensorflow_hub as hub

nltk.download('punkt')

def run_model(model,articles_sent_tokenized,title):
    sentences_score = []

    #embedd title in the model
    model.train([title.lower().split()], total_examples=1, epochs=1)

    for sentence in articles_sent_tokenized:
        distance = model.wv.n_similarity(sentence.lower().split(), title.lower().split())
        sentences_score.append((distance, sentence))

    top_sentences = sorted(sentences_score)
    return top_sentences[-3:]

def run_tfhub_model(model,articles_sent_tokenized,title):
    # Embed sentences using the Universal Sentence Encoder
    sentence_embeddings = model(articles_sent_tokenized)
    similarities = [1 - distance.cosine(title.lower().split(), sentence_embedding) for sentence_embedding in sentence_embeddings]
    top_sentences_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]  # Get top 3 indices
    return [articles_sent_tokenized[i] for i in top_sentences_indices]

def getmodel(selectedmodel):
    # model = pickle.load(open('word2vec_model.pkl','rb'))
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    return model

# FOR WORD2VEC
# def get_top_sentences(model2, filename):
#   sentences_score = []


#   for s1 in articles_sent_tokenized[idx]:
#       distance = model2.wv.n_similarity(s1.lower().split(), reference.lower().split())
#       sentences_score.append((distance, s1))

#   top_sentences = sorted(sentences_score)
#   return top_sentences

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

models = ['Feature Vector',"Word2Vec","TFHub","T5"]

# Dropdown to select model
selected_model = st.selectbox('Select Model', models)

# Function to summarize and highlight
def summarize_and_highlight(text,model):
    # TODO: Pick title 
    title = text[0]
    sentences = " ".join(text[1:])
    articles_sent_tokenized = sent_tokenize(sentences)
    # top_sentences = run_model(model,articles_sent_tokenized,title)
    top_sentences = run_tfhub_model(model,articles_sent_tokenized,title)
    st.write(title)
    topG = " ".join([sublist[1] for sublist in top_sentences])
    for sentence in articles_sent_tokenized:
        if sentence in topG:
            highlight_text(sentence)
        else:
            st.write(sentence)


def highlight_text(text, color='yellow'):
    highlighted_text = f'<mark style="background-color: {color};">{text}</mark>'
    st.markdown(highlighted_text, unsafe_allow_html=True)


# Button to trigger the process
if st.button('Process'):
    with st.spinner('Summarizing...'):
        document_text = filemappings[selected_doc]  # Fetch text from the selected document
        summarize_and_highlight(document_text,getmodel(selected_model))  # Summarize and highlight