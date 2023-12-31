import streamlit as st
import pickle
import os
import nltk
from nltk.tokenize import sent_tokenize
import tensorflow_hub as hub
from scipy.spatial import distance
from featureVector import compute_sentence_score,calculateLongestSent

#FOR T5
from transformers import AutoTokenizer, AutoModelWithLMHead


nltk.download('punkt')
    

def runFVec_model(articles_sent_tokenized,title):
    # Create a list of tuples (sentence, f_score)
    sentence_f_scores = []
    text = " ".join(articles_sent_tokenized)
    weights = {
      'F1': 0.2,  # Title feature
      'F2': 0.1,  # Sentence length
      'F3': 0.3,  # Sentence position
      'F5': 0.2,  # Term weight
      'F6': 0.1,  # Proper noun
      'F7': 0.1   # Numerical data
    }

    for pos in range(1, len(articles_sent_tokenized)):
        sentence = articles_sent_tokenized[pos]
        f_score = compute_sentence_score(sentence=sentence, entire_text=text, title=title,
                                         weights=weights, longest_sentence=calculateLongestSent(articles_sent_tokenized), position=pos,
                                         total_sentences=len(articles_sent_tokenized))

        sentence_f_scores.append((f_score,sentence))

    # Sort sentences by F score in ascending order
    sorted_sentences = sorted(sentence_f_scores, key=lambda x: x[1])
    sorted_sentences = sorted_sentences[-3:]
    summary = " ".join([sublist[1] for sublist in sorted_sentences])

    #Return top sentences
    return summary

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
    return summary_sentences,summary

@st.cache_resource
def getT5():
        st.write("Loading T5....")
        model=AutoModelWithLMHead.from_pretrained('T5-small', return_dict=True)
        return model

@st.cache_resource
def getTFHub():
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
    return model


def get_documents():
    corpus = []
    filenames = []

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

def get_summaries():
    corpus = []
    filenames = []

    corpus_dir = 'Summaries'

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
_, _, summaryMappings = get_summaries()

# Dropdown to select document
selected_doc = st.selectbox('Select Document', filemappings.keys())

models = ["Feature Vector","Word2Vec","TFHub","T5"]

# Dropdown to select model
selected_model = st.selectbox('Select Model', models)

# Function to summarize and highlight
def summarize_and_highlight(text,model,reference_summary):
    title = text[0]
    sentences = " ".join(text[1:])  
    articles_sent_tokenized = sent_tokenize(sentences)
    if model == 'TFHub':
        st.write("Getting top sentences from TFHub")
        top_sentences,summary= run_tfhub_model(getmodel(model),articles_sent_tokenized,title)
        render(title,summary,articles_sent_tokenized,reference_summary)
    if model == 'Word2Vec':
        st.write("Getting top sentences from Word2Vec")
        top_sentences,summary = run_Word2Vec_model(getmodel(model),articles_sent_tokenized,title)
        render(title,summary,articles_sent_tokenized,reference_summary)
    if model == 'T5':
        st.write("Getting top sentences from T5")
        summary = run_t5_model(getmodel(model),articles_sent_tokenized,title)
        renderForT5(title,summary,articles_sent_tokenized,reference_summary)
    if model == 'Feature Vector':
        st.write("Getting top sentences from Feature Vector")   
        summary = runFVec_model(articles_sent_tokenized,title)
        render(title,summary,articles_sent_tokenized,reference_summary)


def render(title,summary,articles_sent_tokenized,affiliate_summary):
    st.markdown("#### Top 3 sentences by model for article with title:"+ title)
    st.markdown("---")

    for sentence in summary.split("."):
        st.write(sentence)
    st.markdown("---")

    st.markdown("#### Comparsion With Affiliated Summary")
    for sentence in affiliate_summary.split("."):
        if sentence in summary:
            highlight_text(sentence)
        else:
            st.write(sentence)

def renderForT5(title,summary,articles_sent_tokenized,reference_summary):

    st.markdown("---")
    st.write(str(summary).capitalize())
    st.markdown("---")

    st.markdown("#### T5 does not pick top sentences directly, but summarizes the article using natural language.")
    st.markdown("#### So, while we won't see any highlighted sentences for T5, we are printing the T5 summary above. Affiliated summary is below.")

    st.write(title)
    st.write(reference_summary)

def highlight_text(text, color='yellow'):
    highlighted_text = f'<mark style="background-color: {color};">{text}</mark>'
    st.markdown(highlighted_text, unsafe_allow_html=True)


# Button to trigger the process
if st.button('Process'):
    with st.spinner('Summarizing...'):
        document_text = filemappings[selected_doc]  # Fetch text from the selected document
        reference_summary = summaryMappings[selected_doc]
        summarize_and_highlight(document_text,selected_model," ".join(reference_summary))  # Summarize and highlight