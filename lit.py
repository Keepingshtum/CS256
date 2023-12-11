import streamlit as st
import pickle
import os

model = pickle.load(open('word2vec_model.pkl','rb'))

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

    return corpus, filenames


# Function to get text from the selected document
def get_text_from_selected_doc(selected_doc):
    # Implement this function based on how your documents are stored or fetched
    # For example, if you have the documents stored locally:
    corpus_dir = 'politics'

    for filename in os.listdir(corpus_dir):
        if filename == selected_doc and filename.endswith('.txt'):
            file_path = os.path.join(corpus_dir, filename)
            with open(file_path, mode='rt', encoding='utf-8') as fp:
                document_text = fp.read()
            return document_text  # Return the text when the selected file is found

    # If the loop completes without finding the selected file, return an appropriate value or handle the case
    return "Selected file not found"

# Streamlit app setup
st.title('Document Summarizer')
document_list, filenames = get_documents()  # Get document paths from Google Drive

# Dropdown to select document
selected_doc = st.selectbox('Select Document', '001.txt')

models = ['Feature Vector',"Word2Vec","TFHub","T5"]

# Dropdown to select model
selected_model = st.selectbox('Select Model', models)

# Function to summarize and highlight
def summarize_and_highlight(text):
    # summary = summarize_text(text)  # Use your ML model to summarize
    # Highlight sentences returned by the model
    for sentence in text.split('.'):
            st.write(sentence.strip())

# Button to trigger the process
if st.button('Process'):
    with st.spinner('Summarizing...'):
        document_text = get_text_from_selected_doc(selected_doc)  # Fetch text from the selected document
        summarize_and_highlight(document_text)  # Summarize and highlight