import streamlit as st
import torch
import torch.nn as nn
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# --- 1. Page Config & NLTK Setup ---
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")

@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
setup_nltk()

# --- 2. Model Architecture (Must match training exactly) ---
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, dropout_rate):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim, 
            num_layers=num_layers, bidirectional=bidirectional, 
            dropout=dropout_rate if num_layers > 1 else 0, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, 1)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden_final = hidden[-1,:,:]
        hidden_final = self.dropout(hidden_final)
        logits = self.fc(hidden_final)
        return logits.squeeze(-1)

# --- 3. Load Artifacts & Model ---
@st.cache_resource
def load_pipeline():
    # Load vocabulary and configs
    with open('pipeline_artifacts.json', 'r') as f:
        artifacts = json.load(f)
    
    # Initialize model
    model = SentimentLSTM(**artifacts['model_config'])
    
    # CRITICAL: Map location to CPU. You trained on GPU, but web servers (like Streamlit Cloud) use CPU.
    device = torch.device('cpu')
    model.load_state_dict(torch.load('best_lstm_model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    return model, artifacts['vocab'], artifacts['max_seq_len'], device

model, vocab, max_seq_len, device = load_pipeline()

# --- 4. Preprocessing Functions ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
negations = {'not', 'no', 'nor', 'never', 'none', 'neither', 'cannot', 'wasn\'t', 'isn\'t', 'aren\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'shouldn\'t', 'couldn\'t', 'hasn\'t', 'haven\'t', 'hadn\'t'}
stop_words = stop_words - negations

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'&[^;\s]+;', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def text_to_indices(text, word2idx, max_seq_length):
    tokens = text.split()
    indices = [word2idx.get(word, 1) for word in tokens]
    if len(indices) > max_seq_length:
        indices = indices[:max_seq_length]
    else:
        indices = indices + [0] * (max_seq_length - len(indices))
    return indices

def predict(text):
    cleaned = clean_text(text)
    indices = text_to_indices(cleaned, vocab, max_seq_len)
    tensor_input = torch.tensor([indices], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logit = model(tensor_input)
        prob = torch.sigmoid(logit).item()
        
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if sentiment == "Positive" else 1 - prob
    return sentiment, confidence

# --- 5. Streamlit UI ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Powered by a custom Bidirectional LSTM trained on 50,000 IMDB reviews.")

user_review = st.text_area("Enter a movie review:", height=150, placeholder="e.g., The cinematography was breathtaking, but the plot dragged on forever...")

if st.button("Analyze Sentiment", type="primary"):
    if user_review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            sentiment, confidence = predict(user_review)
            
            st.markdown("---")
            if sentiment == "Positive":
                st.success(f"**Sentiment:** {sentiment} 🟢")
            else:
                st.error(f"**Sentiment:** {sentiment} 🔴")
                
            st.info(f"**Model Confidence:** {confidence * 100:.2f}%")
            
            with st.expander("See preprocessed text"):
                st.write(clean_text(user_review))