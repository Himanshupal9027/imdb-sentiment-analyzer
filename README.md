# 🎬 IMDB Movie Review Sentiment Analyzer

A web application built with **Streamlit** and **PyTorch** that classifies movie reviews as **Positive** or **Negative** using a custom **Bidirectional LSTM** neural network trained on IMDB review data.

> **🚀 Live App:** [https://imdb-sentiment-analyzer-gthmz8shme6oq7g9zgcvxy.streamlit.app/](https://imdb-sentiment-analyzer-gthmz8shme6oq7g9zgcvxy.streamlit.app/)

## 📁 Project Structure

```text
├── app.py                        # Streamlit web app, text preprocessing, and inference pipeline
├── best_lstm_model.pt            # Trained PyTorch model weights (state_dict)
├── pipeline_artifacts.json       # Saved model_config, vocab mapping, and max_seq_len
├── requirements.txt              # Python dependencies for local and Streamlit Cloud deployment
└── IMDB_SENTIMENT_ANALYZER.ipynb # Notebook for sentiment model development & evaluation
```

## 🧠 Functions & Components Explained

### `setup_nltk()`
* **Decorator:** `@st.cache_resource`
* **Purpose:** Downloads the required NLTK data packages (`punkt` for tokenization and `stopwords` for stopword filtering) once and caches them across sessions so they aren't re-downloaded on every interaction.
* **Parameters:** None.
* **Returns:** `None`.

### `SentimentLSTM` (`nn.Module`)
* **Purpose:** Custom PyTorch sequence classification model combining a token embedding layer, a configurable (unidirectional or bidirectional) multi-layer LSTM, dropout regularization, and a linear output layer producing a single logit.
* **`__init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, dropout_rate)`**:
  * Initializes `nn.Embedding` (`padding_idx=0`), `nn.LSTM` (`batch_first=True`), `nn.Dropout`, and the final `nn.Linear(..., 1)` layer.
* **`forward(self, x)`**:
  * **Parameters:** `x` (`torch.Tensor`) — Batch of padded token index sequences of shape `(batch_size, seq_len)`.
  * **Returns:** `torch.Tensor` — Raw logits of shape `(batch_size,)` (concatenating final forward and backward hidden states for bidirectional LSTMs, or using the final hidden state for unidirectional LSTMs).

### `load_pipeline()`
* **Decorator:** `@st.cache_resource`
* **Purpose:** Loads `pipeline_artifacts.json` and restores `best_lstm_model.pt` onto the target compute device (`cpu` for broad server compatibility), setting the model to evaluation mode (`model.eval()`).
* **Parameters:** None.
* **Returns:** Tuple `(model, vocab, max_seq_len, device)`.

### `clean_text(text)`
* **Purpose:** Cleans and normalizes raw review text by stripping HTML tags, URLs, HTML entities, and non-alphabetic characters, lowercasing, tokenizing with NLTK `word_tokenize`, removing non-negation stopwords, and stemming tokens with `PorterStemmer`.
* **Parameters:** `text` (`str`) — Raw input review string.
* **Returns:** `str` — Space-separated string of cleaned, stemmed tokens.

### `text_to_indices(text, word2idx, max_seq_length)`
* **Purpose:** Converts a cleaned text string into a fixed-length list of integer vocabulary indices using `1` for unknown tokens (`<UNK>`) and `0` for padding (`<PAD>`), truncating longer sequences or padding shorter ones to `max_seq_length`.
* **Parameters:**
  * `text` (`str`) — Preprocessed text string.
  * `word2idx` (`dict[str, int]`) — Token-to-index vocabulary dictionary.
  * `max_seq_length` (`int`) — Target sequence length.
* **Returns:** `list[int]` — Fixed-length list of token indices.

### `predict(text)`
* **Purpose:** Runs end-to-end inference on a raw review string using `clean_text`, `text_to_indices`, and the cached `SentimentLSTM` model under `torch.no_grad()`, applying `torch.sigmoid` to compute sentiment and confidence.
* **Parameters:** `text` (`str`) — Raw movie review entered by the user.
* **Returns:** Tuple `(sentiment, confidence)` where `sentiment` is `"Positive"` or `"Negative"` (`str`) and `confidence` is a `float` between `0.0` and `1.0`.

## 🌐 Live Demo

Try the deployed web application directly at:
**[https://imdb-sentiment-analyzer-gthmz8shme6oq7g9zgcvxy.streamlit.app/](https://imdb-sentiment-analyzer-gthmz8shme6oq7g9zgcvxy.streamlit.app/)**

## 🚀 Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```
