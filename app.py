import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input, LayerNormalization, MultiHeadAttention

st.title('Comparing Bidirectional LSTM and Transformer for Stock Market Direction Prediction')

# Sidebar for user inputs
st.sidebar.header('Model Configuration')
stock_symbol = st.sidebar.text_input('Stock Symbol', 'AAPL')
date_range = st.sidebar.date_input('Date Range', value=[pd.to_datetime('2018-01-01'), pd.to_datetime('2023-01-01')])
sequence_length = st.sidebar.slider('Sequence Length', 10, 60, 30)
test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.2)
epochs = st.sidebar.slider('Epochs', 10, 100, 50)
batch_size = st.sidebar.slider('Batch Size', 16, 128, 32)

# Data loading and preprocessing
def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    return data

def preprocess_data(data, sequence_length, test_size):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(data['Target'].iloc[i])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler

# Bidirectional LSTM model
def create_bilstm_model(sequence_length):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Transformer model
def create_transformer_model(sequence_length, d_model=64, n_heads=2, ff_dim=128):
    inputs = Input(shape=(sequence_length, 1))
    
    # Embedding
    x = Dense(d_model)(inputs)
    
    # Multi-Head Attention
    x = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(d_model)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training and evaluation
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return history, accuracy, report, cm

# Main app logic
if st.sidebar.button('Run Analysis'):
    with st.spinner('Loading data...'):
        data = load_data(stock_symbol, date_range[0], date_range[1])
    
    with st.spinner('Preprocessing data...'):
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data, sequence_length, test_size)
    
    st.subheader('Data Overview')
    st.dataframe(data.tail())
    
    # Train Bidirectional LSTM
    with st.spinner('Training Bidirectional LSTM...'):
        bilstm_model = create_bilstm_model(sequence_length)
        bilstm_history, bilstm_accuracy, bilstm_report, bilstm_cm = train_and_evaluate(
            bilstm_model, X_train, y_train, X_test, y_test, epochs, batch_size
        )
    
    # Train Transformer
    with st.spinner('Training Transformer...'):
        transformer_model = create_transformer_model(sequence_length)
        transformer_history, transformer_accuracy, transformer_report, transformer_cm = train_and_evaluate(
            transformer_model, X_train, y_train, X_test, y_test, epochs, batch_size
        )
    
    # Results
    st.subheader('Model Performance Comparison')
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('**Bidirectional LSTM**')
        st.write(f'Accuracy: {bilstm_accuracy:.4f}')
        st.text(bilstm_report)
    
    with col2:
        st.write('**Transformer**')
        st.write(f'Accuracy: {transformer_accuracy:.4f}')
        st.text(transformer_report)
    
    # Plots
    st.subheader('Training History')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(bilstm_history.history['accuracy'], label='LSTM Train')
    ax1.plot(bilstm_history.history['val_accuracy'], label='LSTM Val')
    ax1.plot(transformer_history.history['accuracy'], label='Transformer Train')
    ax1.plot(transformer_history.history['val_accuracy'], label='Transformer Val')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(bilstm_history.history['loss'], label='LSTM Train')
    ax2.plot(bilstm_history.history['val_loss'], label='LSTM Val')
    ax2.plot(transformer_history.history['loss'], label='Transformer Train')
    ax2.plot(transformer_history.history['val_loss'], label='Transformer Val')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    st.pyplot(fig)
    
    # Prediction visualization
    st.subheader('Prediction Visualization')
    bilstm_pred = (bilstm_model.predict(X_test) > 0.5).astype(int)
    transformer_pred = (transformer_model.predict(X_test) > 0.5).astype(int)
    
    results = pd.DataFrame({
        'Actual': y_test,
        'LSTM Prediction': bilstm_pred.flatten(),
        'Transformer Prediction': transformer_pred.flatten()
    })
    
    st.dataframe(results.tail(20))
    
    # Feature importance (simplified for demonstration)
    st.subheader('Model Comparison Summary')
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Training Time (approx)'],
        'Bidirectional LSTM': [f'{bilstm_accuracy:.4f}', 'Faster'],
        'Transformer': [f'{transformer_accuracy:.4f}', 'Slower but more powerful']
    })
    st.table(comparison)