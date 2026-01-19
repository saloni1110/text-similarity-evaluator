import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import pandas as pd
import gc

# Initialize torch with specific settings for Mac
import torch
if hasattr(torch, 'set_default_tensor_type'):
    torch.set_default_tensor_type('torch.FloatTensor')
torch.set_grad_enabled(False)

# Rest of your imports
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Initialize session state for storing data
if 'similarity_data' not in st.session_state:
    st.session_state.similarity_data = None
if 'has_computed' not in st.session_state:
    st.session_state.has_computed = False

# Callback function for data editor
def handle_data_edit(edited_df):
    st.session_state.similarity_data = edited_df

# -- S-BERT Model --
def sbert_similarity(gt_text, llm_text, model):
    embeddings = model.encode([gt_text, llm_text])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

# -- Cross-Encoder Model --
class CrossEncoderModel:
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def compute_similarity(self, text1, text2):
        inputs = self.tokenizer(text1, text2, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        outputs = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        similarity_score = self.classifier(pooled_output)
        return similarity_score.item()

# -- NLI-BART Model --
def initialize_nli_model():
    model_name = "valhalla/distilbart-mnli-12-3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).cpu()
    return model, tokenizer

def compute_nli_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer(text1, text2, max_length=256, truncation=True, padding=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions[0][2].item()

# -- Streamlit UI and App Logic --
st.title("Multi-Model Sentence Similarity Calculator")

gt_file = st.file_uploader("Upload Ground Truth (GT) CSV", type=["csv"])
llm_file = st.file_uploader("Upload LLM-generated output CSV", type=["csv"])

model_choices = st.multiselect("Select similarity models (choose multiple)", ["S-BERT", "Cross-Encoder", "NLI BART"])

# Initialize models only when needed
models = {}
if model_choices:
    if "S-BERT" in model_choices:
        models["S-BERT"] = SentenceTransformer('all-MiniLM-L6-v2')
    if "Cross-Encoder" in model_choices:
        models["Cross-Encoder"] = CrossEncoderModel()
    if "NLI BART" in model_choices:
        models["NLI BART"], nli_tokenizer = initialize_nli_model()

if gt_file and llm_file and not st.session_state.has_computed:
    try:
        gt_df = pd.read_csv(gt_file, encoding='utf-8')
        llm_df = pd.read_csv(llm_file, encoding='utf-8')
        required_columns = {"Column", "Data_Description", "Data_Type"}
        
        if not required_columns.issubset(gt_df.columns) or not required_columns.issubset(llm_df.columns):
            st.error("Both CSVs must contain 'Column', 'Data_Description', and 'Data_Type'.")
        else:
            merged_df = pd.merge(gt_df, llm_df, on="Column", suffixes=("_GT", "_LLM"))
            compare_column = st.selectbox("Select the column for comparison", ["Data_Description", "Data_Type"])
            
            if st.button("Compute Similarity"):
                progress_bar = st.progress(0)
                similarities = []
                
                for idx, row in merged_df.iterrows():
                    gt_text = str(row[f"{compare_column}_GT"])
                    llm_text = str(row[f"{compare_column}_LLM"])
                    similarity_scores = {"Column Name": row["Column"], "GT Description": gt_text, "LLM Description": llm_text}
                    
                    sbert_score = cross_score = nli_score = None
                    
                    for model_name, model in models.items():
                        if model_name == "S-BERT":
                            sbert_score = sbert_similarity(gt_text, llm_text, model)
                            similarity_scores["S-BERT Similarity"] = sbert_score
                        elif model_name == "Cross-Encoder":
                            cross_score = model.compute_similarity(gt_text, llm_text)
                            similarity_scores["Cross-Encoder Similarity"] = cross_score
                        elif model_name == "NLI BART":
                            nli_score = compute_nli_similarity(model, nli_tokenizer, gt_text, llm_text)
                            similarity_scores["NLI BART Similarity"] = nli_score
                    
                    if all(v is not None for v in [sbert_score, cross_score, nli_score]):
                        similarity_scores["Weighted Average Similarity"] = round(
                            (0.4 * cross_score) + (0.3 * sbert_score) + (0.3 * nli_score), 4)
                    
                    similarity_scores["Human Score"] = ""
                    similarities.append(similarity_scores)
                    progress_bar.progress((idx + 1) / len(merged_df))
                
                st.session_state.similarity_data = pd.DataFrame(similarities)
                st.session_state.has_computed = True
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display results if they exist in session state
if st.session_state.similarity_data is not None:
    # Calculate and display averages
    avg_scores = st.session_state.similarity_data.mean(numeric_only=True).to_dict()
    for key, value in avg_scores.items():
        st.write(f"### 📊 Average {key}: **{value:.4f}**")
    
    st.write("### Similarity Results")
    # Use data editor with callback and key
    edited_df = st.data_editor(
        st.session_state.similarity_data,
        num_rows="dynamic",
        key="similarity_table",
        on_change=handle_data_edit,
        args=(st.session_state.similarity_data,)
    )
    
    # Update session state with edited data
    st.session_state.similarity_data = edited_df
    
    # Download button for current state
    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Results CSV",
        data=csv,
        file_name="similarity_results.csv",
        mime="text/csv"
    )

# Add a reset button
if st.session_state.has_computed:
    if st.button("Reset"):
        st.session_state.similarity_data = None
        st.session_state.has_computed = False
        st.rerun()