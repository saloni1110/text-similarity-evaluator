import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import pandas as pd
import gc

# Initialize torch with specific settings for Mac
import torch
torch.set_grad_enabled(False)

# Set device
device = 'cpu'  # Use CPU for Mac compatibility
if torch.cuda.is_available():
    device = 'cuda'

# Rest of your imports
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Initialize session state for storing data
if 'similarity_data' not in st.session_state:
    st.session_state.similarity_data = None
if 'has_computed' not in st.session_state:
    st.session_state.has_computed = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'nli_tokenizer' not in st.session_state:
    st.session_state.nli_tokenizer = None

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
    def __init__(self, model_name='bert-base-uncased', max_length=128, device='cpu'):
        self.device = device
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        ).to(device)

    def compute_similarity(self, text1, text2):
        inputs = self.tokenizer(text1, text2, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        similarity_score = self.classifier(pooled_output)
        return similarity_score.item()

# -- NLI-BART Model --
def initialize_nli_model(device='cpu'):
    model_name = "valhalla/distilbart-mnli-12-3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return model, tokenizer

def compute_nli_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer(text1, text2, max_length=256, truncation=True, padding=True, return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions[0][2].item()

# -- Streamlit UI and App Logic --
st.title("Multi-Model Sentence Similarity Calculator")

gt_file = st.file_uploader("Upload Ground Truth (GT) CSV", type=["csv"])
llm_file = st.file_uploader("Upload LLM-generated output CSV", type=["csv"])

model_choices = st.multiselect("Select similarity models (choose multiple)", ["S-BERT", "Cross-Encoder", "NLI BART"])

if gt_file and llm_file and not st.session_state.has_computed:
    try:
        gt_df = pd.read_csv(gt_file, encoding='utf-8')
        llm_df = pd.read_csv(llm_file, encoding='utf-8')
        required_columns = {"Column", "Data_Description", "Data_Type", "Validation", "Constriants"}
        
        #if not required_columns.issubset(gt_df.columns) or not required_columns.issubset(llm_df.columns):
            #st.error("Both CSVs must contain 'Column', 'Data_Description', and 'Data_Type'.")
        #else:
        merged_df = pd.merge(gt_df, llm_df, on="Column", suffixes=("_GT", "_LLM"))
        compare_column = st.selectbox("Select the column for comparison", ["Data_Description", "Data_Type", "Validation", "Constriants"])
     
        if st.button("Compute Similarity"):
                # Initialize models based on selected choices
                if not model_choices:
                    st.error("Please select at least one similarity model.")
                else:
                    # Initialize models if not already initialized or if selection changed
                    st.session_state.models = {}
                    if "S-BERT" in model_choices:
                        with st.spinner("Loading S-BERT model..."):
                            st.session_state.models["S-BERT"] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                    if "Cross-Encoder" in model_choices:
                        with st.spinner("Loading Cross-Encoder model..."):
                            st.session_state.models["Cross-Encoder"] = CrossEncoderModel(device=device)
                    if "NLI BART" in model_choices:
                        with st.spinner("Loading NLI BART model..."):
                            st.session_state.models["NLI BART"], st.session_state.nli_tokenizer = initialize_nli_model(device=device)
                    
                    progress_bar = st.progress(0)
                    similarities = []
                    
                    for idx, row in merged_df.iterrows():
                        gt_text = str(row[f"{compare_column}_GT"])
                        llm_text = str(row[f"{compare_column}_LLM"])
                        similarity_scores = {"Column Name": row["Column"], "GT": gt_text, "LLM": llm_text}
                        
                        sbert_score = cross_score = nli_score = None
                        
                        for model_name, model in st.session_state.models.items():
                            if model_name == "S-BERT":
                                sbert_score = sbert_similarity(gt_text, llm_text, model)
                                similarity_scores["S-BERT Similarity"] = sbert_score
                            elif model_name == "Cross-Encoder":
                                cross_score = model.compute_similarity(gt_text, llm_text)
                                similarity_scores["Cross-Encoder Similarity"] = cross_score
                            elif model_name == "NLI BART":
                                nli_score = compute_nli_similarity(model, st.session_state.nli_tokenizer, gt_text, llm_text)
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
    st.session_state.similarity_data,  # Data to edit
    column_config={
        "Human Score": st.column_config.NumberColumn(
            "Human Score", min_value=None, max_value=1.0, step=0.01, format="%.3f"
        )
    },
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