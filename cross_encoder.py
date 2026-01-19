import streamlit as st
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class CrossEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text1, text2):
        # Tokenize inputs
        inputs = self.tokenizer(
            text1,
            text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get model outputs
        outputs = self.encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        
        # Use [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Calculate similarity score
        similarity_score = self.classifier(pooled_output)
        return similarity_score

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CrossEncoder().to(device)
    return model

def compute_similarity(model, text1, text2):
    model.eval()
    with torch.no_grad():
        score = model(text1, text2)
    return score.item()

# Streamlit UI
st.title("Cross-Encoder Sentence Similarity Calculator")

# File uploaders for GT and LLM CSVs
gt_file = st.file_uploader("Upload Ground Truth (GT) CSV", type=["csv"])
llm_file = st.file_uploader("Upload LLM-generated output CSV", type=["csv"])

# Load model
model = load_model()

if gt_file and llm_file:
    # Read CSV files
    gt_df = pd.read_csv(gt_file)
    llm_df = pd.read_csv(llm_file)

    # Check if required columns exist
    required_columns = {"Column", "Data_Description", "Data_Type"}
    if not required_columns.issubset(gt_df.columns) or not required_columns.issubset(llm_df.columns):
        st.error("Both CSVs must contain 'Column', 'Data_Description', and 'Data_Type'.")
    else:
        # Merge data on 'Column' to ensure row-by-row matching
        merged_df = pd.merge(gt_df, llm_df, on="Column", suffixes=("_GT", "_LLM"))

        # Let the user choose comparison column
        compare_column = st.selectbox("Select the column for comparison", ["Data_Description", "Data_Type"])

        if st.button("Compute Similarity"):
            # Show progress bar
            progress_bar = st.progress(0)
            similarities = []
            
            # Compute similarity for each matching row
            for idx, row in merged_df.iterrows():
                gt_text = str(row[f"{compare_column}_GT"])
                llm_text = str(row[f"{compare_column}_LLM"])

                # Compute similarity using cross-encoder
                similarity = compute_similarity(model, gt_text, llm_text)
                similarities.append(similarity)
                
                # Update progress bar
                progress_bar.progress((idx + 1) / len(merged_df))

            # Add similarity scores to DataFrame
            merged_df["Similarity Score"] = similarities

            # Compute average similarity
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            # Keep only relevant columns
            output_df = merged_df[["Column", f"{compare_column}_GT", f"{compare_column}_LLM", "Similarity Score"]]
            output_df.columns = ["Column Name", "GT Description", "LLM Description", "Similarity Score"]

            # Display results
            st.write("### Similarity Results")
            st.dataframe(output_df)

            # Display average similarity
            st.write(f"### 📊 Average Similarity Score: **{avg_similarity:.4f}**")

            # Download button for CSV
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="similarity_results.csv",
                mime="text/csv"
            )