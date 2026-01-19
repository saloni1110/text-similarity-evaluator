import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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


# -- S-BERT Model (from sbert.py) --
def sbert_similarity(gt_text, llm_text, model):
    embeddings = model.encode([gt_text, llm_text])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

# -- Cross-Encoder Model (from cross_encoder.py) --
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

# -- NLI-BART Model (from NLI_BART.py) --
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

# File uploaders for GT and LLM CSVs
gt_file = st.file_uploader("Upload Ground Truth (GT) CSV", type=["csv"])
llm_file = st.file_uploader("Upload LLM-generated output CSV", type=["csv"])

# Multi-model selection dropdown
model_choices = st.multiselect(
    "Select similarity models (choose multiple)",
    ["S-BERT", "Cross-Encoder", "NLI BART"]
)

# Initialize models based on choices
models = {}
if "S-BERT" in model_choices:
    models["S-BERT"] = SentenceTransformer('all-MiniLM-L6-v2')

if "Cross-Encoder" in model_choices:
    models["Cross-Encoder"] = CrossEncoderModel()

if "NLI BART" in model_choices:
    models["NLI BART"], nli_tokenizer = initialize_nli_model()

# Process the files once both are uploaded
if gt_file and llm_file:
    try:
        # Read CSV files
        gt_df = pd.read_csv(gt_file, encoding='utf-8')
        llm_df = pd.read_csv(llm_file, encoding='utf-8')

        # Ensure required columns are present
        required_columns = {"Column", "Data_Description", "Data_Type"}
        if not required_columns.issubset(gt_df.columns) or not required_columns.issubset(llm_df.columns):
            st.error("Both CSVs must contain 'Column', 'Data_Description', and 'Data_Type'.")
        else:
            # Merge data based on 'Column' to match rows
            merged_df = pd.merge(gt_df, llm_df, on="Column", suffixes=("_GT", "_LLM"))

            # Let the user select the column to compare
            compare_column = st.selectbox("Select the column for comparison", ["Data_Description", "Data_Type"])

            if st.button("Compute Similarity"):
                progress_bar = st.progress(0)
                similarities = []

                # Iterate over each row
                for idx, row in merged_df.iterrows():
                    gt_text = str(row[f"{compare_column}_GT"])
                    llm_text = str(row[f"{compare_column}_LLM"])

                    # Initialize a dictionary to hold similarity scores
                    similarity_scores = {"Column Name": row["Column"], "GT Description": gt_text, "LLM Description": llm_text}

                    # Compute similarity for each selected model
                    for model_name, model in models.items():
                        if model_name == "S-BERT":
                            similarity_scores["S-BERT Similarity"] = sbert_similarity(gt_text, llm_text, model)
                        elif model_name == "Cross-Encoder":
                            similarity_scores["Cross-Encoder Similarity"] = model.compute_similarity(gt_text, llm_text)
                        elif model_name == "NLI BART":
                            similarity_scores["NLI BART Similarity"] = compute_nli_similarity(model, nli_tokenizer, gt_text, llm_text)

                    similarities.append(similarity_scores)
                    progress_bar.progress((idx + 1) / len(merged_df))

                # Create a DataFrame from the similarities
                output_df = pd.DataFrame(similarities)

                # Display results
                st.write("### Similarity Results")
                st.dataframe(output_df)

                # Compute average similarity for each model and display
                for model_name in models.keys():
                    avg_similarity = output_df[f"{model_name} Similarity"].mean() if f"{model_name} Similarity" in output_df else 0.0
                    st.write(f"### 📊 Average {model_name} Similarity: **{avg_similarity:.4f}**")

                # Download option for results
                csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results CSV", data=csv, file_name="similarity_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please ensure your CSV files are properly formatted and try again.")
