import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc
import sys

# Force CPU usage to avoid CUDA memory issues
torch.cuda.is_available = lambda: False

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

def initialize_model():
    """Initialize the model and tokenizer with a smaller model"""
    try:
        with st.spinner('Loading NLI model... This may take a minute...'):
            # Clear memory
            gc.collect()
            
            # Use a smaller model
            model_name = "valhalla/distilbart-mnli-12-3"
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Force CPU mode
            model = model.cpu()
            
            # Store in session state
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            
            return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

def compute_similarity(text1, text2):
    """Compute similarity score"""
    try:
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        
        # Prepare inputs
        inputs = tokenizer(
            text1,
            text2,
            max_length=256,  # Reduced from 512
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Ensure CPU
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return predictions[0][2].item()
    
    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return 0.0

# Main UI
st.title("NLI Similarity Calculator")

# Display Python and PyTorch versions
st.write(f"Python version: {sys.version}")
st.write(f"PyTorch version: {torch.__version__}")

# Initialize model
if st.session_state.model is None:
    if not initialize_model():
        st.stop()
    else:
        st.success("✅ Model loaded successfully!")

# File uploaders
gt_file = st.file_uploader("Upload Ground Truth (GT) CSV", type=["csv"])
llm_file = st.file_uploader("Upload LLM-generated output CSV", type=["csv"])

if gt_file and llm_file:
    try:
        # Read CSV files with explicit encoding
        gt_df = pd.read_csv(gt_file, encoding='utf-8')
        llm_df = pd.read_csv(llm_file, encoding='utf-8')

        # Check required columns
        required_columns = {"Column", "Data_Description", "Data_Type"}
        if not required_columns.issubset(gt_df.columns) or not required_columns.issubset(llm_df.columns):
            st.error("Both CSVs must contain 'Column', 'Data_Description', and 'Data_Type'.")
        else:
            # Merge data
            merged_df = pd.merge(gt_df, llm_df, on="Column", suffixes=("_GT", "_LLM"))

            # Column selection
            compare_column = st.selectbox("Select column for comparison", ["Data_Description", "Data_Type"])

            if st.button("Compute Similarity"):
                with st.spinner("Computing similarities..."):
                    progress_bar = st.progress(0)
                    similarities = []
                    
                    # Process each row
                    for idx, row in merged_df.iterrows():
                        gt_text = str(row[f"{compare_column}_GT"])
                        llm_text = str(row[f"{compare_column}_LLM"])
                        
                        similarity = compute_similarity(gt_text, llm_text)
                        similarities.append(similarity)
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(merged_df))

                    # Create results DataFrame
                    output_df = pd.DataFrame({
                        "Column Name": merged_df["Column"],
                        "GT Description": merged_df[f"{compare_column}_GT"],
                        "LLM Description": merged_df[f"{compare_column}_LLM"],
                        "NLI Similarity": similarities
                    })

                    # Display results
                    st.write("### Results")
                    st.dataframe(output_df)
                    
                    # Average similarity
                    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                    st.write(f"### Average Similarity Score: {avg_similarity:.4f}")
                    
                    # Download option
                    csv = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "similarity_results.csv",
                        "text/csv"
                    )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please ensure your CSV files are properly formatted and try again.")