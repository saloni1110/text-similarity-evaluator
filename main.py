import streamlit as st
import pandas as pd
from importlib import import_module
import sys
import os

def load_module(module_name, file_path):
    """Dynamically import a Python module from file path"""
    try:
        # Add the directory containing the module to Python path
        module_dir = os.path.dirname(os.path.abspath(file_path))
        if module_dir not in sys.path:
            sys.path.append(module_dir)
            
        # Import the module
        module = import_module(module_name)
        return module
    except Exception as e:
        st.error(f"Error loading {module_name}: {str(e)}")
        return None

def main():
    st.title("Text Similarity Analysis Tool")
    
    # Sidebar for method selection
    st.sidebar.title("Settings")
    methods = st.sidebar.multiselect(
        "Select Similarity Methods",
        ["S-BERT", "NLI-BART", "Cross-Encoder"],
        default=["S-BERT"]
    )
    
    # File uploaders
    gt_file = st.file_uploader("Upload Ground Truth (GT) CSV", type=["csv"])
    llm_file = st.file_uploader("Upload LLM-generated output CSV", type=["csv"])
    
    if gt_file and llm_file:
        # Read CSV files
        gt_df = pd.read_csv(gt_file)
        llm_df = pd.read_csv(llm_file)
        
        # Check required columns
        required_columns = {"Column", "Data_Description", "Data_Type"}
        if not required_columns.issubset(gt_df.columns) or not required_columns.issubset(llm_df.columns):
            st.error("Both CSVs must contain 'Column', 'Data_Description', and 'Data_Type'.")
        else:
            # Merge data on 'Column' to ensure row-by-row matching
            merged_df = pd.merge(gt_df, llm_df, on="Column", suffixes=("_GT", "_LLM"))
        
        # Column selection
        compare_column = st.selectbox(
            "Select column for comparison",
            ["Data_Description", "Data_Type"]
        )
        
        if st.button("Compute Similarities"):
            # Dictionary to store results from each method
            all_results = {}
            
            # Load and run selected methods
            with st.spinner("Computing similarities..."):
                for method in methods:
                    try:
                        if method == "S-BERT":
                            sbert = load_module("s_bert_v2", "s_bert_v2.py")
                            if sbert:
                                similarities = sbert.compute_similarities(gt_df, llm_df, compare_column)
                                all_results["S-BERT"] = similarities
                        
                        elif method == "NLI-BART":
                            nli_bart = load_module("NLI_BART", "NLI-BART.py")
                            if nli_bart:
                                similarities = nli_bart.compute_similarities(gt_df, llm_df, compare_column)
                                all_results["NLI-BART"] = similarities
                        
                        elif method == "Cross-Encoder":
                            cross_encoder = load_module("cross_encoder", "cross_encoder.py")
                            if cross_encoder:
                                similarities = cross_encoder.compute_similarities(gt_df, llm_df, compare_column)
                                all_results["Cross-Encoder"] = similarities
                    
                    except Exception as e:
                        st.error(f"Error running {method}: {str(e)}")
            
            # Display results
            if all_results:
                # Create results DataFrame
                results_df = pd.DataFrame({
                    "Column Name": gt_df["Column"],
                    "GT Description": gt_df[compare_column],
                    "LLM Description": llm_df[compare_column],
                })
                
                # Add similarity scores from each method
                for method, similarities in all_results.items():
                    results_df[f"{method} Similarity"] = similarities
                
                # Display results table
                st.write("### Similarity Results")
                st.dataframe(results_df)
                
                # Display average similarities
                st.write("### Average Similarity Scores")
                for method, similarities in all_results.items():
                    avg_similarity = sum(similarities) / len(similarities)
                    st.write(f"{method}: {avg_similarity:.4f}")
                
                # Download results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "similarity_results.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    main()