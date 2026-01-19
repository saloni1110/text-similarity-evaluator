#Taking input from streamlit UI - two i/p files - one for GT, one for LLM-generated
# Output - Pair wise s-bert similarity basis on the column name, avg similarity
#ensures that each row in both files has the same value in the "column_name" field 

#import libraries
import streamlit as st #Used to build the web UI for the app.
import pandas as pd #For handling CSV files and data manipulation.
from sentence_transformers import SentenceTransformer, util #These are used to load the pre-trained model and compute cosine similarity between text embeddings.

# Load S-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("S-BERT Sentence Similarity Calculator")

# File uploaders for GT and LLM CSVs
gt_file = st.file_uploader("Upload Ground Truth (GT) CSV", type=["csv"])
llm_file = st.file_uploader("Upload LLM-generated output CSV", type=["csv"])

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
            similarities = []
            
            # Compute similarity for each matching row
            for _, row in merged_df.iterrows():
                gt_text = str(row[f"{compare_column}_GT"])
                llm_text = str(row[f"{compare_column}_LLM"])

                # Encode and compute cosine similarity
                embeddings = model.encode([gt_text, llm_text])
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                similarities.append(similarity)

            # Add similarity scores to DataFrame
            merged_df["Cosine Similarity"] = similarities

            # Compute average similarity
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            # Keep only relevant columns
            output_df = merged_df[["Column", f"{compare_column}_GT", f"{compare_column}_LLM", "Cosine Similarity"]]
            output_df.columns = ["Column Name", "GT Description", "LLM Description", "Cosine Similarity"]

            # Display results
            st.write("### Similarity Results")
            st.dataframe(output_df)

            # Display average similarity
            st.write(f"### 📊 Average Cosine Similarity: **{avg_similarity:.4f}**")

            # Download button for CSV
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="similarity_results.csv",
                mime="text/csv"
            )
