import streamlit as st
import pandas as pd
import json
import io

def extract_data_dict(json_data, selected_key):
    """Extract relevant fields from JSON and convert to DataFrame."""
    try:
        # Load JSON data
        data = json.loads(json_data)

        # Extract user-selected dictionary
        data_dict = data.get(selected_key, {})

        # Check if the extracted data is a valid dictionary
        if not isinstance(data_dict, dict):
            st.error(f"Error: The extracted data for key '{selected_key}' is not a dictionary.")
            return None

        # Convert into a structured DataFrame
        rows = []
        for column, details in data_dict.items():
            rows.append({
                "Column": column,
                "Data_Type": details.get("data_type", "Unknown"),
                "Data_Description": details.get("data_description", "No description available")
            })

        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error processing JSON: {e}")
        return None

# Streamlit UI
st.title("📑 JSON to CSV Data Dictionary Converter")

# File uploader
uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

# Dropdown for dictionary key selection
selected_key = st.selectbox("Select the dictionary key:", ["llm_generated", "human_edited_gt"])

if uploaded_file:
    # Read uploaded file
    json_data = uploaded_file.read().decode("utf-8")

    if st.button("Process JSON"):
        # Process the JSON data
        df = extract_data_dict(json_data, selected_key)

        if df is not None and not df.empty:
            # Display the DataFrame
            st.subheader("Extracted Data Dictionary:")
            st.dataframe(df)

            # Convert DataFrame to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            # Download button
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="data_dictionary.csv",
                mime="text/csv"
            )
        else:
            st.warning("No valid data extracted. Please check the JSON file.")
