import os
import sys
import pandas as pd
import streamlit as st
import time
from dataclasses import dataclass, field
import typing as t
from datetime import datetime
import tempfile
from io import BytesIO

# RAGAS core imports
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import AnswerSimilarity
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.base import SingleTurnMetric, MetricType

# Updated HuggingFace embeddings import
from langchain_huggingface import HuggingFaceEmbeddings

# Set page config
st.set_page_config(page_title="RAGAS Evaluation Tool", layout="wide")

# Title and description
st.title("RAGAS Evaluation Tool")
st.markdown("""
This tool evaluates conversational AI responses using RAGAS metrics.
Upload a CSV file with the following columns:
- **User Query**: The question asked by the user
- **Grounding Content**: The reference content that should inform the answer
- **Bot Response**: The actual response from your AI system
""")
import streamlit as st
st.title("🔍 App Started")
st.write("✅ Streamlit script has loaded.")

# Function to remove duplicate columns
def remove_duplicate_columns(df):
    """Remove duplicate columns from dataframe"""
    return df.loc[:, ~df.columns.duplicated()]

# Custom Hallucinations metric = 1 - similarity
@dataclass
class HallucinationsMetric(SingleTurnMetric):
    name: str = "hallucinations"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    def __init__(self):
        super().__init__()
        
    def init(self, run_config):
        # Implementation of the required abstract method
        super().init(run_config)

    async def _single_turn_ascore(self, sample, callbacks=None):
        score = await similarity.single_turn_ascore(sample)
        return 1.0 - score

# Functions for RAGAS evaluation
def process_file(uploaded_file, progress_bar):
    # Load data
    progress_bar.progress(10, text="Loading CSV file...")
    df = pd.read_csv(uploaded_file)
    progress_bar.progress(20, text=f"Loaded {len(df)} rows")
    
    # Build RAGAS samples
    progress_bar.progress(30, text="Building evaluation samples...")
    samples = [
        SingleTurnSample(
            user_input=row["User Query"],
            response=row["Bot Response"],
            retrieved_contexts=[row["Grounding Content"]],
            reference=row["Grounding Content"],
        )
        for _, row in df.iterrows()
    ]
    progress_bar.progress(40, text=f"{len(samples)} samples built")
    
    # Setup embeddings & metrics
    progress_bar.progress(50, text="Initializing embeddings & metrics...")
    lc_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
    
    # Create metrics
    global similarity
    similarity = AnswerSimilarity()
    halluc = HallucinationsMetric()
    progress_bar.progress(60, text="Starting evaluation...")
    
    # Run evaluation
    ds = EvaluationDataset(samples)
    results = evaluate(
        ds,
        metrics=[similarity, halluc],
        embeddings=ragas_embeddings,
        llm=None,
        show_progress=False,
    )
    progress_bar.progress(80, text="Evaluation complete")
    
    # Process results
    metrics_df = results.to_pandas()
    metric_columns = [col for col in metrics_df.columns if col not in ['user_input', 'retrieved_contexts', 'response', 'reference']]
    metrics_only_df = metrics_df[metric_columns]
    
    # Check if columns already exist in the original dataframe and rename if needed
    existing_columns = df.columns
    renamed_metrics = {}
    
    for col in metrics_only_df.columns:
        if col in existing_columns:
            renamed_metrics[col] = f"{col}_ragas"
    
    # Rename columns if needed to avoid duplicates
    if renamed_metrics:
        metrics_only_df = metrics_only_df.rename(columns=renamed_metrics)
    
    # Merge with original data
    df_out = pd.concat([df.reset_index(drop=True), metrics_only_df.reset_index(drop=True)], axis=1)
    
    # Remove any remaining duplicate columns
    df_out = remove_duplicate_columns(df_out)
    
    progress_bar.progress(100, text="Processing complete!")
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_with_ragas_{timestamp}.csv"
    
    return df_out, output_filename

# File upload section
st.header("1. Upload Input CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Preview uploaded file
if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded file:")
    st.dataframe(df_preview.head())
    
    # Reset file pointer after preview
    uploaded_file.seek(0)
    
    # Check for required columns
    required_columns = ["User Query", "Bot Response", "Grounding Content"]
    missing_columns = [col for col in required_columns if col not in df_preview.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        # Process button
        if st.button("Run RAGAS Evaluation"):
            # Progress bar
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                progress_bar = st.progress(0, text="Starting...")
                
                try:
                    # Process file
                    results_df, output_filename = process_file(uploaded_file, progress_bar)
                    
                    # Make sure there are no duplicate columns before displaying
                    results_df = remove_duplicate_columns(results_df)
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    st.dataframe(results_df)
                    
                    # Visualization section
                    st.subheader("Metrics Visualization")
                    
                    # Find metric columns
                    metric_cols = [col for col in results_df.columns 
                                if "semantic_similarity" in col.lower() or "hallucination" in col.lower()]
                    
                    if metric_cols:
                        # Create two columns for charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Distribution of semantic similarity
                            similarity_col = next((col for col in metric_cols if "similarity" in col), None)
                            if similarity_col:
                                st.write(f"Distribution of {similarity_col}")
                                hist_data = results_df[similarity_col].dropna()
                                st.bar_chart(hist_data.value_counts().sort_index())
                        
                        with col2:
                            # Distribution of hallucinations
                            halluc_col = next((col for col in metric_cols if "hallucination" in col), None)
                            if halluc_col:
                                st.write(f"Distribution of {halluc_col}")
                                hist_data = results_df[halluc_col].dropna()
                                st.bar_chart(hist_data.value_counts().sort_index())
                    
                    # Download button
                    csv_buffer = BytesIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_buffer,
                        file_name=output_filename,
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
            
# Metrics explanation section
st.header("Understanding the Metrics")
st.subheader("Semantic Similarity")
st.markdown("""
**Semantic Similarity** measures how similar the bot's response is to the grounding content semantically.
- **Higher values (closer to 1.0)** indicate the bot's response aligns well with the reference content.
- **Lower values (closer to 0.0)** suggest the response deviates from the reference content.
""")

st.subheader("Hallucinations Metric")
st.markdown("""
**Hallucinations Metric** is calculated as (1 - Semantic Similarity) and measures potential fabrication.
- **Lower values (closer to 0.0)** are better, indicating minimal hallucination.
- **Higher values (closer to 1.0)** suggest the response contains information not supported by the reference content.
""")

# Instructions for running
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Upload a CSV file with required columns:
   - User Query
   - Grounding Content
   - Bot Response
2. Click "Run RAGAS Evaluation"
3. View results and download the output CSV
""")