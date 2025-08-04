import os
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import logging
logging.basicConfig(level=logging.DEBUG)
# Import RAGAS and LangChain components directly from their packages
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import AnswerSimilarity
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

# If you have a custom HallucinationsMetric, define it here
from dataclasses import dataclass, field
import typing as t
from ragas.metrics.base import SingleTurnMetric, MetricType

@dataclass
class HallucinationsMetric(SingleTurnMetric):
    name: str = "hallucinations_metric"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    def __init__(self):
        super().__init__()

    def init(self, run_config):
        super().init(run_config)

    async def _single_turn_ascore(self, sample, callbacks=None):
        score = await similarity.single_turn_ascore(sample)
        return 1.0 - score

st.set_page_config(
    page_title="RAGAS Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

def remove_duplicate_columns(df):
    """Remove duplicate columns from dataframe"""
    return df.loc[:, ~df.columns.duplicated()]

def run_ragas_evaluation(input_file):
    """Run RAGAS evaluation on the given input file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"output_with_ragas_{timestamp}.csv"

    # Load data
    df = pd.read_csv(input_file)

    # Build RAGAS samples
    samples = [
        SingleTurnSample(
            user_input=row["User Query"],
            response=row["Bot Response"],
            retrieved_contexts=[row["Grounding Content"]],
            reference=row["Grounding Content"],
        )
        for _, row in df.iterrows()
    ]

    # Setup embeddings & similarity metric
    lc_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    # Instantiate metrics
    global similarity
    similarity = AnswerSimilarity()
    halluc = HallucinationsMetric()

    # Run evaluation
    ds = EvaluationDataset(samples)
    results = evaluate(
        ds,
        metrics=[similarity, halluc],
        embeddings=ragas_embeddings,
        llm=None,
        show_progress=True,
    )

    metrics_df = results.to_pandas()

    # Only take the metric columns from the results
    metric_columns = [col for col in metrics_df.columns if col not in ['user_input', 'retrieved_contexts', 'response', 'reference']]
    metrics_only_df = metrics_df[metric_columns]

    # Check if columns already exist in the original dataframe and avoid duplicates
    existing_columns = df.columns
    metrics_to_add = []
    renamed_metrics = {}

    for col in metrics_only_df.columns:
        if col in existing_columns:
            new_col = f"{col}_ragas"
            renamed_metrics[col] = new_col
            metrics_to_add.append(new_col)
        else:
            metrics_to_add.append(col)

    # Rename columns if needed to avoid duplicates
    if renamed_metrics:
        metrics_only_df = metrics_only_df.rename(columns=renamed_metrics)

    # Concatenate only the unique metric columns
    df_out = pd.concat([df.reset_index(drop=True), metrics_only_df.reset_index(drop=True)], axis=1)

    # Save to CSV
    df_out.to_csv(output_csv, index=False)

    return df_out, output_csv

# App title and description
st.title("🔍 RAGAS Evaluation Dashboard")
st.markdown("""
This application helps you evaluate RAG (Retrieval-Augmented Generation) systems using RAGAS metrics.
Upload your conversation logs to get started.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This tool evaluates RAG systems using:

    * **Semantic Similarity**: Measures how similar the response is to reference content
    * **Hallucination Metric**: Measures the degree of hallucination (1 - similarity)
    """)

    st.header("Instructions")
    st.markdown("""
    1. Upload a CSV file with these columns:
       - User Query
       - Bot Response
       - Grounding Content
    2. Click "Run Evaluation"
    3. View and download results
    """)

# File upload
uploaded_file = st.file_uploader("Upload your conversation logs (CSV)", type=["csv"])

if uploaded_file is not None:
    # Preview the uploaded data
    df = pd.read_csv(uploaded_file)

    # Check if the required columns exist
    required_columns = ["User Query", "Bot Response", "Grounding Content"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        # Display sample of data
        st.subheader("Data Preview")
        st.dataframe(df.head(5), use_container_width=True)

        # Button to run evaluation
        if st.button("Run RAGAS Evaluation", type="primary"):
            with st.spinner("Running evaluation... This may take a few minutes"):
                # Save uploaded file temporarily
                temp_file = "temp_input.csv"
                df.to_csv(temp_file, index=False)

                try:
                    # Run evaluation
                    results_df, output_file = run_ragas_evaluation(temp_file)

                    # Remove any duplicate columns before displaying
                    results_df = remove_duplicate_columns(results_df)

                    st.success(f"Evaluation complete! Results saved to {output_file}")

                    # Display results
                    st.subheader("Evaluation Results")
                    st.dataframe(results_df, use_container_width=True)

                    # Download button for results
                    with open(output_file, "rb") as file:
                        st.download_button(
                            label="Download Results CSV",
                            data=file,
                            file_name=output_file,
                            mime="text/csv",
                        )

                    # Visualizations
                    st.subheader("Metrics Visualization")

                    # Get metrics columns
                    metric_cols = [col for col in results_df.columns if col in ['semantic_similarity', 'hallucinations_metric', 'semantic_similarity_ragas', 'hallucinations_metric_ragas']]

                    if metric_cols:
                        # Create visualizations
                        col1, col2 = st.columns(2)

                        with col1:
                            for metric in metric_cols:
                                fig = px.histogram(results_df, x=metric, title=f"Distribution of {metric}")
                                st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # If we have both similarity and hallucination metrics
                            similarity_col = next((col for col in metric_cols if 'similarity' in col.lower()), None)
                            hallucination_col = next((col for col in metric_cols if 'hallucination' in col.lower()), None)

                            if similarity_col and hallucination_col:
                                fig = px.scatter(results_df, x=similarity_col, y=hallucination_col, 
                                                 title=f"{similarity_col} vs {hallucination_col}")
                                st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred during evaluation: {str(e)}")
                    st.exception(e)

                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)