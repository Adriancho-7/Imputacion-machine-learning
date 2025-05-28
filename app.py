import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_processor import DataProcessor
from utils.imputation import SVRImputer
from utils.visualization import ImputationVisualizer
import io

# Configure page
st.set_page_config(
    page_title="Data Dashboard Pro - ML Imputation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'imputed_data' not in st.session_state:
    st.session_state.imputed_data = None
if 'imputation_report' not in st.session_state:
    st.session_state.imputation_report = None

def main():
    st.title("📊 Data Dashboard Pro - Enhanced ML Imputation")
    st.markdown("### Advanced Data Analysis with SVR-Based Missing Value Imputation")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section", [
        "Data Upload & Overview",
        "Missing Data Analysis", 
        "ML Imputation (SVR)",
        "Post-Imputation Analysis",
        "Download Results"
    ])
    
    if page == "Data Upload & Overview":
        data_upload_section()
    elif page == "Missing Data Analysis":
        missing_data_analysis()
    elif page == "ML Imputation (SVR)":
        ml_imputation_section()
    elif page == "Post-Imputation Analysis":
        post_imputation_analysis()
    elif page == "Download Results":
        download_section()

def data_upload_section():
    st.header("📁 Data Upload & Overview")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your dataset with missing values for imputation analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
            
            # Basic data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                missing_percent = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                st.metric("Missing Data %", f"{missing_percent:.2f}%")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Data types
            st.subheader("🔍 Data Types")
            dtype_df = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes,
                'Non-Null Count': data.count(),
                'Missing Count': data.isnull().sum(),
                'Missing %': (data.isnull().sum() / len(data) * 100).round(2)
            })
            st.dataframe(dtype_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to begin analysis")

def missing_data_analysis():
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Overview' section")
        return
    
    st.header("🔍 Missing Data Analysis")
    data = st.session_state.data
    
    # Missing data summary
    missing_summary = data.isnull().sum().sort_values(ascending=False)
    missing_summary = missing_summary[missing_summary > 0]
    
    if len(missing_summary) == 0:
        st.success("🎉 No missing data found in your dataset!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Missing Data by Column")
        if len(missing_summary) > 0:
            fig = px.bar(
                x=missing_summary.values,
                y=missing_summary.index,
                orientation='h',
                title="Missing Values Count by Column",
                labels={'x': 'Number of Missing Values', 'y': 'Columns'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Missing Data Percentage")
        missing_percent = (missing_summary / len(data) * 100).sort_values(ascending=False)
        fig = px.bar(
            x=missing_percent.values,
            y=missing_percent.index,
            orientation='h',
            title="Missing Values Percentage by Column",
            labels={'x': 'Percentage Missing', 'y': 'Columns'},
            color=missing_percent.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Missing data heatmap
    st.subheader("🗺️ Missing Data Pattern")
    visualizer = ImputationVisualizer()
    missing_heatmap = visualizer.create_missing_data_heatmap(data)
    st.plotly_chart(missing_heatmap, use_container_width=True)
    
    # Missing data statistics
    st.subheader("📋 Detailed Missing Data Statistics")
    detailed_stats = pd.DataFrame({
        'Column': missing_summary.index,
        'Missing Count': missing_summary.values,
        'Missing Percentage': (missing_summary / len(data) * 100).round(2),
        'Data Type': data[missing_summary.index].dtypes.values
    })
    st.dataframe(detailed_stats, use_container_width=True)

def ml_imputation_section():
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Overview' section")
        return
    
    st.header("🤖 Machine Learning Imputation with SVR")
    data = st.session_state.data
    
    # Check for missing data
    if data.isnull().sum().sum() == 0:
        st.success("🎉 No missing data found - imputation not needed!")
        return
    
    # Imputation parameters
    st.subheader("⚙️ SVR Imputation Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        kernel = st.selectbox("SVR Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], index=0)
    with col2:
        C = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
    with col3:
        gamma = st.selectbox("Gamma", ['scale', 'auto'], index=0)
    
    # Advanced parameters
    with st.expander("🔧 Advanced Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1, 0.01)
            max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
        with col2:
            degree = st.slider("Polynomial Degree", 2, 5, 3, 1) if kernel == 'poly' else 3
            coef0 = st.slider("Coefficient 0", 0.0, 10.0, 0.0, 0.1)
    
    # Column selection for imputation
    st.subheader("🎯 Select Columns for Imputation")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    columns_with_missing = [col for col in numeric_columns if data[col].isnull().sum() > 0]
    
    if not columns_with_missing:
        st.warning("⚠️ No numeric columns with missing values found for SVR imputation")
        return
    
    selected_columns = st.multiselect(
        "Choose columns to impute",
        columns_with_missing,
        default=columns_with_missing,
        help="SVR works best with numeric columns"
    )
    
    if st.button("🚀 Start SVR Imputation", type="primary"):
        if not selected_columns:
            st.error("❌ Please select at least one column for imputation")
            return
        
        with st.spinner("🔄 Performing SVR imputation... This may take a few moments."):
            try:
                # Initialize imputer
                imputer = SVRImputer(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    epsilon=epsilon,
                    max_iter=max_iter,
                    degree=degree,
                    coef0=coef0
                )
                
                # Perform imputation
                imputed_data, report = imputer.fit_transform(data, selected_columns)
                
                # Store results
                st.session_state.imputed_data = imputed_data
                st.session_state.imputation_report = report
                
                st.success("✅ SVR Imputation completed successfully!")
                
                # Display imputation summary
                st.subheader("📊 Imputation Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_imputed = sum(report['imputed_counts'].values())
                    st.metric("Total Values Imputed", total_imputed)
                
                with col2:
                    avg_score = np.mean(list(report['model_scores'].values()))
                    st.metric("Average Model Score", f"{avg_score:.4f}")
                
                with col3:
                    columns_processed = len(selected_columns)
                    st.metric("Columns Processed", columns_processed)
                
                # Show imputed counts per column
                imputed_counts_df = pd.DataFrame(list(report['imputed_counts'].items()), 
                                               columns=['Column', 'Values Imputed'])
                st.dataframe(imputed_counts_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error during imputation: {str(e)}")

def post_imputation_analysis():
    if st.session_state.imputed_data is None:
        st.warning("⚠️ Please perform imputation first in the 'ML Imputation (SVR)' section")
        return
    
    st.header("📈 Post-Imputation Analysis")
    
    original_data = st.session_state.data
    imputed_data = st.session_state.imputed_data
    report = st.session_state.imputation_report
    
    # Overall comparison
    st.subheader("🔍 Dataset Comparison Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        original_missing = original_data.isnull().sum().sum()
        st.metric("Original Missing Values", original_missing)
    
    with col2:
        imputed_missing = imputed_data.isnull().sum().sum()
        st.metric("After Imputation Missing Values", imputed_missing)
    
    with col3:
        improvement = ((original_missing - imputed_missing) / original_missing * 100) if original_missing > 0 else 0
        st.metric("Improvement", f"{improvement:.1f}%")
    
    # Model performance metrics
    if 'model_scores' in report:
        st.subheader("🎯 SVR Model Performance")
        scores_df = pd.DataFrame(list(report['model_scores'].items()), 
                                columns=['Column', 'R² Score'])
        scores_df['Performance'] = scores_df['R² Score'].apply(
            lambda x: 'Excellent' if x > 0.8 else 'Good' if x > 0.6 else 'Fair' if x > 0.4 else 'Poor'
        )
        st.dataframe(scores_df, use_container_width=True)
        
        # Performance visualization
        fig = px.bar(scores_df, x='Column', y='R² Score', 
                     title='SVR Model Performance by Column',
                     color='R² Score', color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical comparison
    st.subheader("📊 Statistical Comparison")
    
    imputed_columns = list(report['imputed_counts'].keys())
    selected_column = st.selectbox("Select column for detailed analysis", imputed_columns)
    
    if selected_column:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Data Statistics**")
            original_stats = original_data[selected_column].describe()
            st.dataframe(original_stats.to_frame(name='Original'), use_container_width=True)
        
        with col2:
            st.write("**After Imputation Statistics**")
            imputed_stats = imputed_data[selected_column].describe()
            st.dataframe(imputed_stats.to_frame(name='Imputed'), use_container_width=True)
        
        # Distribution comparison
        visualizer = ImputationVisualizer()
        distribution_fig = visualizer.create_distribution_comparison(
            original_data, imputed_data, selected_column
        )
        st.plotly_chart(distribution_fig, use_container_width=True)
        
        # Before/After scatter plot for correlations
        if len(imputed_columns) > 1:
            st.subheader("🔗 Correlation Analysis")
            other_columns = [col for col in imputed_columns if col != selected_column]
            correlation_column = st.selectbox("Compare with", other_columns)
            
            if correlation_column:
                correlation_fig = visualizer.create_correlation_comparison(
                    original_data, imputed_data, selected_column, correlation_column
                )
                st.plotly_chart(correlation_fig, use_container_width=True)
    
    # Imputation quality assessment
    st.subheader("🏆 Imputation Quality Assessment")
    
    # Create quality metrics
    quality_metrics = []
    for column in imputed_columns:
        original_col = original_data[column].dropna()
        imputed_col = imputed_data[column]
        
        # Calculate metrics
        mean_diff = abs(original_col.mean() - imputed_col.mean())
        std_diff = abs(original_col.std() - imputed_col.std())
        
        quality_metrics.append({
            'Column': column,
            'Mean Difference': mean_diff,
            'Std Difference': std_diff,
            'Values Imputed': report['imputed_counts'][column],
            'Model Score': report['model_scores'].get(column, 'N/A')
        })
    
    quality_df = pd.DataFrame(quality_metrics)
    st.dataframe(quality_df, use_container_width=True)
    
    # Missing data pattern before/after
    st.subheader("🗺️ Missing Data Pattern: Before vs After")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Imputation**")
        before_heatmap = visualizer.create_missing_data_heatmap(original_data)
        st.plotly_chart(before_heatmap, use_container_width=True)
    
    with col2:
        st.write("**After Imputation**")
        after_heatmap = visualizer.create_missing_data_heatmap(imputed_data)
        st.plotly_chart(after_heatmap, use_container_width=True)

def download_section():
    if st.session_state.imputed_data is None:
        st.warning("⚠️ No imputed data available for download. Please perform imputation first.")
        return
    
    st.header("💾 Download Results")
    
    imputed_data = st.session_state.imputed_data
    report = st.session_state.imputation_report
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Download Imputed Dataset")
        csv_buffer = io.StringIO()
        imputed_data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Imputed Data (CSV)",
            data=csv_data,
            file_name="imputed_data.csv",
            mime="text/csv",
            help="Download the dataset with imputed values"
        )
    
    with col2:
        st.subheader("📊 Download Imputation Report")
        report_text = f"""
SVR Imputation Report
====================

Imputation Summary:
{'-' * 50}
"""
        for column, count in report['imputed_counts'].items():
            score = report['model_scores'].get(column, 'N/A')
            report_text += f"\nColumn: {column}\n"
            report_text += f"Values Imputed: {count}\n"
            report_text += f"Model R² Score: {score}\n"
            report_text += f"{'-' * 30}\n"
        
        st.download_button(
            label="📥 Download Imputation Report (TXT)",
            data=report_text,
            file_name="imputation_report.txt",
            mime="text/plain",
            help="Download detailed imputation report"
        )
    
    # Data summary
    st.subheader("📋 Final Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", imputed_data.shape[0])
    with col2:
        st.metric("Total Columns", imputed_data.shape[1])
    with col3:
        remaining_missing = imputed_data.isnull().sum().sum()
        st.metric("Remaining Missing Values", remaining_missing)
    with col4:
        total_imputed = sum(report['imputed_counts'].values())
        st.metric("Total Values Imputed", total_imputed)
    
    # Preview of imputed data
    st.subheader("👀 Final Dataset Preview")
    st.dataframe(imputed_data.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
