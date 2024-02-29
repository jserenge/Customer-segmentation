import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

url = 'https://raw.githubusercontent.com/jserenge/Customer-segmentation/main/customer_data.csv'
df = pd.read_csv(url)
df_clean=df.drop(columns='Sell_to_Customer_No')


st.write("""
# ***Customer Segmentation System***

The app will be used to segment a customer and assign them to a pre-defined profile!
""")

# Define the pipeline
pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('pca', PCA(n_components=0.9, svd_solver='full')),
    ('clustering', KMeans(n_clusters=4, random_state=0))
])

# Select specific features to be used in the model
selected_features = ['Days_lastpurchase', 'Total_transactions', 'Total_purchased', 'total_spend', 'Number_productlines', 'RFM_Score']

# Filter the DataFrame to include only the selected features
df_selected_features = df_clean[selected_features]

# Fit and transform the data using the pipeline
pipeline.fit(df_selected_features)
clusters = pipeline.predict(df_selected_features)

# Add the clusters to the DataFrame
df["Clusters"] = clusters
df_clean["Clusters"]=clusters

st.header("The Different Customer Segments")

scaler = StandardScaler()

# Standardize the DataFrame
df_standardized = scaler.fit_transform(df_clean.drop(columns=['Clusters'], axis=1))

# Create a new dataframe with standardized values and add the 'Clusters' column back
df_standardized = pd.DataFrame(df_standardized, columns=df_clean.columns[:-1], index=df.index)
df_standardized['Clusters'] = df_clean['Clusters']

# Calculate the centroids of each cluster
cluster_centroids = df_standardized.groupby('Clusters').mean()

# Function to create a radar chart
def create_radar_chart(ax, angles, data, color, cluster):
    # Plot the data and fill the area
    ax.fill(angles, data, color=color, alpha=0.4)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')

    # Add a title
    ax.set_title(f'Cluster {cluster}', size=20, color=color, y=1.1)

# Set data
labels=np.array(cluster_centroids.columns)
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is circular, so we need to "complete the loop" and append the start to the end
labels = np.concatenate((labels, [labels[0]]))
angles += angles[:1]

# Initialize the figure
fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(polar=True), nrows=1, ncols=len(df['Clusters'].unique()))

# Create radar chart for each cluster
colors = ['b', 'r', 'g', 'y']  # Define as many colors as you have clusters
for i, color in enumerate(colors):
    data = cluster_centroids.loc[i].tolist()
    data += data[:1]  # Complete the loop
    create_radar_chart(ax[i], angles, data, color, i)

    # Add input data
    ax[i].set_xticks(angles[:-1])
    ax[i].set_xticklabels(labels[:-1])

    # Add a grid
    ax[i].grid(color='grey', linewidth=0.5)

# Display the plot
plt.tight_layout()
st.pyplot(plt)

# Display customer profile statistics in a table
selected_customer = st.selectbox("Select Customer ID", df['Sell_to_Customer_No'].unique())
customer_data = df[df['Sell_to_Customer_No'] == selected_customer]
if not customer_data.empty:
    st.subheader(f"Customer Profile for {selected_customer}")
    
    # Calculate summary statistics
    summary_stats = customer_data.describe().loc[['mean', '50%', 'min', 'max', 'std']].T
    summary_stats.columns = ['Mean', 'Median', 'Min', 'Max', 'Std']
    
    # Display summary statistics table
    st.table(summary_stats)

else:
    st.write("No data available for the selected customer.")


# Display the table for selected cluster
st.subheader("Select the cluster to display")

# Select cluster
cluster = st.selectbox('Select cluster', df['Clusters'].unique())
selected_cluster_df = df[df['Clusters'] == cluster][['Sell_to_Customer_No', 'RFM_Score','Total_transactions','Total_purchased','total_spend', 'Clusters']].head(num_rows_to_display)
