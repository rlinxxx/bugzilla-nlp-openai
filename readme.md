### 🐞 Bug Report Analysis with OpenAI

This project leverages OpenAI's language models and machine learning techniques to process, analyze, fine-tune, and cluster Mozilla Firefox bug reports using their Summary and Description text vectorized into embeddings.

### 📊 Dataset

The project uses a dataset of resolved Mozilla Firefox bug reports. It includes:

- <code>summary:</code> A short sentence describing the bug.
- <code>description:</code> A longer, more detailed comment.
- <code>labels:</code> Classification targets (i.e., Priority).

### 📁 Project Structure

<b><u>Notebooks<b><u>

| Notebook | Brief Description |
|----------|----------|
| 01_request_data.ipynb | Download bug reports data from Bugzilla |
| 02_embeddings.ipynb | Generates embeddings for bug report texts using OpenAI models and saves them for downstream tasks like classification, zero-shot, clustering and semantic search |
| 03_classification.ipynb | Trains a traditional ML classifier (Random Forest) using TF-IDF features and OpenAI embeddings to classify bug reports |
| 04_zero_shot.ipynb | Implements zero-shot classification using OpenAI’s GPT models (no task-specific training required) |
| 05_fine_tuning.ipynb | Fine-tunes a GPT model using labeled bug report examples to improve classification accuracy on domain-specific tasks |
| 06_clustering.ipynb | Cluster bug report comments using KMeans++ and visualize with t-SNE |
| 07_questions.ipynb | Calculate semantic similarity to rank the text in searches and make questions about the bug reports |

### 🚀 Project Workflow

<b><u> 1. Request and Prepare Data <b><u>

In <code>01_request_data.ipynb</code>:

- Collect bug reports and associated meta data
- Clean and organize the data into a structured format suitable for analysis
  
<b><u> 2. Generating Semantic Embeddings <b><u>

In <code>02_embeddings.ipynb</code>:

- Filter and preprocess the text
- Converts bug report texts (summary + description) into high-dimensional semantic embeddings using OpenAI’s API
- Saves the generated embeddings along with metadata for future use in tasks like semantic search or clustering

<b><u> 3. Traditional Supervised Classification <b><u>

In <code>03_classification.ipynb</code>:

- Vectorize the text using TF-IDF and train a RandomForestClassifier to predict bug categories
- Evaluate the model using common classification metrics (accuracy, F1, precision, recall) and visual tools like PR and AUC-ROC curves

<b><u> 4. Zero-Shot Learning with GPT <b><u>

In <code>04_zero_shot.ipynb</code>:

- Use OpenAI’s GPT model to classify bug reports into categories without any fine-tuning or training
- Demonstrate the power of zero-shot learning by comparing predictions with ground truth labels and evaluating performance

<b><u> 5. Fine-Tune GPT <b><u>

In <code>05_fine_tuning.ipynb</code>:

- Load and inspect the preprocessed bug report data
- Format and split the data for fine-tuning
- Train a fine-tuned version of gpt-o-mini on the labeled reports
- Evaluate the model against a non-fine-tuned baseline using metrics like accuracy, precision, recall and F1-score

<b><u> 6. Cluster and Visualize <b><u>

In <code>06_clustering.ipynb</code>:

- Apply KMeans clustering to identify similar bug report groups
- Reduce dimensions using t-SNE for 2D visualization
- Plot the clusters to explore natural groupings in the data

<b><u> 7. Semantic Search and Question and Answering <b><u>

In <code>07_questions.ipynb</code>:

- Given a user question, generate an embedding for the query from the OpenAI API
- Using the embeddings, rank the text sections by relevance to the query
- Insert the question and the most relevant sections into a message to GPT
- Return GPT's answer

### 🛠️ Technologies Used

- Python (Pandas, NumPy, scikit-learn, matplotlib)
- OpenAI GPT API
- Jupyter Notebooks
- t-SNE and KMeans for clustering and visualization

### 📦 Requirements

Install dependencies using pip:

<code>pip install -r requirements.txt</code>

### 📌 Future Enhancements

- Integrate vector databases (e.g., Pinecone or FAISS) for scalable search
- Expand fine-tuning with more categories or larger datasets
- Add a frontend interface for querying or classifying new bug reports