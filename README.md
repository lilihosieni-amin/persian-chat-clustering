# Clustering 78,000 Persian User Questions Using NLP and Unsupervised Learning

This project applies NLP techniques and unsupervised clustering to analyze a large collection of Persian-language user messages from consultant chats. The goal is to extract meaningful insights by identifying frequently asked questions and recurring themes, ultimately supporting product and content development.

## 📂 Project Structure

The codebase is organized into several notebooks, each handling a key stage of the pipeline:

- **`preprocecing.ipynb`** – Cleans and normalizes raw text using the Hazm library
- **`embedding.ipynb`** – Generates vector representations of user questions using OpenAI’s `text-embedding-3-large`
- **`findOptimalK.ipynb`** – Evaluates different numbers of clusters using Elbow and Silhouette methods
- **`findClass.ipynb`** – Applies clustering algorithms (KMeans, MiniBatchKMeans, etc.) and saves results
- **`topQuestion.ipynb`** – Selects representative questions from each cluster for labeling and insight extraction

## 🧰 Tools & Technologies

- Python 3.10+
- OpenAI GPT-4o & Embeddings API
- Hazm (Persian NLP preprocessing)
- Scikit-learn
- UMAP
- Pandas, NumPy, Matplotlib

## 🔍 Problem Statement

The raw data consisted of over 23,000 full chat conversations between users and consultants. These were unstructured and noisy, making them hard to analyze. Our main goals were:

- Extract meaningful questions from messy conversations
- Group semantically similar questions
- Label and analyze recurring themes

## 🧠 Approach

1. **Question Extraction**  
   Each chat was sent to GPT-4o using a custom prompt to extract only the user’s questions.

2. **Text Preprocessing**  
   Cleaning and normalization with Hazm (removing stopwords, symbols, redundant verbs, etc.)

3. **Embedding**  
   Used `text-embedding-3-large` to generate 3,100-dimensional semantic vectors for each question.  
   Saved outputs in `.parquet` format for storage efficiency and fast I/O.

4. **Clustering**  
   After evaluating several algorithms (KMeans, MiniBatchKMeans, HDBSCAN, GMM, Birch), we selected **KMeans** with **59 clusters** as the best trade-off between interpretability and quality.

5. **Visualization**  
   Used UMAP for 2D representation of clusters.

6. **Labeling**  
   Questions were reviewed manually (from both center and edges of each cluster), and each cluster was assigned a thematic label.

## 🚧 Challenges

- **Large-scale data** (78,000+ questions)
- **Persian language preprocessing**
- **Limited hardware resources**, especially for memory-intensive clustering algorithms

## 🧭 Future Work

- Sub-clustering within large clusters to find finer-grained topics
- Training a supervised classifier based on labeled data
- Semantic search engine based on embeddings
- Building a chatbot to respond using clustered FAQs

## 📊 Sample Outputs

- 📎 Cluster assignments per question
- 📎 UMAP plots and silhouette/inertia scores

---

## 📄 License

MIT License – feel free to reuse with attribution.

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Feel free to fork the repo or open an issue.

