# All RAG Techniques: A Simpler, Hands-On Approach(korea version)

이 저장소는 [FareedKhan-dev/all-rag-techniques](https://github.com/FareedKhan-dev/all-rag-techniques) 프로젝트의 내용을 **한국어로 번역한 Jupyter Notebook 모음집**입니다.

---

This repository takes a clear, hands-on approach to **Retrieval-Augmented Generation (RAG)**, breaking down advanced techniques into straightforward, understandable implementations. Instead of relying on frameworks like `LangChain` or `FAISS`, everything here is built using familiar Python libraries `openai`, `numpy`, `matplotlib`, and a few others.

The goal is simple: provide code that is readable, modifiable, and educational. By focusing on the fundamentals, this project helps demystify RAG and makes it easier to understand how it really works.

## 🚀 What's Inside?

This repository contains a collection of Jupyter Notebooks, each focusing on a specific RAG technique.  Each notebook provides:

- A concise explanation of the technique.
- A step-by-step implementation from scratch.
- Clear code examples with inline comments.
- Evaluations and comparisons to demonstrate the technique's effectiveness.
- Visualization to visualize the results.

Here's a glimpse of the techniques covered:

| Notebook                                      | Description                                                                                                                                                         |번역 상태|
| :-------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |---------------------------------------|
| [1. Simple RAG](01_simple_rag.ipynb)           | A basic RAG implementation.  A great starting point!                                                                                                       |✅ 완료|
| [2. Semantic Chunking](02_semantic_chunking.ipynb) | Splits text based on semantic similarity for more meaningful chunks.                                                                                           |✅ 완료|
| [3. Chunk Size Selector](03_chunk_size_selector.ipynb) | Explores the impact of different chunk sizes on retrieval performance.                                                                                    |✅ 완료|
| [4. Context Enriched RAG](04_context_enriched_rag.ipynb) | Retrieves neighboring chunks to provide more context.                                                                                                     |✅ 완료|
| [5. Contextual Chunk Headers](05_contextual_chunk_headers_rag.ipynb) | Prepends descriptive headers to each chunk before embedding.                                                                                                |✅ 완료|
| [6. Document Augmentation RAG](06_doc_augmentation_rag.ipynb) | Generates questions from text chunks to augment the retrieval process.                                                                                           |✅ 완료|
| [7. Query Transform](07_query_transform.ipynb)   | Rewrites, expands, or decomposes queries to improve retrieval.  Includes **Step-back Prompting** and **Sub-query Decomposition**.                                      |✅ 완료|
| [8. Reranker](08_reranker.ipynb)               | Re-ranks initially retrieved results using an LLM for better relevance.                                                                                       |✅ 완료|
| [9. RSE](09_rse.ipynb)                         | Relevant Segment Extraction:  Identifies and reconstructs continuous segments of text, preserving context.                                                   |✅ 완료|
| [10. Contextual Compression](10_contextual_compression.ipynb) | Implements contextual compression to filter and compress retrieved chunks, maximizing relevant information.                                                 |✅ 완료|
| [11. Feedback Loop RAG](11_feedback_loop_rag.ipynb) | Incorporates user feedback to learn and improve RAG system over time.                                                                                      |✅ 완료|
| [12. Adaptive RAG](12_adaptive_rag.ipynb)     | Dynamically selects the best retrieval strategy based on query type.                                                                                          |✅ 완료|
| [13. Self RAG](13_self_rag.ipynb)             | Implements Self-RAG, dynamically decides when and how to retrieve, evaluates relevance, and assesses support and utility.                                        |🔄 진행 중|
| [14. Proposition Chunking](14_proposition_chunking.ipynb) | Breaks down documents into atomic, factual statements for precise retrieval.                                                                                      |🔄 진행 중
| [15. Multimodel RAG](15_multimodel_rag.ipynb)   | Combines text and images for retrieval, generating captions for images using LLaVA.                                                                  |🔄 진행 중
| [16. Fusion RAG](16_fusion_rag.ipynb)         | Combines vector search with keyword-based (BM25) retrieval for improved results.                                                                                |🔄 진행 중
| [17. Graph RAG](17_graph_rag.ipynb)           | Organizes knowledge as a graph, enabling traversal of related concepts.                                                                                        |🔄 진행 중
| [18. Hierarchy RAG](18_hierarchy_rag.ipynb)        | Builds hierarchical indices (summaries + detailed chunks) for efficient retrieval.                                                                                   |🔄 진행 중
| [19. HyDE RAG](19_HyDE_rag.ipynb)             | Uses Hypothetical Document Embeddings to improve semantic matching.                                                                                              |🔄 진행 중
| [20. CRAG](20_crag.ipynb)                     | Corrective RAG: Dynamically evaluates retrieval quality and uses web search as a fallback.                                                                           |🔄 진행 중
| [21. Rag with RL](21_rag_with_rl.ipynb)                     | Maximize the reward of the RAG model using Reinforcement Learning.                                                                           |🔄 진행 중
| [Best RAG Finder](best_rag_finder.ipynb)     | Finds the best RAG technique for a given query using Simple RAG + Reranker + Query Rewrite.                                                                        |🔄 진행 중
| [22. Big Data with Knowledge Graphs](22_Big_data_with_KG.ipynb) | Handles large datasets using Knowledge Graphs.                                                                                                                     |🔄 진행 중

## 🗂️ Repository Structure

```
all-rag-techniques-kor/
├── README.md                          <- You are here!
├── 01_simple_rag.ipynb
├── 02_semantic_chunking.ipynb
├── 03_chunk_size_selector.ipynb
├── 04_context_enriched_rag.ipynb
├── 05_contextual_chunk_headers_rag.ipynb
├── 06_doc_augmentation_rag.ipynb
├── 07_query_transform.ipynb
├── 08_reranker.ipynb
├── 09_rse.ipynb
├── 10_contextual_compression.ipynb
├── 11_feedback_loop_rag.ipynb
├── 12_adaptive_rag.ipynb
├── 13_self_rag.ipynb
├── 14_proposition_chunking.ipynb
├── 15_multimodel_rag.ipynb
├── 16_fusion_rag.ipynb
├── 17_graph_rag.ipynb
├── 18_hierarchy_rag.ipynb
├── 19_HyDE_rag.ipynb
├── 20_crag.ipynb
├── 21_rag_with_rl.ipynb
├── 22_big_data_with_KG.ipynb
├── best_rag_finder.ipynb
├── requirements.txt                   <- Python dependencies
└── data/
    └── val.json                       <- Sample validation data (queries and answers)
    └── AI_Information.pdf             <- A sample PDF document for testing.
    └── attention_is_all_you_need.pdf  <- A sample PDF document for testing (for Multi-Modal RAG).
```

## 🛠️ Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/FareedKhan-dev/all-rag-techniques.git
    cd all-rag-techniques
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your OpenAI API key:**

    - Obtain an API key from [Nebius AI](https://studio.nebius.com/).
    - Set the API key as an environment variable:

        ```bash
        export API_KEY='YOUR_OPENAI_API_KEY'
        ```

        or

        ```bash
        setx API_KEY "YOUR_OPENAI_API_KEY"  # On Windows
        ```

        or, within your Python script/notebook:

        ```python
        import os
        os.environ["API_KEY"] = "YOUR_OPENAI_API_KEY"
        ```

        or, use `.env` file
        ```
        API_KEY = "YOUR_OPENAI_API_KEY"
        ```

4. **Run the notebooks:**

    Open any of the Jupyter Notebooks (`.ipynb` files) using Jupyter Notebook or JupyterLab.  Each notebook is self-contained and can be run independently.  The notebooks are designed to be executed sequentially within each file.

    **Note:** The `data/AI_Information.pdf` file provides a sample document for testing. You can replace it with your own PDF.  The `data/val.json` file contains sample queries and ideal answers for evaluation.
    The 'attention_is_all_you_need.pdf' is for testing Multi-Modal RAG Notebook.

## 💡 Core Concepts

- **Embeddings:**  Numerical representations of text that capture semantic meaning.  We use Nebius AI's embedding API and, in many notebooks, also the `BAAI/bge-en-icl` embedding model.

- **Vector Store:**  A simple database to store and search embeddings.  We create our own `SimpleVectorStore` class using NumPy for efficient similarity calculations.

- **Cosine Similarity:**  A measure of similarity between two vectors.  Higher values indicate greater similarity.

- **Chunking:**  Dividing text into smaller, manageable pieces.  We explore various chunking strategies.

- **Retrieval:** The process of finding the most relevant text chunks for a given query.

- **Generation:**  Using a Large Language Model (LLM) to create a response based on the retrieved context and the user's query.  We use the `meta-llama/Llama-3.2-3B-Instruct` model via Nebius AI's API.

- **Evaluation:**  Assessing the quality of the RAG system's responses, often by comparing them to a reference answer or using an LLM to score relevance.

## 🤝 Contributing

Contributions are welcome!
