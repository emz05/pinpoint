# **Evaluation of Composed Image Retrieval with Multi Image Queries**

This notebook evaluates five strategies for multi image Composed Image Retrieval on the [PinPoint benchmark](https://github.com/pinterest/pinpoint-dataset) (CVPR 2026).

In multi image CIR, a query is formed from two reference images and a text instruction. The objective is to retrieve relevant images from a large corpus.

## **Strategies**
| Method | Key idea |
|---|---|
| **Text Only (Baseline)** | Ignore reference images and database retrieval is dependent only on text instruction |
| **Mean Pooling** | Averages numerical vectors of both reference images equally and includes text instruction |
| **Max Pooling** | Compares both reference images and selects highest activation for each feature. Text instruction is combined with surfaced traits |
| **Weighted Pooling** | Custom attention mechanism that maps images and texts into same embedding space through OpenCLIP. Through comparison, higher weight is assigned to more relevant image with text instruction. |
| **LLM Translation** | Uses image captioning model (BLIP) to translate visual contents of references images into text. Appends user instruction with captions for text based retrieval. |