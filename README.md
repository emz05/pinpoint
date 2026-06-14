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


#### Text Only (baseline)
Encodes only the text instruction and retrieves directly. No image information is used. Serves as the lower bound for image conditioning value.

#### Mean Pooling
```python
res = (emb1 + emb2) / 2
query = 0.8 * text_emb + 0.2 * normalize(res)
```
Equally weights both reference images. The fixed α = 0.8 prioritizes the text instruction over the visual aggregate.

#### Max Pooling
```python
res = np.maximum(emb1, emb2)
query = 0.8 * text_emb + 0.2 * normalize(res)
```
Takes the element-wise maximum across both image embeddings. The intent is to preserve the most activated visual features from either image rather than blurring them through averaging.

#### Weighted Pooling
```python
w1 = |dot(emb1, text_emb)| / (|dot(emb1, text_emb)| + |dot(emb2, text_emb)|)
w2 = 1 - w1
res = w1 * emb1 + w2 * emb2
query = 0.8 * text_emb + 0.2 * normalize(res)
```
Weights each reference image by its semantic alignment with the text instruction in CLIP space. The image more relevant to the query's intent receives a proportionally higher weight. This approximates a lightweight attention mechanism without any additional learned parameters.

#### BLIP Captioning (LLM)
Passes each reference image through `blip-image-captioning-large` to generate a natural language description. The two captions are joined and prepended to the text instruction:

```
"{instruction}. Must match this visual style: {caption1} mixed with {caption2}"
```

The combined prompt is encoded with the CLIP text encoder and used for retrieval. Reference image embeddings are not used at all. Captions are cached per image signature to avoid redundant inference.


