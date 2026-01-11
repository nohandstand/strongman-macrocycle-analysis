# Strongman Macrocycle Analysis

Goal: Use YouTube metadata + transcripts to infer an athlete's yearly training macrocycle (base/build/peak/deload) and event emphasis over time.

## Pipeline
1. Pull video metadata from YouTube Data API
2. Pull captions/transcripts where available
3. NLP labeling (Hugging Face) for training phase + event/lift tags
4. Aggregate to monthly/weekly rollups to infer macrocycle
5. Visualize in Power BI

## Tech
- Python
- YouTube Data API
- Hugging Face transformers
- (Optional) Power BI
