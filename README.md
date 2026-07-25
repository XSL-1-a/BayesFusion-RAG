# BayesFusion-RAG

BayesFusion-RAG is a Bayesian multi-expert retrieval framework for industrial and technical document question answering. It combines sparse, dense, multimodal, and entity-aware retrievers through query-adaptive fusion, then generates traceable answers with citation parsing and verification.

The repository includes hierarchical indexing, retrieval fusion, answer generation, evaluation workflows, and local demonstration entry points.

## Highlights

- Multi-expert retrieval with BM25, dense, multimodal, and entity-aware experts.
- Query-adaptive fusion and learned gating strategies.
- Hierarchical indexing at document, section, chunk, image, and entity levels.
- Traceable answer generation with citation parsing and verification.
- BEIR, private-document, ablation, and statistical-analysis scripts.
- Local and web demos for retrieval and document question answering.

## Repository Structure

```text
BayesFusion-RAG/
|-- src/                  Core retrieval, fusion, entity, generation, and evaluation modules
|-- configs/              Local, default, and provider-specific configurations
|-- demo/                 Interactive demonstrations
|-- scripts/              Dataset preparation and experiment entry points
|-- experiments/scripts/  Reproducibility scripts
|-- results/final_figures/ Selected paper figures and tables
|-- simple_retrieval.py   Lightweight retrieval demo
|-- interactive_retrieval.py
|-- simple_web.py
`-- requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

For API-backed generation, create a local `.env` file. Never commit real credentials.

```text
OPENAI_API_KEY=your_api_key
```

## Quick Start

Run the lightweight retrieval example:

```bash
python simple_retrieval.py
```

Start the web demonstration:

```bash
python simple_web.py
```

Prepare public datasets and run experiments:

```bash
python scripts/prepare_public_datasets.py
python scripts/run_beir_experiments.py
```

## Experimental Integrity

The repository contains both full benchmark workflows and synthetic fallback data used for pipeline testing when a public dataset cannot be downloaded. Synthetic outputs are not evidence of benchmark performance. Paper-level results should be reproduced from downloaded benchmark data with the exact configuration and random seed recorded in the experiment output.

Selected figures are included for convenient inspection. Large datasets, embeddings, model weights, private PDFs, and API credentials are intentionally excluded.

## Implementation

The repository implements the retrieval and fusion pipeline, hierarchical document indexing, entity-aware and multimodal retrieval, traceable answer generation, citation verification, and evaluation workflows.

## Usage Note

This code is provided for portfolio review and technical discussion. Please contact the author before redistributing it or using unpublished results.
