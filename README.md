# Neural Prioritisation for Web Crawling

This repository contains the source code used for the experiments presented in the paper "Neural Prioritisation for Web Crawling" by Francesca Pezzuti, Sean MacAvaney and Nicola Tonellotto, published at ICTIR2025 - [PDF](https://dl.acm.org/doi/10.1145/3731120.3744597).

## Usage

### Installation
You can install the requirements using pip: 
```
pip install -r requirements.txt
```

### Supported datasets

Web collection:
- ClueWeb22-B (eng):

Query sets:
- MSM-WS (MS MARCO Web Search): [Link to the dataset](https://github.com/microsoft/MS-MARCO-Web-Search)
- RQ (Researchy Questions): [Link to the dataset](https://huggingface.co/datasets/corbyrosset/researchy_questions)

### Pre-processing
To preprocess the **msm-ws** query set:
1. make sure that queries are stored under "/data/queries/msmarco-ws/msmarco-ws-queries.tsv"
2. make sure that qrels are stored under "./../data/qrels/msmarco-ws/cleaned-msmarco-ws-qrels.tsv"


Then, run:
```bash
python preproc_querysets.py
```

To preprocess ClueWeb22-B run:
``` bash
python preproc_cw22b.py
```

### Crawling

#### BreadthFirstSearch (BFS)
To crawl with BFS, run the following command using the default config file.
```bash
python crawl.py --max_pages -1 --verbosity 1 --exp_name v1 --frontier_type bfs
```

#### DepthFirstSearch (DFS)
To crawl with DFS, run the following command using the default config file.
```bash
python crawl.py --max_pages -1 --verbosity 1 --exp_name v1 --frontier_type dfs
```

#### QFirst
To crawl with QFirst, run the following command using the default config file.
```bash
python crawl.py --max_pages -1 --verbosity 1 --exp_name first --frontier_type quality
```

#### QOracle
To crawl with QOracle, run the following command using the default config file.
```bash
python crawl.py --max_pages -1 --verbosity 1 --exp_name v1 --frontier_type oracle-quality
```

#### QMin
To crawl with QMin, run the following command using the default config file.
```bash
python crawl.py --max_pages -1 --verbosity 1 --exp_name min --frontier_type quality --updates_enabled 1
```

### Ranking with BM25
To index and rank documents with BM25 the set of documents crawled up to time t=limit by a crawler whose experimental name is 'crawler' use: 
```bash
python index.py --periodic 1 --limit 2_500_000 --exp_name crawler --benchmark msmarco-ws
```

By default, this python script automatically retrieves the top k scoring documents for queries for the specified query set and writes the results in the `runs` directory; however, this option can be disabled by launching the script with `--evaluate False`.

### Re-ranking with MonoElectra
To re-rank documents with MonoElectra:

```bash
CUDA_VISIBLE_DEVICES=0 python rerank.py --exp_name crawler --subexp_name limit_2500000 --benchmark msmarco-ws
```

## Evaluation
To evaluate crawling and retrieval effectiveness, use the Jupyter Notebook `plot-metrics.ipynb`.
