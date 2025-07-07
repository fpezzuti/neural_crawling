import argparse
from indexer.electra_reranker import ElectraReranker

SUPPORTED_BENCHMARKS = ["msmarco-ws", "rq"]


from utils.config import config
reranker_cfg = config.get("reranker")

VERBOSITY = reranker_cfg.get("verbosity")
INDEX_DIR = config.get("paths").get("index_dir")
RUNS_DIR = config.get("paths").get("runs_dir")
BATCH_SIZE = reranker_cfg.get("batch_size", 100000)
TOPK = reranker_cfg.get("topk", 100)
DEFAULT_MONOELECTRA_CKPT = reranker_cfg.get("checkpoint", "google/electra-base-discriminator")

def main():
    parser = argparse.ArgumentParser(description="Reranker a ranking list wiht a MonoEncoder.")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MONOELECTRA_CKPT, help = "Monoencoder's huggingface checkpoint.")
    parser.add_argument("--collection", type=str, default="cw22b", help = "Collection to be used [cw22b].")
    parser.add_argument("--benchmark", type=str, default="all", help = "Benchmark to be used for evaluation [msmarco-ws, rq] || [all].")
    parser.add_argument("--topk", type=int, default=TOPK, help = "Number of documents to rerank.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help = "Batch size.")
    parser.add_argument("--exp_name", type=str, required=True, help = "Name of the experiment (outer run directory).")
    parser.add_argument("--subexp_name", type=str, required=True, help = "Name of the sub-experiment (inner run directory).")

    args = parser.parse_args()

    print("************ START JOB ************")
    reranker = ElectraReranker(collection=args.collection, experiment_name=args.exp_name, checkpoint=args.model_name_or_path, topk=args.topk, batch_size=args.batch_size) # load reranker
    
    if args.benchmark != "all":
        benchmarks = [args.benchmark]
    else:
        benchmarks = SUPPORTED_BENCHMARKS
    
    for benchmark_name in benchmarks: # perform reranking for each specified benchmark
        run_fpath =  f"{RUNS_DIR}{args.exp_name}/{args.subexp_name}/{benchmark_name}/{args.subexp_name}.tsv"
      
        print(f"Starting to rerank run={run_fpath}, for benchmark {benchmark_name}")
        
        output_run_fpath =  f"{RUNS_DIR}{args.exp_name}/{args.subexp_name}/{benchmark_name}/reranked-{args.subexp_name}.tsv"
        reranker.rerank(benchmark_name=benchmark_name, run_fpath=run_fpath, output_run_fpath=output_run_fpath, save_results=True)

        print("Done reranking for benchmark=", benchmark_name)
            
    print("************ END JOB ************")

if __name__ == "__main__":
    main()