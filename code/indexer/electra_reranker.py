import sys, os
from pandas import DataFrame
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *
import pyterrier_dr

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from indexer.dataset import RankingListDataframe
from utils.config import config
from utils.component import Component

reranker_cfg = config.get("reranker")

VERBOSITY = reranker_cfg.get("verbosity")

INDEX_DIR = config.get("paths").get("index_dir")
RUNS_DIR = config.get("paths").get("runs_dir")
BATCH_SIZE = reranker_cfg.get("batch_size", 100000)
TOPK = reranker_cfg.get("topk", 100)
DEFAULT_MONOELECTRA_CKPT = reranker_cfg.get("checkpoint", "crystina-z/monoELECTRA_LCE_nneg31")
    
class ElectraReranker(Component):
   
    def __init__(self, collection, experiment_name, checkpoint=DEFAULT_MONOELECTRA_CKPT, verbosity=VERBOSITY, batch_size = BATCH_SIZE, topk=TOPK) -> None:
        """
            Initialise the reranker component:
            - collection is the string identifier of the corpus of documents to be indexed
            - experiment_name is the name of the experiment (the crawler)
            - checkpoint is the huggingface checkpoint of the reranker model
            - verbosity is the integer verbosity level
            - batch_size is the batch size to be used by the reranker
            - topk is the number of documents to be reranked
        """
        Component.__init__(self, verbose=True, verbosity=verbosity)

        self.component_name = "INDEXER"

        self.log(f"Initialising reranker for topk={topk} documents of experiment={experiment_name}.", 1)
       
        self.runs_dir = f"{RUNS_DIR}/{experiment_name}"
        self.experiment_name = experiment_name
        self.topk = topk
        self.batch_size = batch_size
        self.collection = collection

        self.log("Initialising reranker model.", 2)
        self.reranker = pyterrier_dr.ElectraScorer(model_name=checkpoint, batch_size=batch_size, text_field='text', verbose=True, device='cuda')
        self.log("Reranker model initialised.", 1)

        self.pipeline = pt.text.sliding('text', prepend_title=False) >> self.reranker >> pt.text.max_passage()
        self.log("Pipeline created.", 1)

    def __load_rankings(self, ranking_fpath: str, benchmark_name: str) -> DataFrame:
        """
            Given a specified file path and the name of the query set (benchmark_name), load the ranking list to be reranked and return it as a DataFrame.
        """
        # load ranking list + doc texts + query texts
        results_df = RankingListDataframe(collection=self.collection, run_fpath=ranking_fpath, topk=self.topk).get_rankings(benchmark_name)
        return results_df

    def rerank(self, benchmark_name: str, run_fpath: str, output_run_fpath: str, save_results: bool = True) -> None:
        """
            Rerank the top k documents from the ranking list specified in run_fpath, for the benchmark specified by benchmark_name."""
        rankings = self.__load_rankings(ranking_fpath=run_fpath, benchmark_name=benchmark_name)
        results = self.pipeline(rankings)

        print(f"Done reranking the top {self.topk} documents.")

        if save_results:
            self.__save_results(results, output_run_fpath)

        return results
    
    def __save_results(self, run_df: DataFrame, output_run_fpath: str) -> str:
        """
            save results stored in the DataFrame in trec format
        """
        output_dir = os.path.dirname(output_run_fpath)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.log(f"Saving results to file={output_run_fpath}.", 2)

        # convert to trec format
        run_df.columns=["query_id", "score", "experiment", "text", "query", "doc_id", 'rank']
        run_df.drop(columns=['text', 'query'], inplace=True)
        run_df["Q0"] = 0
        run_df = run_df[["query_id", "Q0", "doc_id", "rank", "score", "experiment"]]
        run_df.sort_values(by=["query_id", "rank"], inplace=True)
        run_df.reset_index(drop=True, inplace=True)

        # save to csv
        run_df.to_csv(output_run_fpath, sep="\t", header=False, index=False)

        print(f"Results saved in file={output_run_fpath}.", 1)
        return output_run_fpath