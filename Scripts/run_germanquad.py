# run_germanquad.py
# GermanQuAD reproduction — mirrors Run_model_sum_weights_miracl.py
# but loads dataset from BEIR instead of HuggingFace datasets.
# Used for Table 2 (Language Transfer) in Braga et al. 2025.

from utils import seed_everything, TaskVectorT5, RerankT5
import logging
import os
import click
import torch
import pathlib
import beir
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from ranx import Qrels, Run, compare, fuse


def applyBM25(test_corpus, test_queries, name='test'):
    hostname = "localhost"
    index_name = "germanquad"
    initialize = True
    number_of_shards = 1

    model = BM25(
        index_name=index_name,
        hostname=hostname,
        language="german",
        initialize=initialize,
        number_of_shards=number_of_shards
    )
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(test_corpus, test_queries)
    return results


def apply_reranking(device, model_base_path, model_vector_minus_path,
                    model_vector_plus_path, test_corpus, test_queries,
                    results_bm25_test, coef):
    # GermanQuAD uses ▁no / ▁yes tokens (not ▁false / ▁true)
    token_false = '▁no'
    token_true  = '▁yes'

    cross_encoder_model = MonoT5(
        model_base_path,
        token_false=token_false,
        token_true=token_true
    )

    task_vector = TaskVectorT5(model_vector_minus_path, model_vector_plus_path)
    sum_model, new_state_dict = task_vector.apply_to(model_base_path, scaling_coef=coef)
    sum_model.to(device)
    cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)

    reranker = RerankT5(cross_encoder_model, batch_size=128)
    results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)
    return results


def evaluate_baseline(device, model_path, token_false, token_true,
                      test_corpus, test_queries, results_bm25_test):
    cross_encoder_model = MonoT5(
        model_path,
        token_false=token_false,
        token_true=token_true
    )
    reranker = RerankT5(cross_encoder_model, batch_size=128)
    results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)
    return results


@click.command()
@click.option("--model_base_path",        type=str, required=True,  help="Theta_T: IR fine-tuned model (e.g. unicamp-dl/mt5-base-mmarco-v2)")
@click.option("--model_vector_minus_path", type=str, required=True, help="Theta_0: Base pretrained model (e.g. google/mt5-base)")
@click.option("--model_vector_plus_path",  type=str, required=True, help="Theta_D: Domain model (e.g. airKlizz/mt5-base-wikinewssum-german)")
@click.option("--device",     type=str, default="cuda:1")
@click.option("--alfa",       type=float, required=True, help="Task vector scaling coefficient (0=Theta_T, 0.5=recommended, 1=full vector)")
@click.option("--model_name", type=str, default="mt5_base")
@click.option("--seed",       type=int, default=42)
@click.option("--output_folder", type=str, default="./Risultati")
def main(model_base_path, model_vector_minus_path, model_vector_plus_path,
         device, alfa, model_name, seed, output_folder):

    if seed:
        seed_everything(seed)

    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger('main')
    torch.cuda.set_device(device)

    # ── Load GermanQuAD from BEIR ──────────────────────────────────────────
    logger.info('Loading GermanQuAD dataset from BEIR...')
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/germanquad.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")
    logger.info(f'Corpus: {len(test_corpus)} docs | Queries: {len(test_queries)}')

    # ── BM25 retrieval ─────────────────────────────────────────────────────
    logger.info('Running BM25...')
    results_bm25_test = applyBM25(test_corpus, test_queries)
    bm25_run_test = Run(results_bm25_test, name='BM25')

    test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
    test_qrels_obj = Qrels(test_qrels_correct)

    token_false = '▁no'
    token_true  = '▁yes'

    lista_results_combined = []

    # ── λ=0 : Theta_T alone (no task vector) ──────────────────────────────
    logger.info('Running Theta_T (lambda=0)...')
    results_theta_t = apply_reranking(
        device, model_base_path, model_vector_minus_path,
        model_vector_plus_path, test_corpus, test_queries,
        results_bm25_test, coef=0.0
    )
    run_theta_t = Run(results_theta_t, name=f'BM25 +{model_name}-sum-lambda-0')
    lista_results_combined.append(run_theta_t)

    # ── λ=alfa : Task Arithmetic ───────────────────────────────────────────
    logger.info(f'Running Task Arithmetic (lambda={alfa})...')
    results_ta = apply_reranking(
        device, model_base_path, model_vector_minus_path,
        model_vector_plus_path, test_corpus, test_queries,
        results_bm25_test, coef=alfa
    )
    run_ta = Run(results_ta, name=f'BM25 +{model_name}-sum-lambda-{alfa}')
    lista_results_combined.append(run_ta)

    # ── Baselines: Theta_0 and Theta_D ────────────────────────────────────
    for path, label in [
        (model_vector_minus_path, f'BM25 +{model_name}_original_version'),
        (model_vector_plus_path,  f'BM25 +{model_name}_domain_specific'),
    ]:
        logger.info(f'Running baseline: {label}...')
        results = evaluate_baseline(
            device, path, token_false, token_true,
            test_corpus, test_queries, results_bm25_test
        )
        lista_results_combined.append(Run(results, name=label))

    # ── Final comparison ───────────────────────────────────────────────────
    lista_results_combined.append(bm25_run_test)

    report = compare(
        qrels=test_qrels_obj,
        runs=lista_results_combined,
        metrics=['ndcg@3', 'ndcg@10', 'precision@10', 'recall@100', 'map@100'],
        max_p=0.01
    )

    logger.info(f'\n{report}')
    print(report)


# Import MonoT5 after click definitions to avoid TF noise before args parsed
from monot5 import MonoT5

if __name__ == '__main__':
    main()
