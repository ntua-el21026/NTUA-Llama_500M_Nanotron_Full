import argparse
import itertools
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer

# --- CUSTOM CLASS ΓΙΑ ΝΑ ΛΥΣΟΥΜΕ ΤΟ ΠΡΟΒΛΗΜΑ ΤΟΥ SKIP ---
class SkippableHFReader(HuggingFaceDatasetReader):
    def __init__(self, dataset, skip=0, min_int_score=None, int_score_key="int_score", **kwargs):
        self.skip_n = skip
        self.min_int_score = min_int_score
        self.int_score_key = int_score_key
        super().__init__(dataset, **kwargs)

        # Αφαιρούμε το skip από τα kwargs πριν το στείλουμε στον πατέρα

    def run(self, data=None, rank=0, world_size=1):
        iterator = super().run(data, rank, world_size)

        if self.skip_n > 0:
            print(f"--- INFO: Skipping first {self.skip_n} documents... ---")
            iterator = itertools.islice(iterator, self.skip_n, None)

        # --- NEW: filter by int_score (keep only >= min_int_score) ---
        if self.min_int_score is not None:
            def _keep(doc):
                md = getattr(doc, "metadata", None)
                if not isinstance(md, dict):
                    return False
                v = md.get(self.int_score_key, None)
                return isinstance(v, int) and v >= self.min_int_score

            iterator = (doc for doc in iterator if _keep(doc))
        # ------------------------------------------------------------

        return iterator

# --------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument("--tokenizer-name-or-path", type=str, required=True)
    group.add_argument("--eos-token", type=str, default=None)

    group = parser.add_argument_group(title="Output data")
    group.add_argument("--output-folder", type=str, required=True)

    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument("--logging-dir", type=str, default=None)
    group.add_argument("--n-tasks", type=int, default=8)
    group.add_argument("--min-int-score", type=int, default=None)
    group.add_argument("--int-score-key", type=str, default="int_score")


    # Παράμετροι Split
    group.add_argument("--limit", type=int, default=-1)
    group.add_argument("--skip", type=int, default=0)

    sp = parser.add_subparsers(dest="readers", required=True)

    p1 = sp.add_parser(name="hf")
    p1.add_argument("--dataset", type=str, required=True)
    p1.add_argument("--column", type=str, default="text")
    p1.add_argument("--split", type=str, default="train")
    p1.add_argument("--name", type=str, default=None)

    p2 = sp.add_parser(name="jsonl")
    p2.add_argument("--dataset", type=str, required=True)
    p2.add_argument("--column", type=str, default="text")
    p2.add_argument("--glob-pattern", type=str, default=None)

    return parser.parse_args()


def main(args):
    if args.readers == "hf":
        # Χρησιμοποιούμε τον δικό μας Reader
        # ΠΡΟΣΟΧΗ: Περνάμε το args.split αυτούσιο (π.χ. "train"), ΟΧΙ πειραγμένο
        datatrove_reader = SkippableHFReader(
            dataset=args.dataset,
            text_key=args.column,
            dataset_options={"split": args.split, **({"name": args.name} if args.name else {})}, # Στέλνουμε καθαρό "train"
            streaming=True,
            limit=args.limit if args.limit > 0 else None,
            skip=args.skip,
            min_int_score=args.min_int_score,
            int_score_key=args.int_score_key,
	    # Εδώ μπαίνει το skip μας
        )
    else:
        datatrove_reader = JsonlReader(
            data_folder=args.dataset, 
            text_key=args.column,
            glob_pattern=args.glob_pattern,
            limit=args.limit if args.limit > 0 else None
        )

    preprocess_executor = LocalPipelineExecutor(
        pipeline=[
            datatrove_reader,
            DocumentTokenizer(
                output_folder=args.output_folder,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                eos_token=args.eos_token,
                shuffle_documents=False,
                max_tokens_per_file=1e9,
            ),
        ],
        tasks=args.n_tasks,
        logging_dir=args.logging_dir,
    )
    preprocess_executor.run()

if __name__ == "__main__":
    _args = get_args()
    main(_args)
