import argparse
import itertools
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer

# --- CUSTOM CLASS ΓΙΑ ΝΑ ΛΥΣΟΥΜΕ ΤΟ ΠΡΟΒΛΗΜΑ ΤΟΥ SKIP ---
class SkippableHFReader(HuggingFaceDatasetReader):
    def __init__(self, dataset, skip=0, **kwargs):
        # Αφαιρούμε το skip από τα kwargs πριν το στείλουμε στον πατέρα
        self.skip_n = skip
        super().__init__(dataset, **kwargs)
        
    def run(self, data=None, rank=0, world_size=1):
        # Ζητάμε από τον πατέρα (HuggingFaceDatasetReader) τον iterator
        # Προσοχή: Εδώ το split θα είναι σκέτο "train", οπότε δεν θα σκάσει
        iterator = super().run(data, rank, world_size)
        
        # Αν έχουμε skip > 0, καταναλώνουμε τα πρώτα στοιχεία
        if self.skip_n > 0:
            print(f"--- INFO: Skipping first {self.skip_n} documents... ---")
            # Το islice καταναλώνει αποδοτικά τον iterator
            iterator = itertools.islice(iterator, self.skip_n, None)
            
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

    # Παράμετροι Split
    group.add_argument("--limit", type=int, default=-1)
    group.add_argument("--skip", type=int, default=0)

    sp = parser.add_subparsers(dest="readers", required=True)

    p1 = sp.add_parser(name="hf")
    p1.add_argument("--dataset", type=str, required=True)
    p1.add_argument("--column", type=str, default="text")
    p1.add_argument("--split", type=str, default="train")

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
            dataset_options={"split": args.split}, # Στέλνουμε καθαρό "train"
            streaming=True,
            limit=args.limit if args.limit > 0 else None,
            skip=args.skip # Εδώ μπαίνει το skip μας
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
