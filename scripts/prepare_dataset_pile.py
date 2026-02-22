import argparse
import itertools
import re
from typing import Optional, Set

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s\-_]+", "", s)
    s = re.sub(r"[^\w]", "", s)
    return s


def _parse_csv_list(s: Optional[str]) -> Set[str]:
    if not s:
        return set()
    items = [x.strip() for x in s.split(",") if x.strip()]
    return {_norm(x) for x in items}


def _get_pile_set_name_from_doc(doc) -> Optional[str]:
    md = getattr(doc, "metadata", None)
    if isinstance(md, dict):
        if isinstance(md.get("pile_set_name"), str):
            return md["pile_set_name"]
        meta = md.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("pile_set_name"), str):
            return meta["pile_set_name"]
    return None


class SkippableHFReader(HuggingFaceDatasetReader):
    def __init__(
        self,
        dataset,
        skip=0,
        limit=None,
        include_pile_sets: Optional[Set[str]] = None,
        exclude_pile_sets: Optional[Set[str]] = None,
        **kwargs,
    ):
        self.skip_n = int(skip or 0)
        self.limit_n = int(limit) if (limit and int(limit) > 0) else None
        self.include_pile_sets = include_pile_sets or set()
        self.exclude_pile_sets = exclude_pile_sets or set()

        # μην αφήνεις τον parent να κάνει limit ΠΡΙΝ το filtering
        kwargs.pop("limit", None)
        super().__init__(dataset, **kwargs)

    def _keep(self, doc) -> bool:
        if not self.include_pile_sets and not self.exclude_pile_sets:
            return True

        pile_name = _get_pile_set_name_from_doc(doc)
        if pile_name is None:
            # strict: αν έχεις include list και δεν βρίσκουμε pile_set_name -> drop
            return False if self.include_pile_sets else True

        n = _norm(pile_name)
        if self.include_pile_sets and n not in self.include_pile_sets:
            return False
        if self.exclude_pile_sets and n in self.exclude_pile_sets:
            return False
        return True

    def run(self, data=None, rank=0, world_size=1):
        it = super().run(data, rank, world_size)

        # 1) quality filter
        if self.include_pile_sets or self.exclude_pile_sets:
            it = filter(self._keep, it)

        # 2) skip
        if self.skip_n > 0:
            print(f"--- INFO: Skipping first {self.skip_n} FILTERED docs (rank {rank}) ---")
            it = itertools.islice(it, self.skip_n, None)

        # 3) limit
        if self.limit_n is not None:
            print(f"--- INFO: Taking {self.limit_n} FILTERED docs (rank {rank}) ---")
            it = itertools.islice(it, self.limit_n)

        return it


def get_args():
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("Tokenizer")
    g.add_argument("--tokenizer-name-or-path", type=str, required=True)
    g.add_argument("--eos-token", type=str, default=None)

    g = parser.add_argument_group("Output data")
    g.add_argument("--output-folder", type=str, required=True)

    g = parser.add_argument_group("Misc")
    g.add_argument("--logging-dir", type=str, default=None)
    g.add_argument("--n-tasks", type=int, default=8)

    g = parser.add_argument_group("Partition controls")
    g.add_argument("--limit", type=int, default=-1)
    g.add_argument("--skip", type=int, default=0)

    sp = parser.add_subparsers(dest="readers", required=True)

    p1 = sp.add_parser("hf")
    p1.add_argument("--dataset", type=str, required=True)
    p1.add_argument("--column", type=str, default="text")
    p1.add_argument("--split", type=str, default="train")

    # The Pile: config "all" έχει train/validation/test :contentReference[oaicite:2]{index=2}
    p1.add_argument("--hf-config", type=str, default=None)

    # Quality filters for The Pile (comma-separated pile_set_name values) :contentReference[oaicite:3]{index=3}
    p1.add_argument("--include-pile-sets", type=str, default=None)
    p1.add_argument("--exclude-pile-sets", type=str, default=None)

    p2 = sp.add_parser("jsonl")
    p2.add_argument("--dataset", type=str, required=True)
    p2.add_argument("--column", type=str, default="text")
    p2.add_argument("--glob-pattern", type=str, default=None)

    return parser.parse_args()


def main(args):
    if args.readers == "hf":
        dataset_options = {"split": args.split}
        if args.hf_config:
            dataset_options["name"] = args.hf_config

        include_sets = _parse_csv_list(args.include_pile_sets)
        exclude_sets = _parse_csv_list(args.exclude_pile_sets)

        reader = SkippableHFReader(
            dataset=args.dataset,
            text_key=args.column,
            dataset_options=dataset_options,
            streaming=True,
            skip=args.skip,
            limit=args.limit if args.limit > 0 else None,
            include_pile_sets=include_sets,
            exclude_pile_sets=exclude_sets,
        )
    else:
        reader = JsonlReader(
            data_folder=args.dataset,
            text_key=args.column,
            glob_pattern=args.glob_pattern,
            limit=args.limit if args.limit > 0 else None,
        )

    ex = LocalPipelineExecutor(
        pipeline=[
            reader,
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
    ex.run()


if __name__ == "__main__":
    main(get_args())

