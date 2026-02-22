import dataclasses
from nanotron.config import SFTDatasetsArgs

def show(cls):
    print(f"\n== {cls.__name__} fields ==")
    for f in dataclasses.fields(cls):
        print("-", f.name, "|", getattr(f.type, "__name__", str(f.type)))


show(SFTDatasetsArgs)
