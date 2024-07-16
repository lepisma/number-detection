"""
Evaluate performance of transcriptions.

Usage:
  evaluate.py <truth-csv> <transcriptions-csv> [--debug]

Arguments:
  <truth-csv>                Truth file from the skit dataset
  <transcriptions-csv>       Transcription from the transcription scripts

Options:
  --debug                    Print non-matching cases
"""

import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import werpy
from docopt import docopt
from number_detection.data import normalize
from tqdm import tqdm


def process_utterance(raw: str) -> str:
    data = json.loads(raw)

    if not data:
        return ""

    return data[0][0]["transcript"]


if __name__ == "__main__":
    args = docopt(__doc__)

    truth = pd.read_csv(args["<truth-csv>"])
    pred = pd.read_csv(args["<transcriptions-csv>"])

    # Mapping from filename to string transcription in normalized form
    truth_mapping = {
        it.relative_path.split("/")[1]: " ".join(str(int(it.tags)))
        for _, it in truth.iterrows()
    }

    pred = [
        (it.id.split("/")[-1], normalize(process_utterance(it.utterances)))
        for _, it in pred.iterrows()
    ]

    ref = [truth_mapping[fid] for fid, _ in pred]
    hyp = [txt for _, txt in pred]

    wers = werpy.wers(ref, hyp)

    ser = np.mean([w != 0 for w in wers])

    if args["--debug"]:
        for ref_text, hyp_text, wer in zip(ref, hyp, wers):
            if wer != 0:
                print(f"Error:: ref: {ref_text}, hyp: {hyp_text}")

    print(f"SER: {ser}")
    print(f"Mean WER: {np.mean(wers)}")
    print(f"Median WER: {np.median(wers)}")
