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

import numpy as np
import pandas as pd
import werpy
from docopt import docopt
from number_detection.data import normalize


def process_utterance(raw: str) -> str:
    data = json.loads(raw)

    if not data:
        return ""

    return data[0][0]["transcript"]


def process_truth(label) -> str:
    try:
        return " ".join(str(int(label)))
    except ValueError:
        return ""


if __name__ == "__main__":
    args = docopt(__doc__)

    truth = pd.read_csv(args["<truth-csv>"])
    pred = pd.read_csv(args["<transcriptions-csv>"])

    # Mapping from filename to string transcription in normalized form
    truth_mapping = {
        it.relative_path.split("/")[1]: process_truth(it.tags)
        for _, it in truth.iterrows()
    }

    pred = [
        (it.id.split("/")[-1], normalize(process_utterance(it.utterances)))
        for _, it in pred.iterrows()
    ]

    ref = [truth_mapping[fid] for fid, _ in pred]
    hyp = [txt for _, txt in pred]


    wers = []

    for r, h in zip(ref, hyp):
        # There is one case where the reference is empty, if that's the case, just move ahead.
        if r == h == "":
            wers.append(0.0)
            continue

        try:
            wers.append(werpy.wer(r, h))
        except:
            # For the single reference that's empty (and the hyp is not) we
            # swap arguments and calculate. 0 length ref should be handled by
            # the library.
            wers.append(werpy.wer(h, r))


    ser = np.mean([w != 0 for w in wers])

    if args["--debug"]:
        for (fid, _), ref_text, hyp_text, wer in zip(pred, ref, hyp, wers):
            if wer != 0:
                print(f"Error in {fid}:: ref: {ref_text}, hyp: {hyp_text}")

    print(f"SER: {ser}")
    print(f"Mean WER: {np.mean(wers)}")
    print(f"Median WER: {np.median(wers)}")
