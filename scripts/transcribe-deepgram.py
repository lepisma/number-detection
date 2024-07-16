"""
Transcribe audio files using Deepgram's API

Usage:
  transcribe-deepgram.py <audio-dir> --output-csv=<output-csv>
"""

import json
import os
import re
from glob import glob

import pandas as pd
from deepgram import DeepgramClient, PrerecordedOptions
from docopt import docopt
from tqdm import tqdm

if __name__ == "__main__":
    args = docopt(__doc__)

    files = glob(os.path.join(args["<audio-dir>"], "*.flac"))

    deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

    df = []

    for f in tqdm(files):
        with open(f, "rb") as buffer_data:
            payload = { "buffer": buffer_data }

            options = PrerecordedOptions(
                smart_format=True, model="nova-2", language="en-IN"
            )

            response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

            try:
                top_alternative = response.results.channels[0].alternatives[0]
                utterances = [[{"confidence": top_alternative.confidence, "transcript": top_alternative.transcript}]]
            except:
                utterances = []

        df.append({"id": f, "utterances": json.dumps(utterances)})

    pd.DataFrame(df).to_csv(args["--output-csv"], index=False)
