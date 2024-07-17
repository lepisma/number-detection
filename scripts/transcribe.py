"""
Transcribe audio files

Usage:
  transcribe.py deepgram <audio-dir> --output-csv=<output-csv>
  transcribe.py google <audio-dir> --output-csv=<output-csv>
  transcribe.py speechllm <audio-dir> --output-csv=<output-csv>
  transcribe.py whisper <audio-dir> --output-csv=<output-csv>
"""

import json
import os
from glob import glob

import pandas as pd
from docopt import docopt
from number_detection.models import (DGTranscriber, GoogleTranscriber,
                                     SpeechLLMTranscriber, WhisperTranscriber)
from tqdm import tqdm

if __name__ == "__main__":
    args = docopt(__doc__)

    files = glob(os.path.join(args["<audio-dir>"], "*.flac"))

    if args["deepgram"]:
        t = DGTranscriber(os.getenv("DEEPGRAM_API_KEY"))
    elif args["speechllm"]:
        t = SpeechLLMTranscriber()
    elif args["whisper"]:
        t = WhisperTranscriber()
    elif args["google"]:
        t = GoogleTranscriber()
    else:
        raise ValueError("No transcriber selected")

    df = []

    for f in tqdm(files):
        df.append({"id": f, "utterances": json.dumps(t.transcribe(f))})

    pd.DataFrame(df).to_csv(args["--output-csv"], index=False)
