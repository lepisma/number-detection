"""
Transcribe audio files using Skit's Speech LLM 2B model

Usage:
  transcribe-speech-llm.py <audio-dir> --output-csv=<output-csv>
"""

from transformers import AutoModel
import torchaudio

from docopt import docopt
from tqdm import tqdm
from glob import glob
import os
import re
import pandas as pd
import json


if __name__ == "__main__":
    args = docopt(__doc__)

    files = glob(os.path.join(args["<audio-dir>"], "*.flac"))

    model = AutoModel.from_pretrained("skit-ai/speechllm-2B", trust_remote_code=True)
    transform = torchaudio.transforms.Resample(8_000, 16_000)

    df = []

    for f in tqdm(files):
        waveform, sr = torchaudio.load(f)
        assert sr == 8000
        waveform = transform(waveform)

        output = model.generate_meta(audio_tensor=waveform, instruction="Give me the following information about the audio [Transcript]")
        m = re.search(r"\"Transcript\": \"(.*)\"", output)
        if m:
            transcript = m.groups()[0]
            utterances = [[{"confidence": 1.0, "transcript": transcript}]]
        else:
            utterances = []

        df.append({"id": f, "utterances": json.dumps(utterances)})

    pd.DataFrame(df).to_csv(args["--output-csv"], index=False)
