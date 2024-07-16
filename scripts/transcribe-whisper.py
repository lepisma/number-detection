"""
Transcribe audio files using offline Whisper

Usage:
  transcribe-whisper.py <audio-dir> --output-csv=<output-csv>
"""

import json
import os
from glob import glob

import pandas as pd
import torch
from docopt import docopt
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

if __name__ == "__main__":
    args = docopt(__doc__)

    files = glob(os.path.join(args["<audio-dir>"], "*.flac"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )

    df = []

    for f in tqdm(files):
        result = pipe(f, generate_kwargs={"language": "english"})

        try:
            utterances = [[{"confidence": 0.0, "transcript": result["text"]}]]
        except:
            utterances = []

        df.append({"id": f, "utterances": json.dumps(utterances)})

    pd.DataFrame(df).to_csv(args["--output-csv"], index=False)
