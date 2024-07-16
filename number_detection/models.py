import re
from abc import ABC, abstractmethod

import torch
import torchaudio
from deepgram import DeepgramClient, PrerecordedOptions
from transformers import (AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor,
                          pipeline)


class Transcriber(ABC):

    @abstractmethod
    def transcribe(self, audio_path: str):
        """
        Read given audio file and transcribe. Preprocessing, if needed has
        to be done by the implementation.

        Output is a list of `utterances`, each utterance is a list of
        alternative which is a dictionary with two keys, `confidence` (a float)
        and `transcript` (string). If the model has to indicate that the audio
        is empty, it should give an empty list of utterances.
        """

        ...


    def streaming_transcribe(self, audio_path: str):
        """
        Read given audio file and pass it to a streaming version of the
        transcriber to simulate streaming model performance.
        """

        raise NotImplementedError()


class DGTranscriber(Transcriber):
    """
    Transcriber using DeepGram Speech to Text API.
    """

    def __init__(self, api_key: str, model: str="nova-2", language: str="en-IN") -> None:
        super().__init__()

        self.options = PrerecordedOptions(smart_format=True, model=model, language=language)
        self.client = DeepgramClient(api_key)

    def transcribe(self, audio_path: str):

        with open(audio_path, "rb") as fp:
            payload = {"buffer": fp}

            response = self.client.listen.prerecorded.v("1").transcribe_file(payload, self.options)

        try:
            # Only taking top alternative since we are, for now, only comparing
            # performance on that.
            top_alt = response.results.channels[0].alternatives[0]
            return [[{"confidence": top_alt.confidence, "transcript": top_alt.transcript}]]
        except:
            return []


class WhisperTranscriber(Transcriber):

    def __init__(self) -> None:
        super().__init__()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
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

    def transcribe(self, audio_path: str):
        result = self.pipe(audio_path, generate_kwargs={"language": "english"})

        try:
            return [[{"confidence": 0.0, "transcript": result["text"]}]]
        except:
            return []


class SpeechLLMTranscriber(Transcriber):

    def __init__(self) -> None:
        super().__init__()

        self.model = AutoModel.from_pretrained("skit-ai/speechllm-2B", trust_remote_code=True)

    def transcribe(self, audio_path: str):
        waveform, sr = torchaudio.load(audio_path)
        transform = torchaudio.transforms.Resample(sr, 16_000)
        waveform = transform(waveform)

        output = self.model.generate_meta(audio_tensor=waveform, instruction="Give me the following information about the audio [Transcript]")
        m = re.search(r"\"Transcript\": \"(.*)\"", output)
        if m:
            transcript = m.groups()[0]
            return [[{"confidence": 1.0, "transcript": transcript}]]
        else:
            return []
