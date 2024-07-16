import re

from word2number import w2n


def normalize(text: str) -> str:
    """
    Normalize (en) text for better comparison for number computation.

    This does very basic normalization right now which might not be sufficient
    beyond the working dataset.
    """

    words = re.split(r"[ \-,\.]", text)

    _words = []

    for w in words:
        if w.strip() == "":
            continue

        try:
            _words.append(str(w2n.word_to_num(w)))
        except:
            _words.append(w)

    _text = " ".join(_words)

    _text = re.sub(r"double (\d)", lambda m: m[1] + " " + m[1], _text, flags=re.I)
    _text = re.sub(r"triple (\d)", lambda m: m[1] + " " + m[1] + " " + m[1], _text, flags=re.I)

    _text = re.sub(r"(\d)", r"\1 ", _text).strip()
    _text = re.sub(r"\s+", " ", _text).strip()

    return _text
