"""Language-specific symbol tables.

The reference implementation (ming024/FastSpeech2) uses one global symbol table
that mixes ASCII letters, ARPAbet and pinyin. Here each language gets its own
compact phone inventory so the embedding table only contains phones that can
actually occur, and a Telugu inventory is added.

Index 0 is always the padding symbol.
"""

PAD = "_"
SILENCES = ["sp", "spn", "sil"]

# ARPAbet with lexical stress (same set as CMUdict / MFA english_us_arpa / g2p_en)
_ARPA_BASE_VOWELS = [
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
]
_ARPA_CONSONANTS = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R",
    "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
]
ENGLISH_PHONES = (
    [v + s for v in _ARPA_BASE_VOWELS for s in ("0", "1", "2")]
    + _ARPA_BASE_VOWELS  # stress-less variants (MFA occasionally emits these)
    + _ARPA_CONSONANTS
)

# Telugu phone inventory produced by Telugu/frontend.py (ASCII, ISO-15919-like)
TELUGU_VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
TELUGU_CONSONANTS = [
    "k", "kh", "g", "gh", "ng",
    "c", "ch", "j", "jh", "ny",
    "tt", "tth", "dd", "ddh", "nn",
    "t", "th", "d", "dh", "n",
    "p", "ph", "b", "bh", "m",
    "y", "r", "rx", "l", "ll", "v",
    "sh", "ss", "s", "h",
    "f", "z",  # appear in loanwords written with nukta / in English words
]
TELUGU_PHONES = TELUGU_VOWELS + TELUGU_CONSONANTS


def get_symbols(language):
    language = language.lower()
    if language == "english":
        phones = ENGLISH_PHONES
    elif language == "telugu":
        phones = TELUGU_PHONES
    else:
        raise ValueError(f"Unknown language: {language}")
    return [PAD] + phones + SILENCES


_cache = {}


def _maps(language):
    if language not in _cache:
        syms = get_symbols(language)
        _cache[language] = (
            {s: i for i, s in enumerate(syms)},
            {i: s for i, s in enumerate(syms)},
        )
    return _cache[language]


def phones_to_sequence(phones, language):
    """phones: list[str] or space-separated str -> list[int]. Unknown phones raise."""
    if isinstance(phones, str):
        phones = phones.strip().strip("{}").split()
    s2i, _ = _maps(language)
    seq = []
    for p in phones:
        if p not in s2i:
            raise KeyError(f"Phone '{p}' not in {language} symbol table")
        seq.append(s2i[p])
    return seq


def sequence_to_phones(seq, language):
    _, i2s = _maps(language)
    return [i2s[int(i)] for i in seq]
