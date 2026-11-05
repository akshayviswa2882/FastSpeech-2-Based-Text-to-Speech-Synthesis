"""Telugu text frontend: normalisation + rule-based grapheme-to-phoneme.

The reference implementation has no Telugu support (English ARPAbet and
Mandarin pinyin only), and MFA ships no pretrained Telugu acoustic model or
dictionary. Telugu script is an abugida whose spelling is close to phonemic,
so a deterministic G2P works well:

  * every consonant carries an inherent /a/ unless followed by a vowel sign
    (matra) or the virama (్), which suppresses it -> conjuncts / gemination
  * Telugu does NOT delete the word-final inherent vowel (unlike Hindi schwa
    deletion), so no deletion rule is needed
  * anusvara (ం) is a nasal placeholder resolved to the homorganic nasal of the
    following consonant (ng/ny/nn/n/m), and to /m/ word-finally or before
    sonorants / sibilants (సమావేశం -> s a m aa v ee sh a m)
  * visarga (ః) -> /h/
  * ఋ / ృ -> /r u/ (Telugu pronunciation), nukta forms ఫ఼ / జ఼ -> /f/ /z/

The normaliser expands digits into Telugu words, including clock times
("10:45" -> "పది నలభై ఐదు"), years ("2026" -> "రెండు వేల ఇరవై ఆరు")
and the ordinal suffix వ ("12వ" -> "పన్నెండవ").
Output phones match fs2.text.TELUGU_PHONES.
"""

import re
import unicodedata

# ----------------------------------------------------------------------------- character tables

CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "rx", "ల": "l", "ళ": "ll", "ఴ": "ll", "వ": "v",
    "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
    "ౘ": "c", "ౙ": "j", "ౚ": "rx",
}
NUKTA_MAP = {"ph": "f", "j": "z"}
INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఌ": ["l", "u"], "ౡ": ["l", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ౢ": ["l", "u"], "ౣ": ["l", "uu"],
    "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}
VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"
CHANDRABINDU = {"ఁ", "ఀ", "ౄ"} - {"ౄ"}
NUKTA = "఼"
IGNORE = {"‌", "‍", "ౕ", "ౖ"}

_NASAL_FOR = {
    "k": "ng", "kh": "ng", "g": "ng", "gh": "ng",
    "c": "ny", "ch": "ny", "j": "ny", "jh": "ny",
    "tt": "nn", "tth": "nn", "dd": "nn", "ddh": "nn", "nn": "nn",
    "t": "n", "th": "n", "d": "n", "dh": "n", "n": "n",
    "p": "m", "ph": "m", "b": "m", "bh": "m", "m": "m", "f": "m",
}
_NASAL = "<N>"


def word_to_phones(word):
    """Convert one Telugu word (no spaces / punctuation) to phones."""
    word = unicodedata.normalize("NFC", word)
    phones = []
    pending = False  # last emitted consonant still needs a vowel
    for ch in word:
        if ch in IGNORE:
            continue
        if ch in CONSONANTS:
            if pending:
                phones.append("a")
            phones.append(CONSONANTS[ch])
            pending = True
        elif ch == NUKTA:
            if phones and phones[-1] in NUKTA_MAP:
                phones[-1] = NUKTA_MAP[phones[-1]]
        elif ch in VOWEL_SIGNS:
            phones.extend(VOWEL_SIGNS[ch])
            pending = False
        elif ch == VIRAMA:
            pending = False
        elif ch in INDEPENDENT_VOWELS:
            if pending:
                phones.append("a")
            phones.extend(INDEPENDENT_VOWELS[ch])
            pending = False
        elif ch == ANUSVARA:
            if pending:
                phones.append("a")
            phones.append(_NASAL)
            pending = False
        elif ch == VISARGA:
            if pending:
                phones.append("a")
            phones.append("h")
            pending = False
        elif ch in CHANDRABINDU:
            if pending:
                phones.append("a")
                pending = False
        # anything else (stray marks) is ignored
    if pending:
        phones.append("a")

    # resolve anusvara to the homorganic nasal of the next consonant
    for i, p in enumerate(phones):
        if p == _NASAL:
            nxt = phones[i + 1] if i + 1 < len(phones) else None
            phones[i] = _NASAL_FOR.get(nxt, "m")
    return phones


# ----------------------------------------------------------------------------- numbers

_UNITS = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
          "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
          "పద్దెనిమిది", "పందొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}
# multiplier forms used before వందలు / వేలు / లక్షలు ("రెండు వందలు", "ఐదు వేలు")
_HUNDREDS_EXACT = {1: "వంద", 2: "రెండు వందలు", 3: "మూడు వందలు", 4: "నాలుగు వందలు", 5: "ఐదు వందలు",
                   6: "ఆరు వందలు", 7: "ఏడు వందలు", 8: "ఎనిమిది వందలు", 9: "తొమ్మిది వందలు"}
_HUNDREDS_PREFIX = {1: "నూట", 2: "రెండు వందల", 3: "మూడు వందల", 4: "నాలుగు వందల", 5: "ఐదు వందల",
                    6: "ఆరు వందల", 7: "ఏడు వందల", 8: "ఎనిమిది వందల", 9: "తొమ్మిది వందల"}


def _below_100(n):
    if n < 20:
        return _UNITS[n]
    t, u = divmod(n, 10)
    return _TENS[t] + ("" if u == 0 else " " + _UNITS[u])


def _below_1000(n):
    if n < 100:
        return _below_100(n)
    h, r = divmod(n, 100)
    return _HUNDREDS_EXACT[h] if r == 0 else _HUNDREDS_PREFIX[h] + " " + _below_100(r)


def number_to_telugu(n):
    n = int(n)
    if n < 1000:
        return _below_1000(n)
    if n < 100000:
        th, r = divmod(n, 1000)
        if th == 1:
            head_exact, head_prefix = "వెయ్యి", "వెయ్యి"
        else:
            w = _below_100(th)
            head_exact, head_prefix = w + " వేలు", w + " వేల"
        return head_exact if r == 0 else head_prefix + " " + _below_1000(r)
    if n < 10000000:
        la, r = divmod(n, 100000)
        head = "లక్ష" if la == 1 else _below_100(la) + " లక్షల"
        return head if r == 0 else head + " " + number_to_telugu(r)
    cr, r = divmod(n, 10000000)
    head = "కోటి" if cr == 1 else number_to_telugu(cr) + " కోట్ల"
    return head if r == 0 else head + " " + number_to_telugu(r)


def _ordinal(n):
    """12 + వ -> పన్నెండవ (drop the final vowel sign of the cardinal, add అవ)."""
    w = number_to_telugu(n)
    if w and w[-1] in VOWEL_SIGNS:
        w = w[:-1]
    return w + "వ"


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("‌", "").replace("‍", "")
    # clock time 10:45 -> "పది నలభై ఐదు"; 10:00 -> "పది"; 10:05 -> "పది సున్నా ఐదు"
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f" {number_to_telugu(h)} "
        mins = number_to_telugu(mi) if mi >= 10 else "సున్నా " + number_to_telugu(mi)
        return f" {number_to_telugu(h)} {mins} "
    text = re.sub(r"(\d{1,2}):(\d{2})", _time, text)
    # ordinal suffix: 12వ, 1వ
    text = re.sub(r"(\d+)\s*వ(?![ా-్ౕౖ])", lambda m: " " + _ordinal(m.group(1)) + " ", text)
    # decimals
    text = re.sub(r"(\d+)\.(\d+)", lambda m: f" {number_to_telugu(m.group(1))} పాయింట్ "
                  + " ".join(number_to_telugu(d) for d in m.group(2)) + " ", text)
    text = re.sub(r"(\d+)\s?%", lambda m: f" {number_to_telugu(m.group(1))} శాతం ", text)
    text = re.sub(r"\d[\d,]*", lambda m: f" {number_to_telugu(m.group(0).replace(',', ''))} ", text)
    text = re.sub(r"\s+([,.?!;:।])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------------- sentence level

_TELUGU_CHAR = re.compile(r"[ఀ-౿]")
_PAUSE = re.compile(r"[,;:?!.।|]")


def is_supported(raw_text):
    """Reject transcripts containing Latin-script words: the Telugu phone set cannot speak them and
    MFA would have to absorb the audio into neighbouring words."""
    return re.search(r"[A-Za-z]", raw_text) is None


def words_for_alignment(normalized):
    """Words that go into the MFA .lab file (Telugu script only)."""
    return [w for w in re.split(r"[^ఀ-౿‌‍]+", normalized) if _TELUGU_CHAR.search(w)]


def text_to_phones(text, normalized=False):
    """Returns (phones, word_spans, normalized_text); "sp" inserted at punctuation."""
    norm = text if normalized else normalize_text(text)
    phones, spans = [], []
    for tok in re.findall(r"[ఀ-౿‌‍]+|[,;:?!.।|]|[A-Za-z]+", norm):
        if _PAUSE.fullmatch(tok):
            if phones and phones[-1] != "sp":
                phones.append("sp")
            continue
        if not _TELUGU_CHAR.search(tok):
            continue  # Latin tokens are not modelled by the Telugu voice
        ph = word_to_phones(tok)
        if not ph:
            continue
        spans.append(("".join(ph), len(phones), len(phones) + len(ph)))
        phones.extend(ph)
    while phones and phones[-1] == "sp":
        phones.pop()
    return phones, spans, norm


def lexicon_entries(word):
    return [word_to_phones(word)]


if __name__ == "__main__":
    import sys
    sents = sys.argv[1:] or [
        "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "పులి అడ విలో నిశ్శబ్దంగా నడిచింది.",
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
    ]
    for s in sents:
        ph, sp, n = text_to_phones(s)
        print(n)
        print(" | ".join(w for w, *_ in sp))
        print()
