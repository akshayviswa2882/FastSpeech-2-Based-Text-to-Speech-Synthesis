"""English text frontend: normalisation + grapheme-to-phoneme.

The same normalisation is applied to Svarah transcripts (before alignment)
and to test sentences (before inference), so the model sees identical
token conventions in both cases.

Normalisation covers what the test sentences probe:
  * clock times     "10:45 AM"          -> "ten forty five ay em"
  * dates           "12 November 2026"  -> "twelfth november twenty twenty six"
  * years / ordinals / cardinals / decimals / currency / percent
G2P uses g2p_en: CMUdict lookup, a POS-tag based heteronym table
("conduct" noun = K AA1 N D AH0 K T, verb = K AA0 N D AH1 K T) and a neural
seq2seq model for out-of-vocabulary words (e.g. "antidisestablishmentarianism",
"andaroo").
"""

import re
import unicodedata

import inflect

_inflect = inflect.engine()
_g2p = None

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
_MONTH_RE = "|".join(m.capitalize() + "|" + m for m in MONTHS) + "|" + "|".join(
    m[:3].capitalize() + r"\.?" for m in MONTHS)
_MONTH_ABBR = {m[:3]: m for m in MONTHS}

PAUSE_PUNCT = {",", ";", ":", "?", "!", ".", "-", "--", "..."}


def get_g2p():
    global _g2p
    if _g2p is None:
        import nltk
        for pkg, path in [("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                          ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
                          ("cmudict", "corpora/cmudict")]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


# ----------------------------------------------------------------------------- numbers

def _words(n):
    return _inflect.number_to_words(n, andword="").replace("-", " ").replace(",", "")


def _ordinal(n):
    return _inflect.number_to_words(_inflect.ordinal(int(n)), andword="").replace("-", " ").replace(",", "")


def _year(y):
    y = int(y)
    if 2000 <= y <= 2009:
        return _words(y)
    if 1100 <= y <= 2099 and y % 100 != 0:
        return f"{_words(y // 100)} {_words(y % 100) if y % 100 >= 10 else 'oh ' + _words(y % 100)}"
    if 1100 <= y <= 2099:
        return f"{_words(y // 100)} hundred"
    return _words(y)


def _month(m):
    m = m.lower().rstrip(".")
    return _MONTH_ABBR.get(m[:3], m)


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    ampm = (m.group(3) or "").lower().replace(".", "")
    out = _words(h)
    if mi == 0:
        out += " o'clock" if not ampm else ""
    elif mi < 10:
        out += " oh " + _words(mi)
    else:
        out += " " + _words(mi)
    if ampm == "am":
        out += " ay em"
    elif ampm == "pm":
        out += " pee em"
    return f" {out} "


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)            # [noise], <unk> style annotations
    # times: 10:45, 10:45 AM, 10:45 a.m.
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?\s?[Mm]\.?)?(?![\w])", _time, text)
    text = re.sub(r"\b([AaPp])\.[Mm]\.", lambda m: " ay em " if m.group(1).lower() == "a" else " pee em ", text)
    # dates: 12 November 2026 / 12th Nov, 2026 / November 12, 2026
    text = re.sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\s*,?\s+(\d{{4}})\b",
                  lambda m: f" {_ordinal(m.group(1))} {_month(m.group(2))} {_year(m.group(3))} ", text)
    text = re.sub(rf"\b({_MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(\d{{4}})\b",
                  lambda m: f" {_month(m.group(1))} {_ordinal(m.group(2))} {_year(m.group(3))} ", text)
    text = re.sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\b",
                  lambda m: f" {_ordinal(m.group(1))} {_month(m.group(2))} ", text)
    # currency / percent
    text = re.sub(r"(?:₹|Rs\.?|INR)\s?([\d,]+(?:\.\d+)?)", lambda m: f" {m.group(1)} rupees ", text)
    text = re.sub(r"\$\s?([\d,]+(?:\.\d+)?)", lambda m: f" {m.group(1)} dollars ", text)
    text = re.sub(r"([\d.]+)\s?%", lambda m: f" {m.group(1)} percent ", text)
    # ordinals 1st 2nd 3rd 4th
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: f" {_ordinal(m.group(1))} ", text)
    # standalone 4-digit years
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: f" {_year(m.group(1))} ", text)
    # remaining numbers (with thousands separators / decimals)
    text = re.sub(r"\d[\d,]*\.\d+", lambda m: " " + " point ".join(
        [_words(int(m.group(0).split(".")[0].replace(",", ""))),
         " ".join(_words(int(d)) for d in m.group(0).split(".")[1])]) + " ", text)
    text = re.sub(r"\d[\d,]*", lambda m: f" {_words(int(m.group(0).replace(',', '')))} ", text)
    text = text.replace("&", " and ").replace("@", " at ")
    text = re.sub(r"[^A-Za-z' ,.?!;:\-]", " ", text)
    text = re.sub(r"\s*-\s*", " ", text)
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_supported(raw_text):
    """Reject transcripts with letters outside Latin script (e.g. Devanagari words): normalisation
    would delete them while they are still spoken, which corrupts the forced alignment."""
    for ch in unicodedata.normalize("NFKC", raw_text):
        if ch.isalpha() and not ("a" <= ch.lower() <= "z"):
            return False
    return True


def words_for_alignment(normalized):
    """Lower-cased word list (punctuation removed) - what goes into the MFA .lab file."""
    clean = re.sub(r"[^a-z' ]", " ", normalized.lower())
    return [w.strip("'") for w in clean.split() if w.strip("'")]


# ----------------------------------------------------------------------------- G2P

def text_to_phones(text, normalized=False):
    """Returns (phones, word_spans, normalized_text).

    phones: list of ARPAbet symbols with "sp" at punctuation pauses.
    word_spans: list of (word, start_idx, end_idx) into phones (for plotting).
    """
    g2p = get_g2p()
    norm = text if normalized else normalize_text(text)
    out = g2p(norm)
    groups, cur = [], []
    for tok in out:
        if tok == " ":
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        groups.append(cur)

    from nltk.tokenize import TweetTokenizer
    word_tokenize = TweetTokenizer().tokenize
    cleaned = re.sub(r"[^ a-z'.,?!\-]", "", norm.lower())
    tokens = word_tokenize(cleaned)
    labels = tokens if len(tokens) == len(groups) else [None] * len(groups)

    phones, spans = [], []
    for label, grp in zip(labels, groups):
        if all(not re.search(r"[A-Z]", g) for g in grp):  # punctuation group
            if any(g in PAUSE_PUNCT for g in grp) and phones and phones[-1] != "sp":
                phones.append("sp")
            continue
        start = len(phones)
        phones.extend(grp)
        spans.append((label or "".join(grp), start, len(phones)))
    while phones and phones[-1] == "sp":
        phones.pop()
    return phones, spans, norm


def lexicon_entries(word):
    """All pronunciations of a word for the MFA dictionary.

    Heteronyms get every variant so MFA can pick the one that matches the audio.
    """
    g2p = get_g2p()
    prons = []
    if word in g2p.homograph2features:
        p1, p2, _ = g2p.homograph2features[word]
        prons += [p1, p2]
    if word in g2p.cmu:
        prons += g2p.cmu[word]
    if not prons:
        prons.append([p for p in g2p(word) if p.strip() and re.search(r"[A-Z]", p)])
    uniq = []
    for p in prons:
        if p and p not in uniq:
            uniq.append(p)
    return uniq


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:] or [
        "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
        "The orchestra's conduct during the ceremony was praised by the audience.",
        "The researchers will conduct an experiment to evaluate the proposed method.",
        "The rhyth m of the orchestra changed unexpectedly.",
    ]:
        ph, sp, n = text_to_phones(s)
        print(n)
        print(" ".join(ph))
        print([w for w, *_ in sp])
