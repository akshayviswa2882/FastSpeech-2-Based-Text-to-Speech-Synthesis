"""Small helpers shared by the per-language entry-point scripts."""

import os


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_test_sentences(language):
    path = os.path.join(project_root(), "test_sentences", f"{language}.tsv")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                sid, text = line.split("\t", 1)
                out.append((sid, text))
    return out
