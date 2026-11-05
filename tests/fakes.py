"""Test doubles and synthetic decisions in the OpenCaseLaw scrape format.

The decision texts are invented for the tests. They copy the layout of the real
scrape (page markers on their own lines, hyperlinked references broken onto
their own lines, Erwägung numbers alone on a line), not its content.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

from devils_advocate.corpus import Decision

KUENDIGUNG = """\
Urteilskopf
150 III 100
12. Auszug aus dem Urteil der I. zivilrechtlichen Abteilung i.S. A. gegen B. AG (Beschwerde in Zivilsachen)
4A_1/2023 vom 1. März 2024
Regeste
Art. 337 OR
; fristlose Kündigung. Eine fristlose Kündigung setzt eine besonders schwere Verfehlung voraus (E. 2.2).
Sachverhalt
ab Seite 100
BGE 150 III 100 S. 100
A. war bei der B. AG als Buchhalterin angestellt.
Erwägungen
Aus den Erwägungen:
2.
Die Beschwerdeführerin rügt eine Verletzung von
Art. 337 OR
.
2.1
Nach
Art. 337 Abs. 1 OR
kann der Arbeitgeber das Arbeitsverhältnis aus wichtigen Gründen jederzeit fristlos auflösen. Als wichtiger Grund gilt namentlich jeder Umstand, bei dessen Vorhandensein dem Kündigenden nach Treu und Glauben die Fortsetzung des Arbeitsverhältnisses nicht mehr zugemutet werden darf (
BGE 142 III 579
E. 4.2 S. 579 f.; vgl.
BGE 137 III 303
E. 2.1.1).
BGE 150 III 100 S. 101
2.2
Die fristlose Kündigung ist nur bei besonders schweren Verfehlungen gerechtfertigt. Sind die Verfehlungen weniger schwerwiegend, so müssen sie trotz Verwarnung wiederholt vorgekommen sein. Zu prüfen ist dabei:
1.
ob die Verfehlung das Vertrauensverhältnis zerstört hat, und
2.
ob eine Verwarnung erfolgt ist.
3.
Die Vorinstanz hat diese Grundsätze richtig angewendet. Die Beschwerde ist abzuweisen.
"""

CONSIDERANTS = """\
Urteilskopf
149 II 200
20. Extrait de l'arrêt de la IIe Cour de droit public dans la cause A. contre Office cantonal (recours en matière de droit public)
2C_5/2022 du 3 avril 2023
Regeste
Art. 29 Abs. 2 BV
; rechtliches Gehör. Die Behörde muss ihren Entscheid begründen (E. 4).
Sachverhalt
BGE 149 II 200 S. 200
A. a demandé une autorisation de séjour.
Erwägungen
Extrait des considérants:
4.
Le droit d'être entendu garanti par l'
art. 29 al. 2 Cst.
implique pour l'autorité l'obligation de motiver sa décision. Il suffit qu'elle mentionne, au moins brièvement, les motifs qui l'ont guidée et sur lesquels elle a fondé sa décision (
ATF 150 III 100
consid. 2.1; ATF 142 II 154 consid. 4.2).
4.1
La fristlose Kündigung n'a rien à voir ici. L'autorité cantonale a motivé sa décision de manière suffisante.
"""

ALTE_PRAXIS = """\
Urteilskopf
120 II 50
9. Auszug aus dem Urteil der II. Zivilabteilung vom 3. Februar 1994 i.S. X. gegen Y.
Regeste
Grunddienstbarkeit; Auslegung.
Erwägungen
Aus den Erwägungen:
3.
a) Der Inhalt einer Dienstbarkeit bestimmt sich in erster Linie nach dem Grundbucheintrag.
b) Ist der Eintrag unklar, so ist auf den Erwerbsgrund zurückzugreifen.
aa) Massgebend ist der Wille der ursprünglichen Parteien.
bb) Fehlt ein solcher, so entscheidet die Art der Ausübung.
4.
Die Berufung ist demnach abzuweisen.
"""

DECISIONS = [
    Decision("BGE 150 III 100", "2024-03-01", "de", KUENDIGUNG, "https://example.org/150-III-100"),
    Decision("BGE 149 II 200", "2023-04-03", "fr", CONSIDERANTS, "https://example.org/149-II-200"),
    Decision("BGE 120 II 50", "1994-02-03", "de", ALTE_PRAXIS, "https://example.org/120-II-50"),
]


class FakeModel:
    """A chat model whose replies are produced by `respond(messages, name)`."""

    def __init__(self, respond: Callable[[list[dict], str], dict]) -> None:
        self.respond = respond
        self.calls: list[tuple[str, list[dict]]] = []

    def json(self, messages: list[dict], schema: dict, *, name: str) -> dict[str, Any]:
        self.calls.append((name, messages))
        return self.respond(messages, name)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def passages_in(messages: list[dict]) -> dict[str, str]:
    """The labelled passages a counter prompt shows the model: {"P1": text}."""
    user = messages[-1]["content"]
    blocks = re.split(r"\n\n(?=\[P\d+\] )", user.split("Passages:\n\n", 1)[1])
    out = {}
    for block in blocks:
        header, _, text = block.partition("\n")
        out[header.split("]")[0].lstrip("[")] = text
    return out


class HashEmbedder:
    """Deterministic bag-of-words embedder: shared words give similar vectors."""

    def __init__(self, dims: int = 64, fail_first: int = 0) -> None:
        self.dims = dims
        self.fail_first = fail_first
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        from devils_advocate.client import LLMError

        self.calls += 1
        if self.calls <= self.fail_first:
            raise LLMError("simulated failure")
        rows = []
        for text in texts:
            row = [0.0] * self.dims
            for word in re.findall(r"\w{4,}", text.lower()):
                digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
                row[int.from_bytes(digest, "little") % self.dims] += 1.0
            row[0] += 1e-3  # never all zero
            rows.append(row)
        return rows


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)
