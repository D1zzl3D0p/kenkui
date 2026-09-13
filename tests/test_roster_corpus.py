"""Six books: the roster keeps people apart, and the identity pass merges nicknames."""
# ruff: noqa: D103, PLR2004

from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import pytest

import kenkui as kk
from kenkui._characters import spacy_roster
from kenkui._characters.identity_pass import IdentityPass
from kenkui._domain.grid import build_grid, dialogue_ranges

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._domain.casting import CharacterProfile

pytestmark = pytest.mark.skipif(
    not os.environ.get("KENKUI_RUN_CORPUS"), reason="set KENKUI_RUN_CORPUS=1 to run"
)

# A local Calibre library; the corpus tier skips when it is absent.
LIBRARY = Path(
    os.environ.get("KENKUI_CORPUS_LIBRARY", "~/Calibre Library")
).expanduser()
FIXTURES = Path(__file__).parent / "data" / "roster_identity"
BOOKS: dict[str, str] = {
    "dune": "Frank Herbert/Dune (466)/Dune - Frank Herbert.epub",
    "eotw": (
        "Robert Jordan/The Eye of the World (322)/"
        "The Eye of the World - Robert Jordan.epub"
    ),
    "persuasion": "Jane Austen/Persuasion (244)/Persuasion - Jane Austen.epub",
    "citycity": (
        "China Mieville/The City & the City (463)/"
        "The City & the City - China Mieville.epub"
    ),
    "uprooted": (
        "Naomi Novik/Uprooted_ A Lush Fairytale-Inspired Fantasy by the Author of the "
        "Bestselling a Deadly Education (125)/"
        "Uprooted_ A Lush Fairytale-Inspired Fantas - "
        "Naomi Novik.epub"
    ),
    "foundation": "Isaac Asimov/Foundation (453)/Foundation - Isaac Asimov.epub",
}


def T(*words: str) -> set[str]:  # noqa: N802 - compact copied fixture helper
    return set(words)


PRINCIPALS: dict[str, dict[str, set[str]]] = {  # copied from roster_lab.py
    "dune": {
        "Paul": T("paul"),
        "Jessica": T("jessica"),
        "Leto": T("leto"),
        "Baron": T("vladimir"),
        "Stilgar": T("stilgar"),
        "Chani": T("chani"),
        "Thufir": T("thufir", "hawat"),
        "Gurney": T("gurney", "halleck"),
        "Duncan": T("duncan", "idaho"),
        "Yueh": T("yueh", "wellington"),
        "Piter": T("piter"),
        "Feyd": T("feyd", "feyd-rautha"),
        "Kynes": T("kynes", "liet"),
        "Alia": T("alia"),
        "Irulan": T("irulan"),
        "Nefud": T("nefud"),
        "Rabban": T("rabban"),
        "Mapes": T("mapes"),
        "Harah": T("harah"),
        "Jamis": T("jamis"),
        "Fenring": T("fenring"),
        "Emperor": T("emperor", "shaddam"),
    },
    "eotw": {
        n: T(n.lower())
        for n in (
            "Rand",
            "Mat",
            "Thom",
            "Moiraine",
            "Elayne",
            "Gawyn",
            "Perrin",
            "Lan",
            "Agelmar",
            "Mordeth",
            "Egwene",
            "Nynaeve",
            "Elaida",
            "Morgase",
            "Bartim",
            "Paitr",
            "Loial",
            "Min",
            "Tam",
        )
    }
    | {"Ba'alzamon": T("ba'alzamon")},
    "persuasion": {
        "Anne": T("anne"),
        "Wentworth": T("wentworth", "frederick"),
        "Russell": T("russell"),
        "Harville": T("harville"),
        "Benwick": T("benwick"),
        "Clay": T("clay"),
        "Smith": T("smith"),
        "Louisa": T("louisa"),
        "Henrietta": T("henrietta"),
        "Mary": T("mary"),
        "Elizabeth": T("elizabeth"),
        "Walter": T("walter"),
        "Hayter": T("hayter"),
    },
    "citycity": {
        "Borlu": T("borlú", "borlu", "tyador"),
        "Corwi": T("corwi", "lizbyet"),
        "Dhatt": T("dhatt", "qussim"),
        "Bowden": T("bowden"),
        "Nancy": T("nancy"),
        "Yolanda": T("yolanda"),
        "Gadlem": T("gadlem"),
        "Syedr": T("syedr"),
        "Buric": T("buric"),
        "Aikam": T("aikam"),
    },
    "uprooted": {
        "Agnieszka": T("agnieszka", "nieshka"),
        "Sarkan": T("sarkan", "dragon"),
        "Kasia": T("kasia"),
        "Marek": T("marek"),
        "Solya": T("solya", "falcon"),
        "Alosha": T("alosha"),
        "Ballo": T("ballo"),
        "Danka": T("danka"),
    },
    "foundation": {
        "Seldon": T("seldon", "hari"),
        "Gaal": T("gaal", "dornick"),
        "Hardin": T("hardin", "salvor"),
        "Pirenne": T("pirenne"),
        "Wienis": T("wienis"),
        "Lepold": T("lepold"),
        "Verisof": T("verisof"),
        "Mallow": T("mallow", "hober"),
        "Sutt": T("sutt", "jorane"),
        "Aporat": T("aporat"),
        "Chen": T("chen", "linge"),
    },
}
APART: dict[str, list[tuple[str, str]]] = {
    "dune": [
        ("Paul Atreides", "Leto Atreides"),
        ("Vladimir Harkonnen", "Beast Rabban"),
        ("Count Fenring", "Lady Fenring"),
    ],
    "eotw": [
        ("Rand al'Thor", "Tam al'Thor"),
        ("Rand", "Tam"),
        ("Master Luhhan", "Mistress Luhhan"),
        ("Master al'Vere", "Mistress al'Vere"),
        ("Master Cauthon", "Mistress Cauthon"),
        ("Master Aybara", "Mistress Aybara"),
        ("Master Grinwell", "Mistress Grinwell"),
    ],
    "persuasion": [
        ("Charles Hayter", "Charles Musgrove"),
        ("Anne Elliot", "Walter Elliot"),
        ("Mr Elliot", "Anne Elliot"),
        ("Admiral Croft", "Mrs Croft"),
        ("Mrs Musgrove", "Louisa Musgrove"),
        ("Mr Musgrove", "Charles Musgrove"),
        ("Mr Elliot", "Walter Elliot"),
    ],
    "citycity": [("Mr Geary", "Mrs Geary")],
    "uprooted": [],
    "foundation": [],
}
SAME: dict[str, list[tuple[str, str]]] = {
    "dune": [
        ("Paul", "Usul"),
        ("Paul", "Muad'Dib"),
        ("Liet", "Kynes"),
        ("Paul", "Paul Atreides"),
    ],
    "eotw": [
        ("Mat", "Matrim"),
        ("Mat", "Matrim Cauthon"),
        ("Bran", "Brandelwyn al'Vere"),
    ],
    "uprooted": [("Agnieszka", "Nieshka"), ("Sarkan", "Dragon"), ("Solya", "Falcon")],
    "persuasion": [("Frederick", "Wentworth")],
    "foundation": [("Seldon", "Raven Seldon"), ("Seldon", "Hari Seldon")],
    "citycity": [("Tyador", "Borlú"), ("Tye", "Borlú")],
}
_DOCTYPE = re.compile(rb"<!DOCTYPE[^>\[]*(\[[^\]]*\])?[^>]*>", re.IGNORECASE)
_LINE = re.compile(
    r'^(\d+)\. (.+?) \((\d+) mentions(?:; also "(.*?)")?\): ', re.MULTILINE
)


def _norm(text: str) -> str:
    return text.casefold().replace("\u2019", "'").replace("\u2018", "'")


def _inspection(book: str, tmp_path: Path) -> kk.BookInspection:
    """Parse a copy with DOCTYPE removed: the parser rejects EPUB2 declarations."""
    source = LIBRARY / BOOKS[book]
    copy = tmp_path / f"{book}.epub"
    with (
        zipfile.ZipFile(source) as archive,
        zipfile.ZipFile(copy, "w", zipfile.ZIP_DEFLATED) as out,
    ):
        if "mimetype" in archive.namelist():
            out.writestr(
                zipfile.ZipInfo("mimetype"),
                archive.read("mimetype"),
                zipfile.ZIP_STORED,
            )
        for item in archive.infolist():
            if item.filename == "mimetype":
                continue
            data = archive.read(item.filename)
            if item.filename.lower().endswith(
                (".xhtml", ".html", ".htm", ".opf", ".ncx", ".xml")
            ):
                data = _DOCTYPE.sub(b"", data)
            out.writestr(item, data)
    return kk.epub(str(copy)).inspect()


class _Replay:
    """Returns the recorded answers, renumbered onto this prompt by alias set."""

    def __init__(self, fixture: Mapping[str, Any]) -> None:
        self.entries = [
            frozenset(cast("list[str]", entry))
            for entry in cast("list[object]", fixture["entries"])
        ]
        self.responses = cast("list[str]", fixture["responses"]).copy()

    def complete(self, model: str, prompt: str) -> str:  # noqa: ARG002
        here: dict[frozenset[str], int] = {}
        for match in _LINE.finditer(prompt):
            aliases = {match.group(2)} | (
                set(match.group(4).split(", ")) if match.group(4) else set()
            )
            here[frozenset(aliases)] = int(match.group(1))
        missing = [sorted(e) for e in self.entries if e not in here]
        assert not missing, f"base roster differs from the harness: {missing[:5]}"
        raw = self.responses.pop(0)
        payload = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        renumber = {old: here[entry] for old, entry in enumerate(self.entries, start=1)}
        payload["same_person"] = [
            [renumber[int(n)] for n in g if str(n).isdigit() and int(n) in renumber]
            for g in payload.get("same_person", [])
        ]
        payload["not_individuals"] = [
            renumber[int(n)]
            for n in payload.get("not_individuals", [])
            if str(n).isdigit() and int(n) in renumber
        ]
        return json.dumps(payload)


Score: TypeAlias = tuple[list[str], list[str], list[str], int]
CorpusResults: TypeAlias = dict[str, dict[str, Score]]


def _score(
    book: str,
    roster: Sequence[CharacterProfile],
    mentions: Mapping[str, int],
) -> Score:
    def words(character: CharacterProfile) -> set[str]:
        return {
            _norm(w)
            for a in (*character.aliases, character.display_name)
            for w in a.split()
        }

    kept = {a for c in roster for a in c.aliases}
    lost: list[str] = []
    for label, toks in PRINCIPALS[book].items():
        mine = [n for n in mentions if {_norm(w) for w in n.split()} & toks]
        total = sum(mentions[n] for n in mine)
        if not total or sum(mentions[n] for n in mine if n in kept) / total < 0.5:
            lost.append(label)
    merged = [
        c.display_name
        for c in roster
        if len([p for p, toks in PRINCIPALS[book].items() if toks & words(c)]) > 1
    ]
    apart = [
        f"{a}+{b}"
        for a, b in APART[book]
        for c in roster
        if {_norm(a), _norm(b)} <= {_norm(x) for x in (*c.aliases, c.display_name)}
    ]
    same = sum(
        1
        for a, b in SAME[book]
        if any(a in c.aliases and b in c.aliases for c in roster)
    )
    return lost, merged, apart, same


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> CorpusResults:
    out: CorpusResults = {}
    for book in BOOKS:
        inspection = _inspection(book, tmp_path_factory.mktemp(book))
        spans = {c.id: dialogue_ranges(build_grid(c)) for c in inspection.chapters}
        mentions = spacy_roster.collect_signals(inspection.chapters, spans).mentions
        offline, _ = spacy_roster.infer_roster(inspection.chapters, spans)
        fixture = cast(
            "dict[str, Any]", json.loads((FIXTURES / f"{book}.json").read_text())
        )
        identity = IdentityPass(
            fixture["model"], client=_Replay(fixture), backoff_base=0
        )
        online, _ = spacy_roster.infer_roster(
            inspection.chapters, spans, identity=identity
        )
        out[book] = {
            "offline": _score(book, offline, mentions),
            "online": _score(book, online, mentions),
        }
    return out


@pytest.mark.parametrize("path", ["offline", "online"])
def test_no_person_lost_merged_or_joined(results: CorpusResults, path: str) -> None:
    for book, scored in results.items():
        lost, merged, apart, _ = scored[path]
        assert (lost, merged, apart) == ([], [], []), f"{book} {path}"


def test_the_identity_pass_merges_nicknames(results: CorpusResults) -> None:
    assert sum(scored["online"][3] for scored in results.values()) >= 14


def test_the_fallback_matches_its_measured_baseline(results: CorpusResults) -> None:
    assert sum(scored["offline"][3] for scored in results.values()) >= 3
