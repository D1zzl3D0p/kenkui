from kenkui.models import CharacterRecord, CharacterInfo
from kenkui.nlp.models import (
    CharacterRecord as NLPCharacterRecord,
    TitleRecord,
)


def _make_nlp_record() -> NLPCharacterRecord:
    return NLPCharacterRecord(
        slug="elizabeth_bennet",
        canonical_name="Elizabeth Bennet",
        aliases=["Lizzy", "Miss Bennet"],
        titles=[TitleRecord(title="Miss", chapters=[0, 1], book_slug=None)],
        gender="she/her",
        role="protagonist",
        description="Witty and independent heroine.",
        chapters=[0, 1, 2],
        first_appearance=("pride_prejudice", 0),
        last_appearance=("pride_prejudice", 60),
        mention_count=150,
        quote_count=80,
    )


def test_character_record_from_nlp():
    nlp_rec = _make_nlp_record()
    rec = CharacterRecord.from_nlp(nlp_rec)
    assert rec.slug == "elizabeth_bennet"
    assert rec.canonical_name == "Elizabeth Bennet"
    assert rec.role == "protagonist"
    assert rec.mention_count == 150
    assert rec.first_appearance == ("pride_prejudice", 0)


def test_character_record_to_character_info():
    rec = CharacterRecord.from_nlp(_make_nlp_record())
    info = rec.to_character_info()
    assert isinstance(info, CharacterInfo)
    # character_id is now the slug (used as speaker_voices lookup key)
    assert info.character_id == "elizabeth_bennet"
    assert info.display_name == "Elizabeth Bennet"
    assert info.mention_count == 150
    assert info.quote_count == 80
    # gender_pronoun is the first word of gender ("she/her" -> "she")
    assert info.gender_pronoun == "she"


def test_character_record_to_character_info_no_pronouns():
    nlp_rec = NLPCharacterRecord(slug="gollum", canonical_name="Gollum", gender="")
    rec = CharacterRecord.from_nlp(nlp_rec)
    info = rec.to_character_info()
    assert info.gender_pronoun == ""


def test_character_record_prominence():
    nlp_rec = NLPCharacterRecord(
        slug="frodo", canonical_name="Frodo Baggins",
        mention_count=200, quote_count=50
    )
    rec = CharacterRecord.from_nlp(nlp_rec)
    info = rec.to_character_info()
    assert info.prominence == 200  # prefers mention_count
