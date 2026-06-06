from kenkui.nlp.models import (
    CharacterRecord as NLPCharacterRecord,
)
from kenkui.nlp.models import (
    CharacterRoster,
    TitleRecord,
    slugify,
)


def test_slugify_basic():
    assert slugify("Elizabeth Bennet") == "elizabeth_bennet"


def test_slugify_apostrophe():
    assert slugify("Rand al'Thor") == "rand_althor"


def test_slugify_dots_and_spaces():
    assert slugify("Mr. Darcy") == "mr_darcy"


def test_slugify_multiple_spaces():
    assert slugify("The  Dark  Lord") == "the_dark_lord"


def test_slugify_already_lower():
    assert slugify("frodo") == "frodo"


def test_title_record_default_book_slug():
    t = TitleRecord(title="Mr.", chapters=[0, 1, 2])
    assert t.book_slug is None


def test_title_record_with_book_slug():
    t = TitleRecord(title="Queen of Andor", chapters=[5, 6], book_slug="eye_of_world")
    assert t.title == "Queen of Andor"
    assert t.chapters == [5, 6]
    assert t.book_slug == "eye_of_world"


def test_character_record_pydantic_construction():
    c = NLPCharacterRecord(
        slug="rand_althor",
        canonical_name="Rand al'Thor",
        aliases=["The Dragon Reborn", "Lews Therin"],
        titles=[TitleRecord(title="Dragon Reborn", chapters=[5, 6], book_slug="eye_of_world")],
        gender="he/him",
        role="protagonist",
        description="The Dragon Reborn, destined to face the Dark One.",
        chapters=[0, 1, 2, 5, 6],
        first_appearance=("eye_of_world", 0),
        last_appearance=("eye_of_world", 50),
        mention_count=500,
        quote_count=200,
    )
    assert c.slug == "rand_althor"
    assert c.role == "protagonist"
    assert len(c.titles) == 1


def test_character_record_defaults():
    c = NLPCharacterRecord(slug="frodo", canonical_name="Frodo Baggins")
    assert c.aliases == []
    assert c.titles == []
    assert c.gender == ""
    assert c.role == ""
    assert c.description == ""
    assert c.chapters == []
    assert c.first_appearance is None
    assert c.last_appearance is None
    assert c.mention_count == 0
    assert c.quote_count == 0


def test_character_roster_by_slug():
    c = NLPCharacterRecord(slug="frodo", canonical_name="Frodo Baggins")
    roster = CharacterRoster(characters=[c])
    assert roster.by_slug("frodo") is c
    assert roster.by_slug("gandalf") is None


def test_character_roster_all_slugs():
    c1 = NLPCharacterRecord(slug="frodo", canonical_name="Frodo Baggins")
    c2 = NLPCharacterRecord(slug="sam_gamgee", canonical_name="Sam Gamgee")
    roster = CharacterRoster(characters=[c1, c2])
    assert set(roster.all_slugs()) == {"frodo", "sam_gamgee"}
