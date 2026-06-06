"""voice_pool — persistent voice pool template for character voice assignment.

Stored at ~/.config/kenkui/voice_pool.toml

Template TOML schema:

  [protagonist.male]
  1 = "david"
  2 = "james"
  pool = ["oliver", "ethan"]

  [protagonist.female]
  1 = "sarah"
  pool = ["emma", "claire"]

  [protagonist.other]
  pool = ["alex"]

  [supporting.male]
  pool = ["oliver", "ethan", "marcus"]

  [supporting.female]
  pool = ["emma", "claire", "nina"]

  [supporting.other]
  pool = ["alex"]

  [minor]
  pool = []

Characters are ranked by quote_count within their role+gender group. Named slots
(1, 2, …) get specific voices; characters beyond the named slots use the pool
round-robin. Characters not covered by the template fall through to suggest_cast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

VOICE_POOL_PATH = Path.home() / ".config" / "kenkui" / "voice_pool.toml"

_GENDERS = ("male", "female", "other")


@dataclass
class GenderSlot:
    """Named rank assignments and overflow pool for a role+gender bucket."""

    named: dict[int, str] = field(default_factory=dict)  # 1-based rank → voice name
    pool: list[str] = field(default_factory=list)  # round-robin for unranked chars

    def pick(self, rank: int, pool_counter: dict[tuple, int], bucket_key: tuple) -> str | None:
        """Return the voice for the given 1-based rank. None if no assignment defined."""
        if rank in self.named:
            return self.named[rank]
        if self.pool:
            idx = pool_counter.get(bucket_key, 0)
            voice = self.pool[idx % len(self.pool)]
            pool_counter[bucket_key] = idx + 1
            return voice
        return None

    def is_empty(self) -> bool:
        return not self.named and not self.pool


@dataclass
class VoicePoolTemplate:
    """In-memory representation of voice_pool.toml."""

    protagonist: dict[str, GenderSlot] = field(
        default_factory=lambda: {g: GenderSlot() for g in _GENDERS}
    )
    supporting: dict[str, GenderSlot] = field(
        default_factory=lambda: {g: GenderSlot() for g in _GENDERS}
    )
    minor: GenderSlot = field(default_factory=GenderSlot)

    def is_empty(self) -> bool:
        """Return True if no voice assignments are defined anywhere."""
        for role in ("protagonist", "supporting"):
            bucket = self.protagonist if role == "protagonist" else self.supporting
            for g in _GENDERS:
                if not bucket.get(g, GenderSlot()).is_empty():
                    return False
        return self.minor.is_empty()

    def get_slot(self, role: str, gender: str) -> GenderSlot:
        """Return the GenderSlot for a given role+gender bucket."""
        role_n = _normalize_role(role)
        gender_n = _normalize_gender(gender)
        if role_n == "minor":
            return self.minor
        bucket = self.protagonist if role_n == "protagonist" else self.supporting
        return bucket.get(gender_n, GenderSlot())

    def to_dict(self) -> dict:
        d: dict = {}
        for role in ("protagonist", "supporting"):
            d[role] = {}
            bucket = self.protagonist if role == "protagonist" else self.supporting
            for g in _GENDERS:
                slot = bucket.get(g, GenderSlot())
                slot_d: dict = {}
                for rank in sorted(slot.named):
                    slot_d[str(rank)] = slot.named[rank]
                slot_d["pool"] = list(slot.pool)
                d[role][g] = slot_d
        d["minor"] = {"pool": list(self.minor.pool)}
        return d

    @classmethod
    def from_dict(cls, data: dict) -> VoicePoolTemplate:
        t = cls()
        for role in ("protagonist", "supporting"):
            if role not in data:
                continue
            bucket = t.protagonist if role == "protagonist" else t.supporting
            role_data = data[role]
            for g in _GENDERS:
                if g not in role_data:
                    continue
                slot_data = role_data[g]
                named: dict[int, str] = {}
                for k, v in slot_data.items():
                    if k != "pool" and str(k).isdigit():
                        named[int(k)] = str(v)
                pool = [str(x) for x in slot_data.get("pool", [])]
                bucket[g] = GenderSlot(named=named, pool=pool)
        if "minor" in data:
            t.minor = GenderSlot(pool=[str(x) for x in data["minor"].get("pool", [])])
        return t


def _normalize_role(role: str) -> str:
    r = (role or "").lower().strip()
    if r in ("protagonist", "antagonist", "lead", "main", "hero", "heroine"):
        return "protagonist"
    if r in ("supporting", "secondary", "side", "background"):
        return "supporting"
    return "minor"


def _normalize_gender(gender_pronoun: str) -> str:
    """Map a gender pronoun to 'male', 'female', or 'other'."""
    from kenkui.services.voice_service import gender_from_pronoun

    g = gender_from_pronoun(gender_pronoun)
    return "other" if g == "they" else g


def load_voice_pool_template(path: Path | None = None) -> VoicePoolTemplate:
    """Load voice pool template from TOML. Returns empty template if not found."""
    p = path or VOICE_POOL_PATH
    if not p.exists():
        return VoicePoolTemplate()
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        return VoicePoolTemplate.from_dict(data)
    except Exception:
        return VoicePoolTemplate()


def save_voice_pool_template(template: VoicePoolTemplate, path: Path | None = None) -> None:
    """Save voice pool template to TOML."""
    import tomli_w

    p = path or VOICE_POOL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(tomli_w.dumps(template.to_dict()).encode("utf-8"))


def auto_populate_from_voices(excluded: list[str] | None = None) -> VoicePoolTemplate:
    """Build a starter template from currently active (non-excluded) voices."""
    from kenkui.services.voice_service import list_voices

    excluded_set = set(excluded or [])
    all_voices = list_voices()
    active = [v for v in all_voices if v.name not in excluded_set and not v.excluded]

    male = [v.name for v in active if (v.gender or "").lower() == "male"]
    female = [v.name for v in active if (v.gender or "").lower() == "female"]
    other = [v.name for v in active if (v.gender or "").lower() not in ("male", "female")]

    t = VoicePoolTemplate()
    t.protagonist["male"] = GenderSlot(
        named={i + 1: v for i, v in enumerate(male[:2])},
        pool=male[2:],
    )
    t.protagonist["female"] = GenderSlot(
        named={i + 1: v for i, v in enumerate(female[:2])},
        pool=female[2:],
    )
    t.protagonist["other"] = GenderSlot(pool=list(other))
    t.supporting["male"] = GenderSlot(pool=list(male))
    t.supporting["female"] = GenderSlot(pool=list(female))
    t.supporting["other"] = GenderSlot(pool=list(other))
    t.minor = GenderSlot(pool=[])
    return t


__all__ = [
    "GenderSlot",
    "VoicePoolTemplate",
    "load_voice_pool_template",
    "save_voice_pool_template",
    "auto_populate_from_voices",
    "VOICE_POOL_PATH",
]
