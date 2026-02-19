"""Fix audiobook functionality - scans for missing chapters, salvages existing audio, and fixes metadata."""

import sys
import re
import shutil
import subprocess
import tempfile
import io
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from difflib import SequenceMatcher
from contextlib import contextmanager

import imageio_ffmpeg
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from .helpers import Config, Chapter
from .parsing import EpubReader
from .utils import extract_epub_cover, batch_text

try:
    from mutagen.mp4 import MP4, MP4Cover

    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


def _embed_cover_in_m4b(m4b_path: Path, cover_data: bytes, mime_type: str) -> bool:
    """Embed cover image into M4B file using mutagen."""
    if not MUTAGEN_AVAILABLE:
        print(
            "Error: mutagen library not available for cover embedding", file=sys.stderr
        )
        return False

    try:
        image_format = (
            MP4Cover.FORMAT_PNG if mime_type == "image/png" else MP4Cover.FORMAT_JPEG
        )
        audio = MP4(str(m4b_path))
        audio["covr"] = [MP4Cover(cover_data, imageformat=image_format)]
        audio.save()

        print(f"Successfully embedded cover into {m4b_path}")
        return True

    except Exception as e:
        print(f"Error embedding cover: {e}", file=sys.stderr)
        return False


def extract_m4b_chapters(m4b_path: Path) -> List[Dict[str, Any]]:
    """Extract chapter information from M4B file using ffprobe."""
    import json as json_module

    chapters: list[dict[str, Any]] = []

    try:
        # Use ffprobe to get chapter information
        cmd = [
            "ffprobe",
            "-print_format",
            "json",
            "-show_chapters",
            "-loglevel",
            "error",
            str(m4b_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"Error running ffprobe: {result.stderr}", file=sys.stderr)
            return chapters

        data = json_module.loads(result.stdout)
        chapter_data = data.get("chapters", [])

        for i, chapter in enumerate(chapter_data):
            # ffprobe returns start/end in seconds as floats
            start_sec = float(chapter.get("start", 0))
            end_sec = float(chapter.get("end", 0))

            # Convert to milliseconds
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)

            # Get title from tags
            title = chapter.get("tags", {}).get("title", "")

            chapters.append(
                {"title": title, "start_ms": start_ms, "end_ms": end_ms, "index": i}
            )

    except subprocess.TimeoutExpired:
        print("Error: ffprobe timed out", file=sys.stderr)
    except json_module.JSONDecodeError:
        print("Error: Invalid JSON from ffprobe", file=sys.stderr)
    except Exception as e:
        print(f"Error extracting chapters from M4B: {e}", file=sys.stderr)

    return chapters


def fuzzy_title_match(title1: str, title2: str, threshold: float = 0.6) -> bool:
    """Fuzzy match two titles to see if they refer to the same chapter."""
    if not title1 or not title2:
        return False

    if title1.strip().lower() == title2.strip().lower():
        return True

    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    normalized1 = normalize(title1)
    normalized2 = normalize(title2)

    if len(normalized1) > len(normalized2):
        if normalized2 in normalized1:
            return True
    else:
        if normalized1 in normalized2:
            return True

    similarity = SequenceMatcher(None, normalized1, normalized2).ratio()
    return similarity >= threshold


def find_matching_chapter(
    epub_title: str, m4b_chapters: List[Dict[str, Any]], threshold: float = 0.6
) -> Optional[Dict[str, Any]]:
    """Find the best matching M4B chapter for an EPUB chapter."""
    best_match = None
    best_score = 0

    for m4b_chapter in m4b_chapters:
        m4b_title = m4b_chapter.get("title", "")

        if fuzzy_title_match(epub_title, m4b_title, threshold):
            similarity = SequenceMatcher(
                None,
                re.sub(r"[^a-z0-9]", "", epub_title.lower()),
                re.sub(r"[^a-z0-9]", "", m4b_title.lower()),
            ).ratio()

            if similarity > best_score:
                best_score = similarity
                best_match = m4b_chapter

    return best_match


def extract_audio_segment(
    m4b_path: Path, start_ms: int, end_ms: int, temp_dir: Path
) -> Optional[bytes]:
    """Extract an audio segment from M4B file using ffmpeg."""
    try:
        output_file = temp_dir / "segment.wav"

        start_sec: float = start_ms / 1000.0
        duration: float = (end_ms - start_ms) / 1000.0

        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-i",
            str(m4b_path),
            "-ss",
            str(start_sec),
            "-t",
            str(duration),
            "-c:a",
            "pcm_s16le",
            "-ar",
            "24000",
            str(output_file),
        ]

        subprocess.run(cmd, capture_output=True, timeout=120)

        if output_file.exists():
            with open(output_file, "rb") as f:
                audio_data = f.read()
            output_file.unlink()
            return audio_data
        else:
            return None

    except Exception as e:
        print(f"Error extracting audio segment: {e}", file=sys.stderr)
        return None


@contextmanager
def managed_temp_dir():
    """Context manager for temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def extract_epub_chapters(epub_path: Path) -> List[Chapter]:
    """Extract chapters from EPUB file using existing EpubReader."""
    try:
        reader = EpubReader(epub_path)
        return reader.extract_chapters()
    except Exception as e:
        print(f"Error extracting EPUB chapters: {e}", file=sys.stderr)
        return []


def prepare_chapter_audio(
    epub_chapters: List[Chapter], m4b_path: Path, temp_dir: Path, console: Console
) -> Tuple[List[Tuple[Chapter, Optional[bytes]]], List[int]]:
    """Match EPUB chapters to M4B chapters and extract salvaged audio."""
    console.print("[bold blue]Analyzing audiobook chapters...")

    m4b_chapters = extract_m4b_chapters(m4b_path)

    if m4b_chapters:
        console.print(f"[green]Found {len(m4b_chapters)} chapters in audiobook[/green]")
    else:
        console.print(
            "[yellow]No chapters found in audiobook - will generate all[/yellow]"
        )

    chapters_with_audio: list[tuple[Chapter, bytes | None]] = []
    missing_indices: list[int] = []
    matched_count = 0

    for chapter in epub_chapters:
        audio_data = None

        if m4b_chapters:
            m4b_chapter = find_matching_chapter(chapter.title, m4b_chapters)

            if m4b_chapter:
                audio_data = extract_audio_segment(
                    m4b_path, m4b_chapter["start_ms"], m4b_chapter["end_ms"], temp_dir
                )

                if audio_data:
                    matched_count += 1

        if audio_data:
            chapters_with_audio.append((chapter, audio_data))
        else:
            chapters_with_audio.append((chapter, None))
            missing_indices.append(chapter.index)

    if matched_count > 0:
        console.print(
            f"[green]Salvaged {matched_count} chapters from existing audiobook[/green]"
        )

    if missing_indices:
        console.print(f"[yellow]Missing {len(missing_indices)} chapters[/yellow]")
    else:
        console.print("[green]All chapters present![/green]")

    return chapters_with_audio, missing_indices


def generate_missing_audio(
    missing_chapters: List[Chapter], config: Config, console: Console
) -> Dict[int, bytes]:
    """Generate audio for missing chapters using pocket-tts."""
    if not missing_chapters:
        return {}

    console.print(
        f"[bold yellow]Generating {len(missing_chapters)} missing chapters..."
    )

    try:
        from pydub import AudioSegment
        import scipy.io.wavfile
        from pocket_tts import TTSModel
        from .workers import _load_voice

        console.print("Loading TTS model...")
        model = TTSModel.load_model()

        voice = _load_voice(config.voice, verbose=config.verbose)
        voice_state = model.get_state_for_audio_prompt(voice)

        pause_line_ms = getattr(config, "pause_line_ms", 400)
        pause_chapter_ms = getattr(config, "pause_chapter_ms", 2000)

        generated_audio = {}

        for chapter in missing_chapters:
            console.print(f"Generating: {chapter.title}")

            full_audio = AudioSegment.empty()
            silence = AudioSegment.silent(duration=pause_line_ms)

            batches = batch_text(chapter.paragraphs)

            for batch in batches:
                try:
                    audio_tensor = model.generate_audio(
                        voice_state,
                        batch,
                    )

                    if audio_tensor is not None and audio_tensor.numel() > 0:
                        if audio_tensor.dim() > 1:
                            if audio_tensor.dim() == 2:
                                audio_tensor = audio_tensor.squeeze()
                            elif audio_tensor.dim() == 3:
                                audio_tensor = audio_tensor.view(-1)

                        if audio_tensor.dim() == 1:
                            wav_buffer = io.BytesIO()
                            scipy.io.wavfile.write(
                                wav_buffer, model.sample_rate, audio_tensor.numpy()
                            )
                            wav_buffer.seek(0)
                            batch_audio = AudioSegment.from_wav(wav_buffer)
                            full_audio += batch_audio + silence

                except Exception as e:
                    console.print(f"[red]Error generating batch: {e}[/red]")

            full_audio += AudioSegment.silent(duration=pause_chapter_ms)

            wav_buffer = io.BytesIO()
            full_audio.export(wav_buffer, format="wav")
            generated_audio[chapter.index] = wav_buffer.getvalue()

            console.print(f"[green]Generated: {chapter.title}[/green]")

        return generated_audio

    except Exception as e:
        console.print(f"[red]Error generating audio: {e}[/red]")
        import traceback

        traceback.print_exc()
        return {}


def rebuild_audiobook(
    chapters_with_audio: List[Tuple[Chapter, bytes]],
    output_path: Path,
    bitrate: str = "64k",
) -> bool:
    """Rebuild audiobook with all chapters in proper order."""
    try:
        from pydub import AudioSegment

        console = Console()
        console.print("[bold blue]Rebuilding audiobook...")

        temp_dir = Path(tempfile.mkdtemp())
        try:
            file_list = temp_dir / "files.txt"
            meta_file = temp_dir / "metadata.txt"

            # Write audio files (skip None audio data)
            valid_chapters: list[tuple[Chapter, bytes]] = [
                (ch, audio) for ch, audio in chapters_with_audio if audio is not None
            ]
            for idx, (chapter, audio_data) in enumerate(valid_chapters):
                audio_file = temp_dir / f"ch_{idx:04d}.wav"
                with open(audio_file, "wb") as f:
                    f.write(audio_data)

                with open(file_list, "a") as f:
                    f.write(f"file '{audio_file.resolve().as_posix()}'\n")

            # Write metadata
            with open(meta_file, "w", encoding="utf-8") as f:
                f.write(";FFMETADATA1\n")
                t = 0
                for idx, (chapter, audio_data) in enumerate(valid_chapters):
                    # Get duration from audio
                    audio = AudioSegment.from_wav(io.BytesIO(audio_data))
                    duration_ms = len(audio)

                    start, end = int(t), int(t + duration_ms)
                    f.write(
                        f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle={chapter.title}\n"
                    )
                    t += duration_ms

            # Build command
            cmd = [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-stats",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(file_list),
                "-i",
                str(meta_file),
                "-map_metadata",
                "1",
                "-c:a",
                "aac",
                "-b:a",
                bitrate,
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if result.returncode != 0:
                console.print(f"[red]FFmpeg error: {result.stderr.decode()}[/red]")
                return False

            console.print(f"[green]Audiobook rebuilt: {output_path}[/green]")
            return True

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    except Exception as e:
        console.print(f"[red]Error rebuilding audiobook: {e}[/red]")
        import traceback

        traceback.print_exc()
        return False


def fix_audiobook(
    epub_path: Path, audio_path: Path, config: Config, console: Console
) -> bool:
    """
    Fix audiobook by scanning for missing chapters, salvaging existing audio, and fixing metadata.

    Args:
        epub_path: Path to EPUB file
        audio_path: Path to existing audiobook file
        config: Configuration
        console: Rich console for output

    Returns:
        True if successful, False otherwise
    """
    console.print("[bold blue]Fixing audiobook...")
    console.print(f"EPUB: {epub_path}")
    console.print(f"Audiobook: {audio_path}")

    # Extract cover from EPUB
    cover_data, mime_type = extract_epub_cover(epub_path)

    # Extract EPUB chapters using existing EpubReader class
    epub_chapters = extract_epub_chapters(epub_path)

    if not epub_chapters:
        console.print("[red]No chapters found in EPUB[/red]")
        return False

    console.print(f"[bold]Found {len(epub_chapters)} chapters in EPUB[/bold]")

    # Create temporary directory for processing
    with managed_temp_dir() as temp_dir:
        # Match chapters and extract salvaged audio
        chapters_with_audio, missing_indices = prepare_chapter_audio(
            epub_chapters, audio_path, temp_dir, console
        )

        # Ask user if they want to generate missing chapters
        if missing_indices:
            console.print(
                Panel(
                    f"[yellow]Found {len(missing_indices)} missing chapters[/yellow]\n"
                    f"Missing chapters: {', '.join(str(i) for i in missing_indices[:10])}"
                    + ("..." if len(missing_indices) > 10 else ""),
                    title="Missing Chapters",
                    border_style="yellow",
                )
            )

            if (
                Prompt.ask(
                    "[bold yellow]Generate missing chapters? (y/n)[/bold yellow]",
                    default="y",
                ).lower()
                != "y"
            ):
                console.print("[yellow]Skipping chapter generation[/yellow]")
            else:
                # Generate missing chapters
                missing_chapters = [
                    ch for ch in epub_chapters if ch.index in missing_indices
                ]
                generated_audio = generate_missing_audio(
                    missing_chapters, config, console
                )

                # Fill in missing audio data
                for idx, audio_data in generated_audio.items():
                    for i, (chapter, _) in enumerate(chapters_with_audio):
                        if chapter.index == idx:
                            chapters_with_audio[i] = (chapter, audio_data)
                            break

        # Sort by chapter index to ensure proper order
        chapters_with_audio.sort(key=lambda x: x[0].index)

        # Remove any chapters without audio
        chapters_with_audio = [
            (ch, audio)
            for ch, audio in chapters_with_audio
            if audio is not None  # type: ignore[misc]
        ]

        if not chapters_with_audio:
            console.print("[red]No chapter audio available[/red]")
            return False

        # Generate output path (with _fixed suffix)
        output_path = audio_path.parent / f"{audio_path.stem}_fixed{audio_path.suffix}"

        # Rebuild audiobook
        if not rebuild_audiobook(chapters_with_audio, output_path, config.m4b_bitrate):
            return False

        # Embed cover if available
        if cover_data and mime_type:
            success = _embed_cover_in_m4b(output_path, cover_data, mime_type)
            if success:
                console.print("[green]Cover embedded successfully[/green]")

        console.print(f"[bold green]Audiobook fixed: {output_path}[/bold green]")
        console.print("[dim]Original file preserved as backup[/dim]")

    return True


# Export for use in main
__all__ = ["fix_audiobook"]
