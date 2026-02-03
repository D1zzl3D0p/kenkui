#!/usr/bin/env python3
"""
Script to extract cover image from EPUB and embed it into M4B audiobook file.
"""

import zipfile
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_epub_cover(epub_path):
    """
    Extract the cover image from an EPUB file.

    Args:
        epub_path: Path to the EPUB file

    Returns:
        tuple: (cover_image_data, mime_type)
    """
    with zipfile.ZipFile(epub_path, "r") as epub:
        # Try to find container.xml
        try:
            container = epub.read("META-INF/container.xml")
            tree = ET.fromstring(container)

            # Find the OPF file path
            ns = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
            rootfile = tree.find(".//container:rootfile", ns)
            opf_path = rootfile.get("full-path")

            # Parse the OPF file
            opf_content = epub.read(opf_path)
            opf_tree = ET.fromstring(opf_content)

            # Common namespaces
            namespaces = {
                "opf": "http://www.idpf.org/2007/opf",
                "dc": "http://purl.org/dc/elements/1.1/",
            }

            # Try to find cover in metadata
            cover_id = None

            # Method 1: Look for meta tag with name="cover"
            for meta in opf_tree.findall('.//opf:meta[@name="cover"]', namespaces):
                cover_id = meta.get("content")
                break

            # Method 2: Look for item with properties="cover-image"
            if not cover_id:
                for item in opf_tree.findall(
                    './/opf:item[@properties="cover-image"]', namespaces
                ):
                    cover_id = item.get("id")
                    break

            # Find the actual image file
            if cover_id:
                for item in opf_tree.findall(".//opf:item", namespaces):
                    if item.get("id") == cover_id:
                        cover_href = item.get("href")
                        mime_type = item.get("media-type", "")
                        # Resolve relative path
                        opf_dir = os.path.dirname(opf_path)
                        cover_path = os.path.join(opf_dir, cover_href).replace(
                            "\\", "/"
                        )

                        # Read the cover image
                        cover_data = epub.read(cover_path)

                        # Determine MIME type from extension if not provided
                        if not mime_type:
                            ext = os.path.splitext(cover_path)[1].lower()
                            mime_type = {
                                ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg",
                                ".png": "image/png",
                            }.get(ext, "image/jpeg")

                        return cover_data, mime_type

            # Fallback: Look for common cover image names
            for name in epub.namelist():
                lower_name = name.lower()
                if "cover" in lower_name and any(
                    lower_name.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]
                ):
                    cover_data = epub.read(name)
                    ext = os.path.splitext(name)[1].lower()
                    mime_type = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                    }.get(ext, "image/jpeg")
                    return cover_data, mime_type

        except Exception as e:
            print(f"Error extracting cover: {e}", file=sys.stderr)
            return None, None

    return None, None


def embed_cover_in_m4b(m4b_path, cover_data, mime_type):
    """
    Embed cover image into M4B file using mutagen.

    Args:
        m4b_path: Path to the M4B file
        cover_data: Binary data of the cover image
        mime_type: MIME type of the cover image
    """
    try:
        from mutagen.mp4 import MP4, MP4Cover

        # Determine image format
        if mime_type == "image/png":
            image_format = MP4Cover.FORMAT_PNG
        else:
            image_format = MP4Cover.FORMAT_JPEG

        # Open the M4B file
        audio = MP4(m4b_path)

        # Add the cover art
        audio["covr"] = [MP4Cover(cover_data, imageformat=image_format)]

        # Save the file
        audio.save()

        print(f"Successfully embedded cover into {m4b_path}")
        return True

    except ImportError:
        print("Error: mutagen library not found.", file=sys.stderr)
        print("Please install it with: pip install mutagen", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error embedding cover: {e}", file=sys.stderr)
        return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python epub_to_m4b_cover.py <epub_file> <m4b_file>")
        sys.exit(1)

    epub_path = sys.argv[1]
    m4b_path = sys.argv[2]

    # Validate files exist
    if not os.path.exists(epub_path):
        print(f"Error: EPUB file not found: {epub_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(m4b_path):
        print(f"Error: M4B file not found: {m4b_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting cover from {epub_path}...")
    cover_data, mime_type = extract_epub_cover(epub_path)

    if not cover_data:
        print("Error: Could not extract cover from EPUB file", file=sys.stderr)
        sys.exit(1)

    print(f"Found cover image ({mime_type})")
    print(f"Embedding cover into {m4b_path}...")

    success = embed_cover_in_m4b(m4b_path, cover_data, mime_type)

    if success:
        print("Done!")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
