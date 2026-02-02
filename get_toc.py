#!/usr/bin/env python3
"""
Extract the raw Table of Contents from an EPUB file.
This script extracts the TOC in its rawest form by reading the NCX or NAV file directly.
"""

import zipfile
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def find_toc_file(epub_path):
    """
    Find the TOC file (NCX or NAV) in the EPUB.
    Returns tuple of (file_path, toc_type) where toc_type is 'ncx' or 'nav'
    """
    with zipfile.ZipFile(epub_path, "r") as epub:
        # First, find the content.opf file
        namelist = epub.namelist()

        # Look for container.xml to find the OPF file
        container_path = "META-INF/container.xml"
        if container_path in namelist:
            container_xml = epub.read(container_path)
            container_tree = ET.fromstring(container_xml)

            # Find the OPF file path
            ns = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
            rootfile = container_tree.find(".//container:rootfile", ns)
            if rootfile is not None:
                opf_path = rootfile.get("full-path")

                # Read the OPF file
                opf_xml = epub.read(opf_path)
                opf_tree = ET.fromstring(opf_xml)

                # Look for NCX file reference
                opf_ns = {"opf": "http://www.idpf.org/2007/opf"}
                ncx_item = opf_tree.find(
                    ".//opf:item[@media-type='application/x-dtbncx+xml']", opf_ns
                )

                if ncx_item is not None:
                    ncx_href = ncx_item.get("href")
                    # Resolve relative path
                    opf_dir = str(Path(opf_path).parent)
                    if opf_dir == ".":
                        ncx_path = ncx_href
                    else:
                        ncx_path = str(Path(opf_dir) / ncx_href)
                    return (ncx_path, "ncx")

                # Look for NAV file (EPUB3)
                nav_item = opf_tree.find(".//opf:item[@properties='nav']", opf_ns)
                if nav_item is not None:
                    nav_href = nav_item.get("href")
                    opf_dir = str(Path(opf_path).parent)
                    if opf_dir == ".":
                        nav_path = nav_href
                    else:
                        nav_path = str(Path(opf_dir) / nav_href)
                    return (nav_path, "nav")

        # Fallback: search for common TOC file names
        for name in namelist:
            if name.endswith(".ncx"):
                return (name, "ncx")
            if "nav.xhtml" in name.lower() or "toc.xhtml" in name.lower():
                return (name, "nav")

    return (None, None)


def extract_raw_toc(epub_path):
    """
    Extract the raw TOC content from an EPUB file.
    """
    toc_file, toc_type = find_toc_file(epub_path)

    if toc_file is None:
        print("No TOC file found in EPUB")
        return None

    print(f"Found TOC file: {toc_file} (type: {toc_type})")
    print("=" * 80)
    print()

    with zipfile.ZipFile(epub_path, "r") as epub:
        toc_content = epub.read(toc_file).decode("utf-8")
        return toc_content


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_epub_toc.py <epub_file>")
        print("\nThis script extracts the raw Table of Contents from an EPUB file.")
        print("It will output the raw NCX or NAV file content.")
        sys.exit(1)

    epub_path = sys.argv[1]

    if not Path(epub_path).exists():
        print(f"Error: File '{epub_path}' not found")
        sys.exit(1)

    raw_toc = extract_raw_toc(epub_path)

    if raw_toc:
        print(raw_toc)


if __name__ == "__main__":
    main()
