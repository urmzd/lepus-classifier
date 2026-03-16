#!/usr/bin/env python3
"""Download all images from resources/data.csv into resources/images/{rabbit,hare}/."""

import csv
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

def extract_filename(link: str) -> str:
    regex = re.compile(r".*/(.*\.(jpeg|jpg|png)?)\??", flags=re.IGNORECASE)
    m = regex.match(link)
    if not m:
        raise ValueError(f"Cannot extract filename from {link}")
    name = m.group(1)
    if not m.group(2):
        name += ".png"
    return name

def download(link: str, dest: Path) -> bool:
    if dest.exists():
        return True
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    req = Request(link, headers=headers)
    try:
        with urlopen(req, timeout=15) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return False

def main():
    csv_path = Path(__file__).resolve().parent.parent / "resources" / "data.csv"
    images_dir = Path(__file__).resolve().parent.parent / "resources" / "images"

    for label_dir in ["rabbit", "hare"]:
        (images_dir / label_dir).mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            link = row["link"].strip().strip('"')
            if not link:
                continue
            filename = extract_filename(link)
            dest = images_dir / label / filename
            print(f"[{label}] {filename}... ", end="", flush=True)
            if download(link, dest):
                print("OK")
                ok += 1
            else:
                fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed")

if __name__ == "__main__":
    main()
