"""Download the MovingAI MAPF benchmark maps used in this project into
data/movingai/ (they are already included in the repository; this script
documents their origin and lets you fetch additional maps).

Source: Sturtevant, "Benchmarks for Grid-Based Pathfinding", IEEE TCIAIG 2012,
https://movingai.com/benchmarks/mapf.html  (file mapf-map.zip)

Usage:  python scripts/download_movingai.py [map names ...]
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://movingai.com/benchmarks/mapf/mapf-map.zip"
DEFAULT = ["empty-32-32.map", "random-32-32-10.map", "room-32-32-4.map", "maze-32-32-4.map",
           "room-64-64-8.map", "den312d.map", "random-64-64-10.map", "warehouse-10-20-10-2-1.map"]


def main() -> None:
    names = sys.argv[1:] or DEFAULT
    out = Path(__file__).resolve().parents[1] / "data" / "movingai"
    out.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in names:
            (out / n).write_bytes(z.read(n))
            print("saved", out / n)


if __name__ == "__main__":
    main()
