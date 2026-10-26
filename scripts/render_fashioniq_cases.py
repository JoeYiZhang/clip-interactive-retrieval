"""Render compact reference/target/top-1 audit sheets for FashionIQ cases."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.datasets import load_fashioniq  # noqa: E402


def _font(size: int):
    candidates = (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), (size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def _render(
    rows: list[dict],
    image_paths: dict[str, Path],
    output: Path,
    title: str,
) -> None:
    columns = ("Reference", "Target", "Previous B3 Top1", "Dynamic A2 Top1")
    size = 150
    gap = 12
    left = 18
    header_height = 70
    text_height = 68
    row_height = size + text_height + 22
    width = left * 2 + len(columns) * size + (len(columns) - 1) * gap
    height = header_height + len(rows) * row_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    label_font = _font(15)
    body_font = _font(13)
    draw.text((left, 12), title, fill="black", font=title_font)
    for column, label in enumerate(columns):
        x = left + column * (size + gap)
        draw.text((x, 44), label, fill="#333333", font=label_font)

    old_name = "B3_previous_val_tuned"
    new_name = "A2_train_tuned_dynamic"
    for row_index, row in enumerate(rows):
        y = header_height + row_index * row_height
        ids = (
            row["reference"],
            row["target"],
            row[f"{old_name}_top1"],
            row[f"{new_name}_top1"],
        )
        for column, image_id in enumerate(ids):
            x = left + column * (size + gap)
            canvas.paste(_thumbnail(image_paths[image_id], size), (x, y))
            draw.rectangle((x, y, x + size, y + size), outline="#bbbbbb", width=1)
            draw.text((x, y + size + 3), image_id, fill="#333333", font=body_font)
        old_rank = row[f"{old_name}_rank"]
        new_rank = row[f"{new_name}_rank"]
        line = (
            f"{row['query_id']} | B3 rank={old_rank if old_rank <= 50 else '>50'} | "
            f"A2 rank={new_rank if new_rank <= 50 else '>50'}"
        )
        draw.text((left, y + size + 22), line, fill="#111111", font=body_font)
        wrapped = textwrap.shorten(row["text"], width=105, placeholder="...")
        draw.text((left, y + size + 40), wrapped, fill="#555555", font=body_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=6)
    args = parser.parse_args()

    bundle = load_fashioniq(args.data_root, split="val")
    image_paths = {item.image_id: item.path for item in bundle.gallery}
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    outputs = {
        "rescued": args.output_dir / "fashioniq_dynamic_rescued.png",
        "regressed": args.output_dir / "fashioniq_dynamic_regressed.png",
    }
    _render(
        report["rescued_samples"][: args.case_count],
        image_paths,
        outputs["rescued"],
        "Dynamic gate: rescued into Top-10",
    )
    _render(
        report["regressed_samples"][: args.case_count],
        image_paths,
        outputs["regressed"],
        "Dynamic gate: regressed out of Top-10",
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
