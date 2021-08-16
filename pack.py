#!/usr/bin/env python3
import argparse
import zipfile
from pathlib import Path


def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="making submission file zip",
        usage="python3 pack.py",
        description="""
        This module demonstrates zipping submission file.
        """,
        add_help=True,
    )

    parser.add_argument("config", type=str, help="path of a config file")

    return parser.parse_args()


if __name__ == "__main__":
    parser = get_parser()
    model_name = parser.config.split("/")[-1].split(".")[0]
    png_file_dir = Path("submission_segmentation/submission_{:s}".format(model_name))
    zip_filename = "submission_segmentation/submission_{:s}.zip".format(model_name)
    print(zip_filename)

    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filepath in png_file_dir.glob("**/*.png"):
            if filepath.is_file():
                zf.write(filepath, arcname=filepath)
                print("  ", filepath)
