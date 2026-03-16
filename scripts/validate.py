#!/usr/bin/env python3
"""Minimal validation: 1 fold, 2 epochs, small batch — confirms the pipeline runs end-to-end."""

import os
import sys

# Disable W&B for validation runs.
os.environ["WANDB_MODE"] = "disabled"

# Ensure the project root is on sys.path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pathlib import Path

from src.bootstrap import bootstrap, BasicModel, TrainerFactory, LEARNING_RATE

IMAGE_FOLDER_PATH = Path(project_root) / "resources" / "images"

def main():
    print("Running validation (1 fold, 2 epochs)...")

    # Build subdirectory mapping: the data_handler expects a flat image folder,
    # but our images are in rabbit/ and hare/ subdirs. We need to flatten them
    # into a single directory via symlinks for compatibility.
    flat_dir = Path("/tmp/lepus_validation_images")
    flat_dir.mkdir(parents=True, exist_ok=True)

    for label_dir in IMAGE_FOLDER_PATH.iterdir():
        if not label_dir.is_dir():
            continue
        for img in label_dir.iterdir():
            dest = flat_dir / img.name
            if not dest.exists():
                dest.symlink_to(img)

    bootstrap(
        model=BasicModel(learning_rate=LEARNING_RATE),
        image_folder_path=flat_dir,
        num_folds=1,
        batch_size=2,
        trainer_factory=TrainerFactory(max_epochs=2),
        trainer_kwargs={"enable_progress_bar": True},
    )
    print("Validation complete!")

if __name__ == "__main__":
    main()
