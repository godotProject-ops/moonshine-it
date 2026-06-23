#!/usr/bin/env python3
"""
Convert checkpoint pickle file to HuggingFace dataset format.

This allows using partially processed data before segmentation completes.
Uses RAM-efficient batch processing.
"""

import pickle
import argparse
import numpy as np
from pathlib import Path
from datasets import Dataset, Audio, concatenate_datasets


def checkpoint_to_dataset(checkpoint_dir: str, output_dir: str, live: bool = False, max_chunks: int = None, batch_size: int = 25):
    """
    Load checkpoint and convert to HuggingFace dataset using RAM-efficient batching.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_dir: Directory to save HuggingFace dataset
        live: If True, extract data while segmentation is running (safe, read-only)
        max_chunks: Maximum number of chunks to process (None = all)
        batch_size: Number of chunks to load at once (default: 25, ~5k segments, ~2.5GB RAM)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if live:
        print("\n" + "="*80)
        print("WARNING: LIVE EXTRACTION MODE")
        print("="*80)
        print("Extracting data while segmentation is still running.")
        print("This is SAFE - we only read completed chunks.")
        print("Segmentation can continue running without interruption.")
        print("="*80 + "\n")

    # Check for new chunked format
    chunks_dir = checkpoint_dir / "chunks"
    if chunks_dir.exists():
        print(f"[OK] Found chunked checkpoint format in {chunks_dir}")
        chunk_files = sorted(chunks_dir.glob("chunk_*.pkl"))
        print(f"[OK] Found {len(chunk_files)} chunk files")

        # In live mode, skip the last chunk (might be incomplete)
        if live and len(chunk_files) > 0:
            chunk_files = chunk_files[:-1]
            print(f"[LIVE] Using {len(chunk_files)} complete chunks (skipping last one for safety)")

        # Limit chunks if requested
        if max_chunks is not None and len(chunk_files) > max_chunks:
            chunk_files = chunk_files[:max_chunks]
            print(f"[LIMIT] Using first {max_chunks} chunks only")

        if len(chunk_files) == 0:
            print("[WARNING] No complete chunks available yet!")
            print("Wait for segmentation to complete at least one checkpoint interval.")
            return None

        # Process chunks in batches to avoid RAM overflow
        # Save each batch to disk immediately to avoid memory buildup
        print(f"\nProcessing {len(chunk_files)} chunks in batches of {batch_size}...")
        print(f"RAM-efficient mode: Loading {batch_size} chunks, saving to disk, repeat")
        print("="*80)

        import tempfile
        import shutil
        from datasets import load_from_disk

        # Create temporary directory for batch datasets on same drive as output
        # This avoids filling up system drive (C:) with large temp files
        output_parent = Path(output_dir).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        temp_dir = output_parent / f"temp_batches_{Path(tempfile.mktemp()).name}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using temporary directory: {temp_dir}")
        print(f"(Using same drive as output to avoid filling system drive)")

        batch_dirs = []
        total_segments = 0

        try:
            for batch_start in range(0, len(chunk_files), batch_size):
                batch_end = min(batch_start + batch_size, len(chunk_files))
                batch_chunks = chunk_files[batch_start:batch_end]

                batch_num = batch_start//batch_size + 1
                total_batches = (len(chunk_files) + batch_size - 1)//batch_size

                print(f"\nBatch {batch_num}/{total_batches}:")
                print(f"  Processing chunks {batch_start} to {batch_end-1} ({len(batch_chunks)} chunks)")

                # Load this batch into memory
                batch_data = {
                    'audio': [],
                    'transcript': [],
                    'audio_duration': [],
                    'original_id': [],
                    'segment_index': [],
                    'start_time': [],
                    'end_time': []
                }

                for i, chunk_file in enumerate(batch_chunks):
                    try:
                        with open(chunk_file, 'rb') as f:
                            chunk_data = pickle.load(f)
                            for key in batch_data.keys():
                                batch_data[key].extend(chunk_data[key])

                        if (i + 1) % 10 == 0 or i == len(batch_chunks) - 1:
                            print(f"    Loaded {i+1}/{len(batch_chunks)} chunks... ({len(batch_data['audio']):,} segments)")

                    except Exception as e:
                        print(f"    [WARNING] Failed to load {chunk_file.name}: {e}")
                        if live:
                            print(f"    [LIVE] Skipping this chunk (might be currently being written)")
                            continue
                        else:
                            raise

                if len(batch_data['audio']) == 0:
                    print(f"    [WARNING] Batch {batch_num} is empty, skipping")
                    continue

                # Convert batch to dataset and save to disk immediately
                print(f"  Converting batch to HuggingFace dataset...")
                batch_dataset = Dataset.from_dict(batch_data)
                batch_dataset = batch_dataset.cast_column('audio', Audio(sampling_rate=16000))

                batch_dir = temp_dir / f"batch_{batch_num:04d}"
                print(f"  Saving batch to disk: {batch_dir.name}...")
                batch_dataset.save_to_disk(str(batch_dir))

                batch_dirs.append(batch_dir)
                total_segments += len(batch_dataset)

                print(f"  Batch complete: {len(batch_dataset):,} segments saved to disk")
                print(f"  Total so far: {total_segments:,} segments")

                # Clear batch data to free RAM
                del batch_data
                del batch_dataset

            # Load and concatenate all batch datasets from disk
            print(f"\n{'='*80}")
            print(f"Loading and concatenating {len(batch_dirs)} batch datasets from disk...")
            print(f"{'='*80}")

            datasets = []
            for batch_dir in batch_dirs:
                print(f"Loading {batch_dir.name}...")
                batch_ds = load_from_disk(str(batch_dir))
                datasets.append(batch_ds)

            print(f"Concatenating...")
            final_dataset = concatenate_datasets(datasets)

            print(f"[OK] Final dataset created: {len(final_dataset):,} total segments")

            # Clear intermediate datasets
            del datasets

        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                print(f"\nCleaning up temporary directory...")
                shutil.rmtree(temp_dir)
                print(f"[OK] Temporary files deleted")

    else:
        # Try old formats (backward compatibility)
        print("[ERROR] Old format not supported in RAM-efficient mode")
        print("Please use the chunked format from the new segmentation script")
        return None

    # Load metadata for stats
    checkpoint_path = checkpoint_dir / "checkpoint.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            print(f"\nCheckpoint stats:")
            print(f"  Last processed index: {checkpoint.get('last_idx', 'N/A')}")
            print(f"  Successful: {checkpoint.get('successful', 'N/A')}")
            print(f"  Failed: {checkpoint.get('failed', 'N/A')}")
            if 'chunk_counter' in checkpoint:
                print(f"  Total chunks: {checkpoint.get('chunk_counter', 'N/A')}")

    # Validate data
    if len(final_dataset) == 0:
        print("\n[ERROR] No segments in checkpoint!")
        return None

    print(f"\nDataset statistics:")
    print(f"  Total segments: {len(final_dataset):,}")

    durations = final_dataset['audio_duration']
    print(f"  Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
    print(f"  Mean duration: {np.mean(durations):.2f}s")
    print(f"  Median duration: {np.median(durations):.2f}s")

    # Duration distribution
    durations_array = np.array(durations)
    ranges = [
        (4.0, 6.0, "4-6s"),
        (6.0, 8.0, "6-8s"),
        (8.0, 10.0, "8-10s")
    ]
    print(f"\nDuration Distribution:")
    for min_d, max_d, label in ranges:
        count = np.sum((durations_array >= min_d) & (durations_array < max_d))
        pct = 100 * count / len(durations)
        print(f"  {label}: {count:6,} ({pct:5.1f}%)")

    # Sample transcripts
    print(f"\nSample Transcripts:")
    for i in range(min(3, len(final_dataset))):
        transcript = final_dataset[i]['transcript']
        duration = final_dataset[i]['audio_duration']
        print(f"  [{i+1}] ({duration:.1f}s): {transcript[:80]}...")

    # Save to disk
    print(f"\nSaving to {output_dir}...")
    final_dataset.save_to_disk(output_dir)
    print(f"[OK] Dataset saved successfully!")

    return final_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Convert checkpoint pickle to HuggingFace dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert completed checkpoint to dataset (RAM-efficient batching)
  python scripts/checkpoint_to_dataset.py \\
    --checkpoint-dir ./checkpoints \\
    --output-dir ./data/mls_french_final

  # Extract partial data WHILE segmentation is still running (SAFE!)
  python scripts/checkpoint_to_dataset.py \\
    --checkpoint-dir ./checkpoints \\
    --output-dir ./data/mls_french_partial \\
    --live

  # Extract first 200 chunks only (for quick tests)
  python scripts/checkpoint_to_dataset.py \\
    --checkpoint-dir ./checkpoints \\
    --output-dir ./data/mls_french_test \\
    --max-chunks 200

  # Use smaller batches if RAM is very limited (default: 25)
  python scripts/checkpoint_to_dataset.py \\
    --checkpoint-dir ./checkpoints \\
    --output-dir ./data/mls_french_partial \\
    --live \\
    --batch-size 10

  # Then use in training config:
  dataset:
    type: "local"
    path: "./data/mls_french_partial"
        """
    )
    parser.add_argument('--checkpoint-dir', required=True,
                       help='Directory containing checkpoint files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for HuggingFace dataset')
    parser.add_argument('--live', action='store_true',
                       help='Extract data while segmentation is still running (uses existing chunks only)')
    parser.add_argument('--max-chunks', type=int, default=None,
                       help='Maximum number of chunks to process (useful for quick tests with partial data)')
    parser.add_argument('--batch-size', type=int, default=25,
                       help='Number of chunks to load at once (default: 25, ~5k segments, ~2.5GB RAM)')

    args = parser.parse_args()

    print("="*80)
    print("CHECKPOINT TO DATASET CONVERTER (RAM-EFFICIENT)")
    print("="*80)

    if args.live:
        print("\nWARNING: Running in LIVE mode - extracting partial data while segmentation runs")
    if args.max_chunks:
        print(f"\nWARNING: Limited to first {args.max_chunks} chunks")

    print(f"\nBatch size: {args.batch_size} chunks (~{args.batch_size * 200} segments, ~{args.batch_size * 100 / 1000:.1f} GB RAM)")

    dataset = checkpoint_to_dataset(
        args.checkpoint_dir,
        args.output_dir,
        live=args.live,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size
    )

    if dataset is None:
        print("\n[ERROR] Conversion failed!")
        exit(1)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Dataset saved to: {args.output_dir}")
    print(f"Total segments: {len(dataset):,}")
    print(f"\nTo use in training, update your config:")
    print(f"""
dataset:
  type: "local"
  path: "{args.output_dir}"
""")
    print("="*80)


if __name__ == '__main__':
    main()
