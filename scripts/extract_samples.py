#!/usr/bin/env python3
"""
Extract audio samples from segmented dataset with transcriptions.

Exports samples as .wav files with accompanying text files containing transcriptions.
"""

import argparse
import os
from pathlib import Path
from datasets import load_from_disk
import soundfile as sf


def extract_samples(
    dataset_path: str,
    output_dir: str,
    num_samples: int = 10,
    start_index: int = 0
):
    """
    Extract audio samples and transcriptions from dataset.

    Args:
        dataset_path: Path to the segmented dataset
        output_dir: Directory to save extracted samples
        num_samples: Number of samples to extract
        start_index: Starting index in the dataset
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    print(f"Dataset size: {len(dataset):,} samples")
    print(f"Extracting {num_samples} samples starting from index {start_index}...")
    print("="*80)

    # Determine end index
    end_index = min(start_index + num_samples, len(dataset))
    actual_samples = end_index - start_index

    # Extract samples
    for i in range(start_index, end_index):
        sample = dataset[i]

        # Get audio data
        audio_array = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']

        # Get metadata
        transcript = sample['transcript']
        duration = sample['audio_duration']
        original_id = sample.get('original_id', f'sample_{i}')
        segment_idx = sample.get('segment_index', 0)

        # Create filename
        safe_id = str(original_id).replace('/', '_').replace('\\', '_')
        filename = f"{i:04d}_{safe_id}_seg{segment_idx}"

        # Save audio as .wav
        wav_path = output_path / f"{filename}.wav"
        sf.write(wav_path, audio_array, sample_rate)

        # Save transcription
        txt_path = output_path / f"{filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript)

        # Save metadata
        meta_path = output_path / f"{filename}_meta.txt"
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f"Original ID: {original_id}\n")
            f.write(f"Segment Index: {segment_idx}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Sample Rate: {sample_rate} Hz\n")
            f.write(f"Transcript: {transcript}\n")

        print(f"[{i+1-start_index:3d}/{actual_samples}] {filename}.wav ({duration:.2f}s)")

    print("="*80)
    print(f"Extracted {actual_samples} samples to: {output_path}")
    print(f"\nFiles created:")
    print(f"  - {actual_samples} .wav files (audio)")
    print(f"  - {actual_samples} .txt files (transcriptions)")
    print(f"  - {actual_samples} _meta.txt files (metadata)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio samples from segmented dataset'
    )
    parser.add_argument(
        '--dataset-path',
        default='./data/mls_french_segmented_test',
        help='Path to the segmented dataset'
    )
    parser.add_argument(
        '--output-dir',
        default='./extracted_samples',
        help='Output directory for extracted samples'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to extract (default: 10)'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Starting index in the dataset (default: 0)'
    )

    args = parser.parse_args()

    extract_samples(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_index=args.start_index
    )


if __name__ == '__main__':
    main()
