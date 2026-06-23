#!/usr/bin/env python3
"""
Intelligent Audio Segmentation for Moonshine Curriculum Learning

Segments MLS French audiobook samples (10-20s) into curriculum-appropriate
lengths (4-10s) using Whisper Large V3 + Wav2Vec2 forced alignment.

Splits audio at natural phrase boundaries for linguistic coherence.

Author: Your Name
License: MIT
"""

import os
import argparse
import warnings
import numpy as np

# Fix PyTorch 2.6+ weights_only compatibility with pyannote.audio
# Use environment variable to globally disable weights_only=True default
# See: https://github.com/m-bain/whisperX/issues/1304
# See: https://github.com/pyannote/pyannote-audio/issues/1908
# See: https://docs.pytorch.org/docs/stable/miscellaneous_environment_variables.html
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import torch
from datasets import load_dataset, Dataset, DatasetDict, Audio
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import whisperx
except ImportError:
    print("ERROR: whisperx not installed. Install with: pip install whisperx")
    exit(1)


class IntelligentSegmenter:
    """Segment audio at natural phrase boundaries using Whisper + forced alignment."""

    def __init__(
        self,
        whisper_model: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "fr"
    ):
        """
        Initialize segmenter with Whisper model.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to use (cuda or cpu)
            compute_type: Precision (float16 for GPU, int8 for CPU)
            language: Language code (fr for French)
        """
        self.device = device
        self.language = language
        self.compute_type = compute_type

        # Check GPU memory if using CUDA
        if device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {vram_gb:.2f} GB")

            # Recommend model based on VRAM
            if whisper_model == "large-v3" and vram_gb < 12:
                print(f"\n[WARNING] large-v3 requires ~12GB VRAM, you have {vram_gb:.1f}GB")
                print("Recommended: Use 'medium' or 'small' model instead")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Exiting. Rerun with --whisper-model medium")
                    exit(0)

        print(f"\nLoading Whisper '{whisper_model}' on {device}...")
        try:
            self.model = whisperx.load_model(
                whisper_model,
                device,
                compute_type=compute_type,
                language=language
            )
            print("[OK] Whisper model loaded successfully")
        except Exception as e:
            print(f"\n[ERROR] Failed to load Whisper model: {e}")
            if "out of memory" in str(e).lower():
                print("\nTry a smaller model:")
                print("  --whisper-model small   (uses ~2GB VRAM)")
                print("  --whisper-model medium  (uses ~5GB VRAM)")
            exit(1)

        # Will load alignment model on first use
        self.align_model = None
        self.align_metadata = None

    def load_alignment_model(self, language_code: str = "fr"):
        """Load language-specific forced alignment model."""
        if self.align_model is None:
            print(f"\nLoading alignment model for {language_code}...")
            try:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device
                )
                print("[OK] Alignment model loaded successfully")
            except Exception as e:
                print(f"\n[ERROR] Failed to load alignment model: {e}")
                print("Continuing without forced alignment (lower precision timestamps)")
                self.align_model = None

    def transcribe_and_align(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict:
        """
        Transcribe audio and get word-level timestamps.

        Args:
            audio: Audio array
            sample_rate: Sample rate (default 16000)

        Returns:
            Dict with segments containing word-level alignments
        """
        # Transcribe with Whisper
        result = self.model.transcribe(
            audio,
            batch_size=16,
            language=self.language
        )

        # Load and apply alignment model if available
        if self.align_model is None:
            self.load_alignment_model(self.language)

        if self.align_model is not None:
            try:
                # Align transcript to get word-level timestamps
                aligned_result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )
                return aligned_result
            except Exception as e:
                print(f"[WARNING] Alignment failed: {e}")
                return result
        else:
            return result

    def find_best_split_points(
        self,
        words: List[Dict],
        target_duration: float = 7.5,
        min_duration: float = 4.0,
        max_duration: float = 10.0
    ) -> List[float]:
        """
        Find optimal split points that:
        1. Target ~7.5s segments (middle of 4-10s range)
        2. Split at word boundaries (not mid-word)
        3. Prefer splitting after punctuation
        4. Avoid very short/long segments

        Args:
            words: List of word dicts with 'start', 'end', 'word' keys
            target_duration: Target segment duration
            min_duration: Minimum allowed duration
            max_duration: Maximum allowed duration

        Returns:
            List of split timestamps
        """
        if not words or len(words) == 0:
            return []

        split_points = []
        segment_start = words[0].get('start', 0)

        for i, word in enumerate(words):
            word_start = word.get('start', 0)
            word_end = word.get('end', word_start + 0.1)
            current_duration = word_end - segment_start

            # Check if we should split here
            should_split = False

            # Must be within allowed range
            if current_duration >= min_duration:
                # Prefer splitting around target duration
                if current_duration >= target_duration - 1.0:
                    # Look ahead to see if waiting would be better
                    if i + 1 < len(words):
                        next_word = words[i + 1]
                        next_duration = next_word.get('end', word_end + 0.1) - segment_start

                        # Split if next would exceed max
                        if next_duration > max_duration:
                            should_split = True
                        # Split if current is closer to target
                        elif abs(current_duration - target_duration) < abs(next_duration - target_duration):
                            # Bonus for punctuation
                            word_text = word.get('word', '')
                            if any(p in word_text for p in ['.', '!', '?', ',', ';', ':']):
                                should_split = True
                            # Or if close enough to target
                            elif current_duration >= target_duration - 0.5:
                                should_split = True
                    else:
                        # Last word
                        should_split = True

            if should_split:
                split_points.append(word_end)
                segment_start = word_end

        return split_points

    def segment_audio(
        self,
        audio: np.ndarray,
        aligned_result: Dict,
        sample_rate: int = 16000,
        target_duration: float = 7.5,
        min_duration: float = 4.0,
        max_duration: float = 10.0
    ) -> List[Tuple[np.ndarray, str, float, float]]:
        """
        Segment audio at smart boundaries.

        Args:
            audio: Audio array
            aligned_result: Result from transcribe_and_align
            sample_rate: Audio sample rate
            target_duration: Target segment duration
            min_duration: Minimum segment duration
            max_duration: Maximum segment duration

        Returns:
            List of (audio_segment, transcript_segment, start_time, end_time)
        """
        segments = []

        for segment in aligned_result.get("segments", []):
            words = segment.get("words", [])
            if not words:
                # No word-level timestamps, use segment-level
                start_time = segment.get('start', 0)
                end_time = segment.get('end', start_time + 0.1)
                text = segment.get('text', '').strip()

                duration = end_time - start_time
                if min_duration <= duration <= max_duration:
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    audio_segment = audio[start_sample:end_sample]
                    segments.append((audio_segment, text, start_time, end_time))
                continue

            # Find split points within this segment
            split_points = self.find_best_split_points(
                words,
                target_duration=target_duration,
                min_duration=min_duration,
                max_duration=max_duration
            )

            # Create segments
            segment_start_idx = 0

            for split_time in split_points:
                # Find words in this subsegment
                subsegment_words = []
                for word in words[segment_start_idx:]:
                    if word.get('start', 0) < split_time:
                        subsegment_words.append(word)
                        segment_start_idx += 1
                    else:
                        break

                if subsegment_words:
                    start_time = subsegment_words[0].get('start', 0)
                    end_time = subsegment_words[-1].get('end', start_time + 0.1)

                    # Extract audio
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    audio_segment = audio[start_sample:end_sample]

                    # Reconstruct transcript
                    transcript = ' '.join([w.get('word', '') for w in subsegment_words])

                    segments.append((
                        audio_segment,
                        transcript.strip(),
                        start_time,
                        end_time
                    ))

            # Handle remaining words
            if segment_start_idx < len(words):
                remaining_words = words[segment_start_idx:]
                start_time = remaining_words[0].get('start', 0)
                end_time = remaining_words[-1].get('end', start_time + 0.1)
                duration = end_time - start_time

                if duration >= min_duration:
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    audio_segment = audio[start_sample:end_sample]
                    transcript = ' '.join([w.get('word', '') for w in remaining_words])
                    segments.append((audio_segment, transcript.strip(), start_time, end_time))

        return segments


def process_dataset(
    input_dataset,
    segmenter: IntelligentSegmenter,
    target_duration: float = 7.5,
    min_duration: float = 4.0,
    max_duration: float = 10.0,
    checkpoint_dir: str = None,
    checkpoint_interval: int = 100
):
    """
    Process entire dataset with intelligent segmentation using chunked memory management.

    Args:
        input_dataset: Input HuggingFace dataset
        segmenter: IntelligentSegmenter instance
        target_duration: Target segment duration
        min_duration: Minimum segment duration
        max_duration: Maximum segment duration
        checkpoint_dir: Directory to save checkpoints (None = no checkpointing)
        checkpoint_interval: Save checkpoint every N samples (also triggers chunk save to free RAM)

    Returns:
        Processed HuggingFace dataset
    """
    import pickle
    from pathlib import Path

    # Current chunk of data (cleared periodically to free RAM)
    new_data = {
        'audio': [],
        'transcript': [],
        'audio_duration': [],
        'original_id': [],
        'segment_index': [],
        'start_time': [],
        'end_time': []
    }

    start_idx = 0
    successful = 0
    failed = 0
    chunk_counter = 0

    # For incremental saving
    chunks_dir = None
    if checkpoint_dir:
        chunks_dir = Path(checkpoint_dir) / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if exists
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.pkl"
        if checkpoint_path.exists():
            print(f"\n[CHECKPOINT] Found existing checkpoint at {checkpoint_path}")
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                    start_idx = checkpoint['last_idx'] + 1
                    successful = checkpoint['successful']
                    failed = checkpoint['failed']
                    chunk_counter = checkpoint.get('chunk_counter', 0)

                print(f"[CHECKPOINT] Resuming from sample {start_idx:,}/{len(input_dataset):,}")
                print(f"[CHECKPOINT] Already processed: {successful:,} successful, {failed:,} failed")
                print(f"[CHECKPOINT] Existing chunks: {chunk_counter}")
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint: {e}")
                print("[WARNING] Starting from beginning")
                start_idx = 0

    print(f"\nProcessing {len(input_dataset):,} samples...")
    print(f"Target: {target_duration}s segments ({min_duration}s - {max_duration}s range)")
    if checkpoint_dir:
        print(f"Checkpointing: Every {checkpoint_interval} samples to {checkpoint_dir}")
    print("="*80)

    for idx, example in enumerate(tqdm(input_dataset, desc="Segmenting", initial=start_idx, total=len(input_dataset))):
        # Skip already processed samples
        if idx < start_idx:
            continue

        try:
            # Get audio and convert to float32 (NumPy 2.x uses float64 by default)
            audio_data = example['audio']['array'].astype(np.float32)
            sample_rate = example['audio']['sampling_rate']

            # Transcribe and align
            aligned_result = segmenter.transcribe_and_align(audio_data, sample_rate)

            # Segment intelligently
            segments = segmenter.segment_audio(
                audio_data,
                aligned_result,
                sample_rate,
                target_duration=target_duration,
                min_duration=min_duration,
                max_duration=max_duration
            )

            # Add segments to dataset
            for seg_idx, (audio_seg, transcript, start_time, end_time) in enumerate(segments):
                duration = len(audio_seg) / sample_rate

                # Double-check duration (should always pass if segment_audio works correctly)
                if min_duration <= duration <= max_duration:
                    # Store audio in format compatible with Audio feature
                    new_data['audio'].append({
                        'array': audio_seg,
                        'sampling_rate': sample_rate
                    })
                    new_data['transcript'].append(transcript)
                    new_data['audio_duration'].append(duration)
                    new_data['original_id'].append(example.get('id', f'sample_{idx}'))
                    new_data['segment_index'].append(seg_idx)
                    new_data['start_time'].append(start_time)
                    new_data['end_time'].append(end_time)

            successful += 1

        except Exception as e:
            failed += 1
            if failed <= 5:  # Only print first 5 errors
                print(f"\n[ERROR] Sample {idx}: {e}")
            continue

        # Save checkpoint periodically and clear memory
        if checkpoint_dir and (idx + 1) % checkpoint_interval == 0:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pkl"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Save current chunk to disk (frees RAM!)
                if len(new_data['audio']) > 0:
                    chunk_path = chunks_dir / f"chunk_{chunk_counter:05d}.pkl"
                    with open(chunk_path, 'wb') as f:
                        pickle.dump(new_data, f, protocol=4)

                    chunk_size_mb = chunk_path.stat().st_size / 1_000_000
                    print(f"\n[CHUNK] Saved chunk {chunk_counter} with {len(new_data['audio']):,} segments ({chunk_size_mb:.1f} MB)")

                    # Clear memory!
                    new_data = {
                        'audio': [],
                        'transcript': [],
                        'audio_duration': [],
                        'original_id': [],
                        'segment_index': [],
                        'start_time': [],
                        'end_time': []
                    }
                    chunk_counter += 1

                # Save metadata checkpoint (small file)
                checkpoint = {
                    'last_idx': idx,
                    'successful': successful,
                    'failed': failed,
                    'chunk_counter': chunk_counter
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f, protocol=4)

                checkpoint_size_mb = checkpoint_path.stat().st_size / 1_000_000
                print(f"[CHECKPOINT] Saved at sample {idx + 1:,}/{len(input_dataset):,}")
                print(f"[CHECKPOINT] Metadata: {checkpoint_size_mb:.1f} MB, Total chunks: {chunk_counter}")
            except Exception as e:
                print(f"\n[WARNING] Failed to save checkpoint: {e}")
                import traceback
                print(f"[WARNING] Error details: {traceback.format_exc()}")

    print(f"\n{'='*80}")
    print(f"Processing Summary:")
    print(f"  Successful: {successful:,}/{len(input_dataset):,}")
    print(f"  Failed: {failed:,}")
    print(f"{'='*80}")

    # Save final chunk if it has data
    if checkpoint_dir and len(new_data['audio']) > 0:
        chunk_path = chunks_dir / f"chunk_{chunk_counter:05d}.pkl"
        with open(chunk_path, 'wb') as f:
            pickle.dump(new_data, f, protocol=4)
        print(f"\n[CHUNK] Saved final chunk {chunk_counter} with {len(new_data['audio']):,} segments")
        chunk_counter += 1

    # Merge all chunks into final dataset
    print(f"\n{'='*80}")
    print(f"Merging {chunk_counter} chunks into final dataset...")
    print(f"{'='*80}")

    merged_data = {
        'audio': [],
        'transcript': [],
        'audio_duration': [],
        'original_id': [],
        'segment_index': [],
        'start_time': [],
        'end_time': []
    }

    if checkpoint_dir and chunks_dir.exists():
        chunk_files = sorted(chunks_dir.glob("chunk_*.pkl"))
        for chunk_file in chunk_files:
            print(f"Loading {chunk_file.name}...")
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
                for key in merged_data.keys():
                    merged_data[key].extend(chunk_data[key])
            print(f"  Loaded {len(chunk_data['audio']):,} segments (total: {len(merged_data['audio']):,})")
    else:
        # No chunking was used (shouldn't happen if checkpoint_dir is set)
        merged_data = new_data

    # Create dataset
    if len(merged_data['audio']) == 0:
        print("\n[ERROR] No segments created! Check your audio files and parameters.")
        return None

    processed_dataset = Dataset.from_dict(merged_data)

    # Cast audio column
    processed_dataset = processed_dataset.cast_column(
        'audio',
        Audio(sampling_rate=16000)
    )

    print(f"\nSegmentation Results:")
    print(f"  Input samples: {len(input_dataset):,}")
    print(f"  Output segments: {len(processed_dataset):,}")
    print(f"  Multiplication factor: {len(processed_dataset)/len(input_dataset):.2f}x")
    print(f"  Duration range: {min(merged_data['audio_duration']):.2f}s - {max(merged_data['audio_duration']):.2f}s")
    print(f"  Mean duration: {np.mean(merged_data['audio_duration']):.2f}s")
    print(f"  Median duration: {np.median(merged_data['audio_duration']):.2f}s")

    # Duration distribution
    durations = np.array(merged_data['audio_duration'])
    ranges = [
        (4.0, 6.0, "4-6s"),
        (6.0, 8.0, "6-8s"),
        (8.0, 10.0, "8-10s")
    ]
    print(f"\n  Duration Distribution:")
    for min_d, max_d, label in ranges:
        count = np.sum((durations >= min_d) & (durations < max_d))
        pct = 100 * count / len(durations)
        print(f"    {label}: {count:6,} ({pct:5.1f}%)")

    print(f"{'='*80}\n")

    return processed_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Intelligent audio segmentation for curriculum learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 100 samples using medium model (recommended for RTX 2060)
  python scripts/intelligent_segmentation.py --test-mode --whisper-model medium

  # Process full dataset with small model (safer for 6GB GPU)
  python scripts/intelligent_segmentation.py --whisper-model small

  # Use CPU (slow but works on any machine)
  python scripts/intelligent_segmentation.py --device cpu --whisper-model medium --test-mode
        """
    )
    parser.add_argument('--output-dir', default='./data/mls_french_intelligent',
                       help='Output directory for processed dataset')
    parser.add_argument('--cache-dir',
                       default='Z:/HarryPotterCrowdsourcing/Moonshine paper/.cache/huggingface',
                       help='HuggingFace cache directory')
    parser.add_argument('--target-duration', type=float, default=7.5,
                       help='Target segment duration in seconds (default: 7.5)')
    parser.add_argument('--min-duration', type=float, default=4.0,
                       help='Minimum segment duration in seconds (default: 4.0)')
    parser.add_argument('--max-duration', type=float, default=10.0,
                       help='Maximum segment duration in seconds (default: 10.0)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Process only 100 samples for testing')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--whisper-model', default='medium',
                       choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                       help='Whisper model size (default: medium for RTX 2060)')
    parser.add_argument('--language', default='fr',
                       help='Language code (default: fr for French)')
    parser.add_argument('--checkpoint-dir', default=None,
                       help='Directory to save checkpoints for resume capability (default: None = no checkpointing)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Save checkpoint every N samples (default: 100)')

    args = parser.parse_args()

    # Set environment variables for HuggingFace cache
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['HF_HUB_CACHE'] = args.cache_dir
    os.environ['HF_DATASETS_CACHE'] = f"{args.cache_dir}/datasets"

    print("="*80)
    print("INTELLIGENT AUDIO SEGMENTATION FOR MOONSHINE CURRICULUM LEARNING")
    print("="*80)
    print(f"Whisper Model: {args.whisper_model}")
    print(f"Device: {args.device}")
    print(f"Language: {args.language}")
    print(f"Target Duration: {args.target_duration}s")
    print(f"Duration Range: [{args.min_duration}s - {args.max_duration}s]")
    print(f"Test Mode: {'YES (100 samples)' if args.test_mode else 'NO (full dataset)'}")
    print("="*80)

    # Determine compute type based on device
    if args.device == "cpu":
        compute_type = "int8"
    else:
        compute_type = "float16"

    # Load dataset
    print("\nLoading Multilingual LibriSpeech French...")
    split = "train[:100]" if args.test_mode else "train"

    try:
        dataset = load_dataset(
            'facebook/multilingual_librispeech',
            'french',
            split=split,
            cache_dir=f"{args.cache_dir}/datasets"
        )
        print(f"[OK] Loaded {len(dataset):,} samples")
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        exit(1)

    # Initialize segmenter
    try:
        segmenter = IntelligentSegmenter(
            whisper_model=args.whisper_model,
            device=args.device,
            compute_type=compute_type,
            language=args.language
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize segmenter: {e}")
        exit(1)

    # Process dataset
    processed = process_dataset(
        dataset,
        segmenter,
        target_duration=args.target_duration,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval
    )

    if processed is None:
        print("\n[ERROR] Processing failed. Exiting.")
        exit(1)

    # Save dataset
    print(f"Saving to {args.output_dir}...")
    try:
        processed.save_to_disk(args.output_dir)
        print(f"[OK] Dataset saved successfully")
    except Exception as e:
        print(f"\n[ERROR] Failed to save dataset: {e}")
        exit(1)

    # Clean up checkpoint on successful completion
    if args.checkpoint_dir:
        from pathlib import Path
        import shutil
        checkpoint_path = Path(args.checkpoint_dir) / "checkpoint.pkl"
        chunks_dir = Path(args.checkpoint_dir) / "chunks"

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[CHECKPOINT] Cleaned up checkpoint metadata file")

        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
            print(f"[CHECKPOINT] Cleaned up {chunks_dir} directory with all chunk files")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Segmented dataset saved to: {args.output_dir}")
    print(f"Total segments: {len(processed):,}")
    print(f"\nTo use this dataset in training, update your config:")
    print(f"""
dataset:
  type: "local"
  path: "{args.output_dir}"
""")
    print("="*80)


if __name__ == '__main__':
    main()
