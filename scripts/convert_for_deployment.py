#!/usr/bin/env python3
"""
Complete Moonshine Model Conversion Pipeline

Publication-ready conversion script that handles the entire deployment pipeline:
1. Tokenizer vocabulary extension (32000 → 32768 if needed)
2. Model embedding resizing (to match extended tokenizer)
3. ONNX export (encoder + decoder + decoder_with_past)
4. Merged decoder creation (for C++ compatibility)
5. Tokenizer binarization (tokenizer.json → tokenizer.bin)
6. ORT format conversion (optimized for inference)

This creates deployment-ready models compatible with:
- Moonshine C++ inference (https://github.com/usefulsensors/moonshine)
- ONNX Runtime
- Mobile/embedded devices

Usage:
    # Full pipeline with defaults
    python convert_for_deployment.py \\
        --model ./results-moonshine-fr-no-curriculum/final \\
        --output ./deployment/moonshine-fr-v1

    # Skip tokenizer extension (if already 32768 tokens)
    python convert_for_deployment.py \\
        --model ./model --output ./deployment \\
        --skip-tokenizer-extension

    # Platform-specific optimization
    python convert_for_deployment.py \\
        --model ./model --output ./deployment \\
        --target-platform arm
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    MoonshineForConditionalGeneration,
    AutoProcessor
)
from optimum.exporters.onnx import main_export
from optimum.onnx.graph_transformations import merge_decoders


class ModelConverter:
    """Complete model conversion pipeline."""

    def __init__(
        self,
        model_path: str,
        output_dir: Path,
        target_vocab_size: int = 32768,
        skip_tokenizer_extension: bool = False,
        skip_embedding_resize: bool = False,
        skip_onnx_export: bool = False,
        skip_merged_decoder: bool = False,
        skip_tokenizer_bin: bool = False,
        skip_ort_conversion: bool = False,
        target_platform: Optional[str] = None
    ):
        """
        Initialize converter.

        Args:
            model_path: Path to trained model
            output_dir: Output directory for converted model
            target_vocab_size: Target vocabulary size (32768 for C++ compat)
            skip_*: Flags to skip individual steps
            target_platform: 'arm' or 'amd64' for platform-specific ORT optimization
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.target_vocab_size = target_vocab_size
        self.skip_tokenizer_extension = skip_tokenizer_extension
        self.skip_embedding_resize = skip_embedding_resize
        self.skip_onnx_export = skip_onnx_export
        self.skip_merged_decoder = skip_merged_decoder
        self.skip_tokenizer_bin = skip_tokenizer_bin
        self.skip_ort_conversion = skip_ort_conversion
        self.target_platform = target_platform

        # Intermediate paths
        self.tokenizer_extended_dir = self.output_dir / "tokenizer_extended"
        self.model_resized_dir = self.output_dir / "model_resized"
        self.onnx_dir = self.output_dir / "onnx"
        self.onnx_merged_dir = self.output_dir / "onnx_merged"
        self.ort_dir = self.output_dir / "ort"

        self.tools_dir = Path(__file__).parent / "src" / "tools"

    def print_step(self, step_num: int, total_steps: int, description: str):
        """Print formatted step header."""
        print(f"\n{'='*80}")
        print(f"STEP {step_num}/{total_steps}: {description}")
        print(f"{'='*80}\n")

    def extend_tokenizer(self) -> bool:
        """
        Extend tokenizer vocabulary to 32768 tokens.

        Returns:
            True if successful or skipped
        """
        if self.skip_tokenizer_extension:
            print("Skipping tokenizer extension (--skip-tokenizer-extension)")
            return True

        print(f"Loading tokenizer from: {self.model_path}")
        processor = AutoProcessor.from_pretrained(str(self.model_path))

        current_vocab_size = processor.tokenizer.vocab_size
        print(f"Current vocab size: {current_vocab_size}")

        if current_vocab_size >= self.target_vocab_size:
            print(f"Tokenizer already has {current_vocab_size} tokens (>= {self.target_vocab_size})")
            print("Skipping extension.")
            return True

        # Use existing extend_tokenizer_vocab tool
        script_path = self.tools_dir / "extend_tokenizer_vocab.py"

        if not script_path.exists():
            print(f"Warning: {script_path} not found")
            print("Attempting inline extension...")

            # Inline extension as fallback
            self.tokenizer_extended_dir.mkdir(parents=True, exist_ok=True)

            tokenizer_json_path = self.model_path / "tokenizer.json"
            with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)

            vocab = tokenizer_data['model']['vocab']
            tokens_to_add = self.target_vocab_size - len(vocab)
            max_id = max(vocab.values())

            for i in range(tokens_to_add):
                vocab[f"<reserved_{i}>"] = max_id + 1 + i

            tokenizer_data['model']['vocab'] = vocab

            output_json = self.tokenizer_extended_dir / "tokenizer.json"
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

            # Copy other files
            for file in self.model_path.glob("*.json"):
                if file.name != "tokenizer.json":
                    shutil.copy(file, self.tokenizer_extended_dir / file.name)

            print(f"Extended tokenizer saved to: {self.tokenizer_extended_dir}")
            return True

        # Use external script
        cmd = [
            sys.executable,
            str(script_path),
            "--model", str(self.model_path),
            "--output", str(self.tokenizer_extended_dir),
            "--target-size", str(self.target_vocab_size)
        ]

        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0

    def resize_embeddings(self) -> bool:
        """
        Resize model embeddings to match extended tokenizer.

        Returns:
            True if successful or skipped
        """
        if self.skip_embedding_resize:
            print("Skipping embedding resize (--skip-embedding-resize)")
            # Copy model to resized dir
            shutil.copytree(self.model_path, self.model_resized_dir, dirs_exist_ok=True)
            return True

        # Determine tokenizer path
        if self.skip_tokenizer_extension:
            tokenizer_path = self.model_path
        else:
            tokenizer_path = self.tokenizer_extended_dir

        print(f"Loading model from: {self.model_path}")
        model = MoonshineForConditionalGeneration.from_pretrained(str(self.model_path))

        print(f"Loading tokenizer from: {tokenizer_path}")
        processor = AutoProcessor.from_pretrained(str(tokenizer_path))

        current_vocab_size = model.config.vocab_size
        tokenizer_vocab_size = processor.tokenizer.vocab_size

        print(f"Model vocab size: {current_vocab_size}")
        print(f"Tokenizer vocab size: {tokenizer_vocab_size}")

        if current_vocab_size == tokenizer_vocab_size:
            print("Model and tokenizer vocab sizes match!")
            print(f"Saving to {self.model_resized_dir}...")
            self.model_resized_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(self.model_resized_dir))
            processor.save_pretrained(str(self.model_resized_dir))
            return True

        print(f"Resizing embeddings from {current_vocab_size} to {tokenizer_vocab_size}...")

        # Resize
        model.resize_token_embeddings(tokenizer_vocab_size)
        model.config.vocab_size = tokenizer_vocab_size

        # Verify
        decoder_embed_size = model.model.decoder.embed_tokens.num_embeddings
        print(f"Verification:")
        print(f"  Decoder embeddings: {decoder_embed_size}")
        print(f"  Model config vocab: {model.config.vocab_size}")

        if decoder_embed_size != tokenizer_vocab_size:
            print("Error: Size mismatch after resizing!")
            return False

        # Save
        self.model_resized_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving resized model to: {self.model_resized_dir}")
        model.save_pretrained(str(self.model_resized_dir))
        processor.save_pretrained(str(self.model_resized_dir))

        print(f"[OK] Model resized: {current_vocab_size} → {tokenizer_vocab_size} tokens")
        return True

    def export_to_onnx(self) -> bool:
        """
        Export model to ONNX format.

        Returns:
            True if successful or skipped
        """
        if self.skip_onnx_export:
            print("Skipping ONNX export (--skip-onnx-export)")
            return True

        print(f"Exporting model to ONNX...")
        print(f"Input: {self.model_resized_dir}")
        print(f"Output: {self.onnx_dir}")

        self.onnx_dir.mkdir(parents=True, exist_ok=True)

        # Export with decoder_with_past for KV cache
        main_export(
            model_name_or_path=str(self.model_resized_dir),
            output=str(self.onnx_dir),
            task="automatic-speech-recognition-with-past",
            opset=17,
        )

        # Verify files
        encoder_path = self.onnx_dir / "encoder_model.onnx"
        decoder_path = self.onnx_dir / "decoder_model.onnx"
        decoder_with_past_path = self.onnx_dir / "decoder_with_past_model.onnx"

        if not all([encoder_path.exists(), decoder_path.exists(), decoder_with_past_path.exists()]):
            print("Error: Expected ONNX files not generated!")
            return False

        print(f"\n[OK] ONNX export successful!")
        print(f"Generated files:")
        print(f"  - encoder_model.onnx ({encoder_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  - decoder_model.onnx ({decoder_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  - decoder_with_past_model.onnx ({decoder_with_past_path.stat().st_size / 1024 / 1024:.1f} MB)")

        return True

    def create_merged_decoder(self) -> bool:
        """
        Create merged decoder for C++ compatibility.

        Returns:
            True if successful or skipped
        """
        if self.skip_merged_decoder:
            print("Skipping merged decoder creation (--skip-merged-decoder)")
            # Copy ONNX files to merged dir
            shutil.copytree(self.onnx_dir, self.onnx_merged_dir, dirs_exist_ok=True)
            return True

        print("Creating merged decoder...")

        decoder_path = self.onnx_dir / "decoder_model.onnx"
        decoder_with_past_path = self.onnx_dir / "decoder_with_past_model.onnx"

        if not decoder_path.exists() or not decoder_with_past_path.exists():
            print("Error: Decoder ONNX files not found!")
            return False

        self.onnx_merged_dir.mkdir(parents=True, exist_ok=True)
        merged_decoder_path = self.onnx_merged_dir / "decoder_model_merged.onnx"

        # Merge decoders
        merged_model = merge_decoders(
            decoder=str(decoder_path),
            decoder_with_past=str(decoder_with_past_path),
            graph_name="merged_decoder",
            producer_name="moonshine-fine-tuned",
            save_path=str(merged_decoder_path),
            strict=False  # Moonshine has cross-attention differences
        )

        # Copy encoder and config files
        encoder_path = self.onnx_dir / "encoder_model.onnx"
        if encoder_path.exists():
            shutil.copy(encoder_path, self.onnx_merged_dir / "encoder_model.onnx")

        for config_file in ["config.json", "generation_config.json", "preprocessor_config.json",
                           "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src = self.onnx_dir / config_file
            if src.exists():
                shutil.copy(src, self.onnx_merged_dir / config_file)

        print(f"\n[OK] Merged decoder created!")
        print(f"  - decoder_model_merged.onnx ({merged_decoder_path.stat().st_size / 1024 / 1024:.1f} MB)")

        return True

    def convert_tokenizer_to_bin(self) -> bool:
        """
        Convert tokenizer.json to binary format.

        Returns:
            True if successful or skipped
        """
        if self.skip_tokenizer_bin:
            print("Skipping tokenizer binarization (--skip-tokenizer-bin)")
            return True

        tokenizer_json = self.onnx_merged_dir / "tokenizer.json"

        if not tokenizer_json.exists():
            print(f"Warning: {tokenizer_json} not found, skipping tokenizer binarization")
            return True

        tokenizer_bin = self.onnx_merged_dir / "tokenizer.bin"

        script_path = self.tools_dir / "convert_tokenizer_to_bin.py"

        if not script_path.exists():
            print(f"Warning: {script_path} not found, skipping tokenizer binarization")
            return True

        print(f"Converting tokenizer to binary format...")

        cmd = [
            sys.executable,
            str(script_path),
            str(tokenizer_json),
            "--output", str(tokenizer_bin)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"[OK] Tokenizer binary created!")
            print(f"  - tokenizer.bin ({tokenizer_bin.stat().st_size / 1024:.1f} KB)")
            return True
        else:
            print(f"Warning: Tokenizer binarization failed")
            if result.stderr:
                print(result.stderr)
            return True  # Non-fatal

    def convert_to_ort(self) -> bool:
        """
        Convert ONNX models to ORT format.

        Returns:
            True if successful or skipped
        """
        if self.skip_ort_conversion:
            print("Skipping ORT conversion (--skip-ort-conversion)")
            return True

        print("Converting to ORT format...")

        script_path = self.tools_dir / "convert_to_ort.py"

        if not script_path.exists():
            print(f"Warning: {script_path} not found")
            print("Attempting direct conversion...")

            # Direct conversion using onnxruntime
            cmd = [
                sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort",
                str(self.onnx_merged_dir),
                "--output_dir", str(self.ort_dir),
                "--optimization_style", "Fixed"
            ]

            if self.target_platform:
                cmd.extend(["--target_platform", self.target_platform])

            result = subprocess.run(cmd, capture_output=False)
            return result.returncode == 0

        # Use external script
        cmd = [
            sys.executable,
            str(script_path),
            str(self.onnx_merged_dir),
            "--output-dir", str(self.ort_dir),
            "--optimization-style", "Fixed"
        ]

        if self.target_platform:
            cmd.extend(["--target-platform", self.target_platform])

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            # Copy tokenizer.bin if exists
            tokenizer_bin_src = self.onnx_merged_dir / "tokenizer.bin"
            if tokenizer_bin_src.exists():
                shutil.copy(tokenizer_bin_src, self.ort_dir / "tokenizer.bin")

            return True

        return False

    def run(self) -> bool:
        """
        Run complete conversion pipeline.

        Returns:
            True if all steps succeeded
        """
        total_steps = 6
        current_step = 0

        print(f"\n{'='*80}")
        print(f"MOONSHINE MODEL CONVERSION PIPELINE")
        print(f"{'='*80}")
        print(f"Input model: {self.model_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target vocab size: {self.target_vocab_size}")
        print(f"{'='*80}\n")

        # Step 1: Extend tokenizer
        current_step += 1
        self.print_step(current_step, total_steps, "Extend Tokenizer Vocabulary")
        if not self.extend_tokenizer():
            print("[FAILED] Tokenizer extension failed!")
            return False

        # Step 2: Resize embeddings
        current_step += 1
        self.print_step(current_step, total_steps, "Resize Model Embeddings")
        if not self.resize_embeddings():
            print("[FAILED] Embedding resize failed!")
            return False

        # Step 3: Export to ONNX
        current_step += 1
        self.print_step(current_step, total_steps, "Export to ONNX")
        if not self.export_to_onnx():
            print("[FAILED] ONNX export failed!")
            return False

        # Step 4: Create merged decoder
        current_step += 1
        self.print_step(current_step, total_steps, "Create Merged Decoder")
        if not self.create_merged_decoder():
            print("[FAILED] Merged decoder creation failed!")
            return False

        # Step 5: Convert tokenizer to binary
        current_step += 1
        self.print_step(current_step, total_steps, "Convert Tokenizer to Binary")
        self.convert_tokenizer_to_bin()  # Non-fatal

        # Step 6: Convert to ORT
        current_step += 1
        self.print_step(current_step, total_steps, "Convert to ORT Format")
        if not self.convert_to_ort():
            print("[WARNING] ORT conversion failed (non-fatal)")

        # Success!
        print(f"\n{'='*80}")
        print("CONVERSION COMPLETE!")
        print(f"{'='*80}")
        print(f"\nDeployment-ready models:")
        print(f"  ONNX (merged decoder): {self.onnx_merged_dir}")
        print(f"  ORT (optimized):       {self.ort_dir}")
        print(f"\nFiles for C++ inference:")
        print(f"  - encoder_model.onnx / encoder_model.ort")
        print(f"  - decoder_model_merged.onnx / decoder_model_merged.ort")
        print(f"  - tokenizer.bin")
        print(f"  - config.json, preprocessor_config.json")
        print(f"\nNext steps:")
        print(f"  1. Test ONNX inference:")
        print(f"     python src/test_moonshine_onnx.py --model {self.onnx_merged_dir}")
        print(f"  2. Deploy to production (use ORT for best performance)")
        print(f"  3. Integrate with Moonshine C++ library")
        print(f"{'='*80}\n")

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete Moonshine model conversion pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with defaults
  python convert_for_deployment.py \\
      --model ./results-moonshine-fr-no-curriculum/final \\
      --output ./deployment/moonshine-fr-v1

  # Skip tokenizer extension (if already 32768 tokens)
  python convert_for_deployment.py \\
      --model ./model --output ./deployment \\
      --skip-tokenizer-extension

  # ARM platform optimization
  python convert_for_deployment.py \\
      --model ./model --output ./deployment \\
      --target-platform arm

  # Skip specific steps
  python convert_for_deployment.py \\
      --model ./model --output ./deployment \\
      --skip-merged-decoder --skip-ort-conversion
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for converted models'
    )
    parser.add_argument(
        '--target-vocab-size',
        type=int,
        default=32768,
        help='Target vocabulary size (default: 32768)'
    )

    # Step control
    parser.add_argument(
        '--skip-tokenizer-extension',
        action='store_true',
        help='Skip tokenizer vocabulary extension'
    )
    parser.add_argument(
        '--skip-embedding-resize',
        action='store_true',
        help='Skip model embedding resizing'
    )
    parser.add_argument(
        '--skip-onnx-export',
        action='store_true',
        help='Skip ONNX export (use existing ONNX files)'
    )
    parser.add_argument(
        '--skip-merged-decoder',
        action='store_true',
        help='Skip merged decoder creation'
    )
    parser.add_argument(
        '--skip-tokenizer-bin',
        action='store_true',
        help='Skip tokenizer binarization'
    )
    parser.add_argument(
        '--skip-ort-conversion',
        action='store_true',
        help='Skip ORT format conversion'
    )

    # ORT options
    parser.add_argument(
        '--target-platform',
        choices=['arm', 'amd64'],
        help='Target platform for ORT optimization'
    )

    args = parser.parse_args()

    converter = ModelConverter(
        model_path=args.model,
        output_dir=Path(args.output),
        target_vocab_size=args.target_vocab_size,
        skip_tokenizer_extension=args.skip_tokenizer_extension,
        skip_embedding_resize=args.skip_embedding_resize,
        skip_onnx_export=args.skip_onnx_export,
        skip_merged_decoder=args.skip_merged_decoder,
        skip_tokenizer_bin=args.skip_tokenizer_bin,
        skip_ort_conversion=args.skip_ort_conversion,
        target_platform=args.target_platform
    )

    success = converter.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
