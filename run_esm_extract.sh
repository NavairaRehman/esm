#!/usr/bin/env bash

set -e

echo "=========================================="
echo " ESM-2 Embedding Extraction Script"
echo "=========================================="
echo ""

# Default values based on your working command
DEFAULT_MODEL="esm2_t36_3B_UR50D"
DEFAULT_FASTA="examples/data/some_proteins.fasta"
DEFAULT_OUTPUT_DIR="esm2_embeddings"
DEFAULT_REPR_LAYER="36"
DEFAULT_INCLUDE="mean"
DEFAULT_TOKS_PER_BATCH="4096"
DEFAULT_TRUNCATION_LEN="1022"
DEFAULT_NOGPU="n"

# Check that extract.py exists
if [ ! -f "scripts/extract.py" ]; then
    echo "ERROR: scripts/extract.py not found."
    echo "Please run this script from inside the ESM repo directory."
    echo "Example:"
    echo "  cd /path/to/esm"
    echo "  ./run_esm_extract.sh"
    exit 1
fi

echo "Press Enter to use the default value shown in brackets."
echo ""

read -p "Model name [$DEFAULT_MODEL]: " MODEL
MODEL=${MODEL:-$DEFAULT_MODEL}

read -p "Input FASTA path [$DEFAULT_FASTA]: " FASTA
FASTA=${FASTA:-$DEFAULT_FASTA}

read -p "Output directory [$DEFAULT_OUTPUT_DIR]: " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}

read -p "Representation layer [$DEFAULT_REPR_LAYER]: " REPR_LAYER
REPR_LAYER=${REPR_LAYER:-$DEFAULT_REPR_LAYER}

echo ""
echo "Include options:"
echo "  mean      = one embedding vector per protein"
echo "  per_tok   = one embedding vector per amino acid/residue"
echo "  bos       = beginning-of-sequence token embedding"
echo "  contacts  = predicted contact map"
echo ""
echo "Examples:"
echo "  mean"
echo "  mean per_tok"
echo "  mean per_tok contacts"
echo ""

read -p "Include options [$DEFAULT_INCLUDE]: " INCLUDE
INCLUDE=${INCLUDE:-$DEFAULT_INCLUDE}

read -p "Tokens per batch [$DEFAULT_TOKS_PER_BATCH]: " TOKS_PER_BATCH
TOKS_PER_BATCH=${TOKS_PER_BATCH:-$DEFAULT_TOKS_PER_BATCH}

read -p "Truncation sequence length [$DEFAULT_TRUNCATION_LEN]: " TRUNCATION_LEN
TRUNCATION_LEN=${TRUNCATION_LEN:-$DEFAULT_TRUNCATION_LEN}

read -p "Force CPU only? y/n [$DEFAULT_NOGPU]: " NOGPU
NOGPU=${NOGPU:-$DEFAULT_NOGPU}

echo ""
echo "=========================================="
echo " Configuration"
echo "=========================================="
echo "Model:                  $MODEL"
echo "Input FASTA:            $FASTA"
echo "Output directory:       $OUTPUT_DIR"
echo "Representation layer:   $REPR_LAYER"
echo "Include:                $INCLUDE"
echo "Tokens per batch:       $TOKS_PER_BATCH"
echo "Truncation length:      $TRUNCATION_LEN"
echo "Force CPU only:         $NOGPU"
echo "=========================================="
echo ""

# Validate FASTA file
if [ ! -f "$FASTA" ]; then
    echo "ERROR: FASTA file not found:"
    echo "  $FASTA"
    exit 1
fi

# Create output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Build command
CMD=(
    python scripts/extract.py
    "$MODEL"
    "$FASTA"
    "$OUTPUT_DIR"
    --repr_layers "$REPR_LAYER"
    --include $INCLUDE
    --toks_per_batch "$TOKS_PER_BATCH"
    --truncation_seq_length "$TRUNCATION_LEN"
)

if [[ "$NOGPU" == "y" || "$NOGPU" == "Y" ]]; then
    CMD+=(--nogpu)
fi

echo "Command to run:"
echo "${CMD[@]}"
echo ""

read -p "Run this command? y/n [y]: " CONFIRM
CONFIRM=${CONFIRM:-y}

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting ESM embedding extraction..."
echo ""

"${CMD[@]}"

echo ""
echo "=========================================="
echo " Done"
echo "=========================================="
echo "Embeddings saved to:"
echo "  $OUTPUT_DIR"
echo ""
echo "Each protein should have a .pt file containing the requested outputs."