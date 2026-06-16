#!/usr/bin/env bash

set -e

echo "=========================================="
echo " ESM-2 Embedding Extraction Script"
echo " With FASTA Header Cleaning"
echo "=========================================="
echo ""

# Defaults based on your working command
DEFAULT_MODEL="esm2_t36_3B_UR50D"
DEFAULT_FASTA="examples/data/some_proteins.fasta"
DEFAULT_OUTPUT_DIR="esm2_embeddings"
DEFAULT_REPR_LAYER="36"
DEFAULT_INCLUDE="mean"
DEFAULT_TOKS_PER_BATCH="4096"
DEFAULT_TRUNCATION_LEN="1022"
DEFAULT_NOGPU="n"
DEFAULT_CLEAN_FASTA="cleaned_proteins_for_esm.fasta"
DEFAULT_MAPPING_CSV="cleaned_fasta_header_mapping.csv"

# Check that extract.py exists
if [ ! -f "scripts/extract.py" ]; then
    echo "ERROR: scripts/extract.py not found."
    echo "Please run this script from inside the ESM repo directory."
    echo "Example:"
    echo "  cd /path/to/esm"
    echo "  ./run_esm_extract_interactive.sh"
    exit 1
fi

echo "Press Enter to use the default value shown in brackets."
echo ""

read -p "Input FASTA path [$DEFAULT_FASTA]: " FASTA
FASTA=${FASTA:-$DEFAULT_FASTA}

if [ ! -f "$FASTA" ]; then
    echo "ERROR: FASTA file not found:"
    echo "  $FASTA"
    exit 1
fi

read -p "Cleaned FASTA output path [$DEFAULT_CLEAN_FASTA]: " CLEAN_FASTA
CLEAN_FASTA=${CLEAN_FASTA:-$DEFAULT_CLEAN_FASTA}

read -p "Header mapping CSV path [$DEFAULT_MAPPING_CSV]: " MAPPING_CSV
MAPPING_CSV=${MAPPING_CSV:-$DEFAULT_MAPPING_CSV}

read -p "Model name [$DEFAULT_MODEL]: " MODEL
MODEL=${MODEL:-$DEFAULT_MODEL}

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
echo " Step 1: Cleaning FASTA Headers"
echo "=========================================="
echo ""

python - <<PY
from pathlib import Path
import csv
import re

input_fasta = Path("$FASTA")
clean_fasta = Path("$CLEAN_FASTA")
mapping_csv = Path("$MAPPING_CSV")

def sanitize(text):
    """
    Make text safe for FASTA IDs and output filenames.
    """
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def parse_header(header):
    """
    Expected example:
    ENSP00000040877.1|ENST00000040877.2|ENSG00000059588.11|...|TARBP1-201|TARBP1|1621__1 changed

    Clean output target:
    seq_000001_ENSP00000040877.1_TARBP1_1621_changed
    """

    original_header = header.strip()

    # Split by whitespace first, because your header has a tab/space before 'changed'
    parts_by_space = original_header.split()
    pipe_part = parts_by_space[0]
    status_part = "_".join(parts_by_space[1:]) if len(parts_by_space) > 1 else ""

    pipe_fields = pipe_part.split("|")

    protein_id = pipe_fields[0] if len(pipe_fields) > 0 else "NA"
    transcript_id = pipe_fields[1] if len(pipe_fields) > 1 else "NA"
    gene_id = pipe_fields[2] if len(pipe_fields) > 2 else "NA"

    # In your PrecisionProDB-style header, gene symbol appears second-last,
    # and length/status info appears last.
    gene_symbol = pipe_fields[-2] if len(pipe_fields) >= 2 else "NA"
    last_field = pipe_fields[-1] if len(pipe_fields) >= 1 else "NA"

    # Example last_field: 1621__1
    length_match = re.search(r"(\\d+)", last_field)
    protein_length = length_match.group(1) if length_match else "NA"

    status = status_part if status_part else "protein"

    return {
        "original_header": original_header,
        "protein_id": sanitize(protein_id),
        "transcript_id": sanitize(transcript_id),
        "gene_id": sanitize(gene_id),
        "gene_symbol": sanitize(gene_symbol),
        "protein_length": sanitize(protein_length),
        "status": sanitize(status),
    }

records = []
current_header = None
current_seq = []

with input_fasta.open("r") as f:
    for line in f:
        line = line.rstrip("\\n")
        if not line:
            continue

        if line.startswith(">"):
            if current_header is not None:
                records.append((current_header, "".join(current_seq)))
            current_header = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line.strip())

    if current_header is not None:
        records.append((current_header, "".join(current_seq)))

if not records:
    raise SystemExit("ERROR: No FASTA records found.")

clean_fasta.parent.mkdir(parents=True, exist_ok=True)
mapping_csv.parent.mkdir(parents=True, exist_ok=True)

used_ids = set()
mapping_rows = []

with clean_fasta.open("w") as fasta_out:
    for idx, (header, seq) in enumerate(records, start=1):
        info = parse_header(header)

        base_clean_id = (
            f"seq_{idx:06d}_"
            f"{info['protein_id']}_"
            f"{info['gene_symbol']}_"
            f"{info['protein_length']}_"
            f"{info['status']}"
        )

        clean_id = sanitize(base_clean_id)

        # Ensure uniqueness
        original_clean_id = clean_id
        counter = 2
        while clean_id in used_ids:
            clean_id = f"{original_clean_id}_{counter}"
            counter += 1
        used_ids.add(clean_id)

        # Write cleaned FASTA
        fasta_out.write(f">{clean_id}\\n")

        # Wrap sequence at 80 chars
        seq = re.sub(r"\\s+", "", seq).upper()
        for i in range(0, len(seq), 80):
            fasta_out.write(seq[i:i+80] + "\\n")

        mapping_rows.append({
            "clean_id": clean_id,
            "protein_id": info["protein_id"],
            "transcript_id": info["transcript_id"],
            "gene_id": info["gene_id"],
            "gene_symbol": info["gene_symbol"],
            "protein_length_from_header": info["protein_length"],
            "actual_sequence_length": len(seq),
            "status": info["status"],
            "original_header": info["original_header"],
        })

with mapping_csv.open("w", newline="") as csvfile:
    fieldnames = [
        "clean_id",
        "protein_id",
        "transcript_id",
        "gene_id",
        "gene_symbol",
        "protein_length_from_header",
        "actual_sequence_length",
        "status",
        "original_header",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mapping_rows)

print(f"Cleaned FASTA saved to: {clean_fasta}")
print(f"Header mapping CSV saved to: {mapping_csv}")
print(f"Number of sequences processed: {len(records)}")

long_count = sum(1 for row in mapping_rows if row["actual_sequence_length"] > int("$TRUNCATION_LEN"))
if long_count > 0:
    print("")
    print(f"WARNING: {long_count} sequences are longer than truncation length $TRUNCATION_LEN.")
    print("These will be truncated by extract.py unless you change the truncation setting.")
PY

echo ""
echo "=========================================="
echo " Step 2: ESM Extraction Configuration"
echo "=========================================="
echo "Model:                  $MODEL"
echo "Original FASTA:         $FASTA"
echo "Cleaned FASTA:          $CLEAN_FASTA"
echo "Mapping CSV:            $MAPPING_CSV"
echo "Output directory:       $OUTPUT_DIR"
echo "Representation layer:   $REPR_LAYER"
echo "Include:                $INCLUDE"
echo "Tokens per batch:       $TOKS_PER_BATCH"
echo "Truncation length:      $TRUNCATION_LEN"
echo "Force CPU only:         $NOGPU"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT_DIR"

CMD=(
    python scripts/extract.py
    "$MODEL"
    "$CLEAN_FASTA"
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
    echo "Cancelled before running ESM extraction."
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
echo "Cleaned FASTA used:"
echo "  $CLEAN_FASTA"
echo ""
echo "Header mapping saved to:"
echo "  $MAPPING_CSV"
echo ""
echo "Each .pt file name should now correspond to a clean FASTA ID."