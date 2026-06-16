#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# ESM-2 Storage-Safe Embedding Extraction Pipeline
#
# What it does:
# 1. Cleans FASTA headers into safe IDs.
# 2. Skips invalid-token protein records only if invalid count is below threshold.
# 3. Splits valid proteins into:
#      - short proteins: length <= truncation length, extracted directly with --include mean
#      - long proteins: length > truncation length, processed one protein at a time
# 4. For each long protein:
#      - creates temporary chunks
#      - runs extract.py with --include per_tok
#      - aggregates per-token chunk embeddings into one mean protein embedding
#      - deletes temporary chunk FASTA and chunk .pt files immediately
# 5. Final output: one .pt file per original protein in final_mean_embeddings/
###############################################################################

SCRIPT_NAME="$(basename "$0")"

read_default() {
    local prompt="$1"
    local default="$2"
    local var
    read -r -p "$prompt [$default]: " var
    echo "${var:-$default}"
}

read_yes_no_default() {
    local prompt="$1"
    local default="$2"
    local var
    read -r -p "$prompt [$default]: " var
    echo "${var:-$default}"
}

safe_mkdir() {
    mkdir -p "$1"
}

# Defaults based on your working command
DEFAULT_MODEL="esm2_t36_3B_UR50D"
DEFAULT_FASTA="examples/data/some_proteins.fasta"
DEFAULT_OUTPUT_DIR="esm2_embeddings"
DEFAULT_REPR_LAYER="36"
DEFAULT_SHORT_TOKS_PER_BATCH="4096"
DEFAULT_LONG_TOKS_PER_BATCH="4096"
DEFAULT_TRUNCATION_LEN="1022"
DEFAULT_NOGPU="n"
DEFAULT_MAX_INVALID_PERCENT="1.0"
DEFAULT_DELETE_LONG_TEMP="y"
DEFAULT_FORCE_RERUN_SHORT="n"
DEFAULT_OVERWRITE_FINAL="n"

if [ ! -f "scripts/extract.py" ]; then
    echo "ERROR: scripts/extract.py not found."
    echo "Run this script from inside the ESM repo directory."
    echo "Example:"
    echo "  cd /path/to/esm"
    echo "  ./$SCRIPT_NAME"
    exit 1
fi

echo "=========================================="
echo " ESM-2 Storage-Safe Embedding Pipeline"
echo " Header Cleaning + Invalid Filtering"
echo " Short Direct Mean + Long Per-Protein Chunk Aggregation"
echo "=========================================="
echo ""
echo "Press Enter to use the default value shown in brackets."
echo ""

FASTA=$(read_default "Input FASTA path" "$DEFAULT_FASTA")
if [ ! -f "$FASTA" ]; then
    echo "ERROR: FASTA file not found: $FASTA"
    exit 1
fi

MODEL=$(read_default "Model name" "$DEFAULT_MODEL")
OUTPUT_DIR=$(read_default "Main output directory" "$DEFAULT_OUTPUT_DIR")
REPR_LAYER=$(read_default "Representation layer" "$DEFAULT_REPR_LAYER")
SHORT_TOKS_PER_BATCH=$(read_default "Tokens per batch for short proteins" "$DEFAULT_SHORT_TOKS_PER_BATCH")
LONG_TOKS_PER_BATCH=$(read_default "Tokens per batch for long-protein chunks" "$DEFAULT_LONG_TOKS_PER_BATCH")
TRUNCATION_LEN=$(read_default "Max residues per ESM input chunk" "$DEFAULT_TRUNCATION_LEN")
MAX_INVALID_PERCENT=$(read_default "Maximum invalid-token percentage allowed for skipping" "$DEFAULT_MAX_INVALID_PERCENT")
NOGPU=$(read_yes_no_default "Force CPU only? y/n" "$DEFAULT_NOGPU")
DELETE_LONG_TEMP=$(read_yes_no_default "Delete temporary long-protein chunk .pt files immediately? y/n" "$DEFAULT_DELETE_LONG_TEMP")
FORCE_RERUN_SHORT=$(read_yes_no_default "Re-run short protein extraction if short outputs already exist? y/n" "$DEFAULT_FORCE_RERUN_SHORT")
OVERWRITE_FINAL=$(read_yes_no_default "Overwrite existing final long-protein embeddings? y/n" "$DEFAULT_OVERWRITE_FINAL")

METADATA_DIR="$OUTPUT_DIR/metadata"
FINAL_DIR="$OUTPUT_DIR/final_mean_embeddings"
SHORT_FASTA="$METADATA_DIR/short_proteins_len_le_${TRUNCATION_LEN}.fasta"
LONG_SEQS_DIR="$METADATA_DIR/long_protein_fastas"
LONG_IDS_FILE="$METADATA_DIR/long_parent_ids.txt"
MAPPING_CSV="$METADATA_DIR/cleaned_fasta_header_mapping.csv"
INVALID_REPORT_CSV="$METADATA_DIR/invalid_token_proteins_report.csv"
SPLIT_SUMMARY_CSV="$METADATA_DIR/protein_split_summary.csv"
LONG_CHUNK_REPORT_CSV="$METADATA_DIR/long_chunk_aggregation_report.csv"
TEMP_DIR="$OUTPUT_DIR/tmp_long_chunk_processing"

safe_mkdir "$OUTPUT_DIR"
safe_mkdir "$METADATA_DIR"
safe_mkdir "$FINAL_DIR"
safe_mkdir "$LONG_SEQS_DIR"
safe_mkdir "$TEMP_DIR"

echo ""
echo "=========================================="
echo " Configuration"
echo "=========================================="
echo "Input FASTA:                $FASTA"
echo "Model:                      $MODEL"
echo "Main output directory:      $OUTPUT_DIR"
echo "Final embeddings directory: $FINAL_DIR"
echo "Metadata directory:         $METADATA_DIR"
echo "Representation layer:       $REPR_LAYER"
echo "Short toks per batch:       $SHORT_TOKS_PER_BATCH"
echo "Long toks per batch:        $LONG_TOKS_PER_BATCH"
echo "Chunk/truncation length:    $TRUNCATION_LEN"
echo "Invalid skip threshold:     < $MAX_INVALID_PERCENT%"
echo "Force CPU only:             $NOGPU"
echo "Delete long temp outputs:   $DELETE_LONG_TEMP"
echo "=========================================="
echo ""

read -r -p "Continue with preprocessing? y/n [y]: " CONFIRM_PRE
CONFIRM_PRE=${CONFIRM_PRE:-y}
if [[ "$CONFIRM_PRE" != "y" && "$CONFIRM_PRE" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

###############################################################################
# Step 1: Clean headers, skip invalid records if below threshold, split short/long
###############################################################################

echo ""
echo "=========================================="
echo " Step 1: Cleaning FASTA + Splitting Short/Long Proteins"
echo "=========================================="
echo ""

python - <<PY
from pathlib import Path
import csv
import re
import shutil
import sys

input_fasta = Path(r"$FASTA")
short_fasta = Path(r"$SHORT_FASTA")
long_seqs_dir = Path(r"$LONG_SEQS_DIR")
long_ids_file = Path(r"$LONG_IDS_FILE")
mapping_csv = Path(r"$MAPPING_CSV")
invalid_report_csv = Path(r"$INVALID_REPORT_CSV")
split_summary_csv = Path(r"$SPLIT_SUMMARY_CSV")
max_invalid_percent = float("$MAX_INVALID_PERCENT")
truncation_len = int("$TRUNCATION_LEN")

VALID_ESM_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")

# Clean old split artifacts so reruns do not mix records from older FASTAs.
short_fasta.parent.mkdir(parents=True, exist_ok=True)
long_seqs_dir.mkdir(parents=True, exist_ok=True)
for old in long_seqs_dir.glob("*.fasta"):
    old.unlink()


def sanitize(text):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "NA"


def parse_header(header):
    original_header = header.strip()
    parts_by_space = original_header.split()
    pipe_part = parts_by_space[0] if parts_by_space else original_header
    status_part = "_".join(parts_by_space[1:]) if len(parts_by_space) > 1 else ""
    pipe_fields = pipe_part.split("|")

    protein_id = pipe_fields[0] if len(pipe_fields) > 0 else "NA"
    transcript_id = pipe_fields[1] if len(pipe_fields) > 1 else "NA"
    gene_id = pipe_fields[2] if len(pipe_fields) > 2 else "NA"
    gene_symbol = pipe_fields[-2] if len(pipe_fields) >= 2 else "NA"
    last_field = pipe_fields[-1] if len(pipe_fields) >= 1 else "NA"

    length_match = re.search(r"(\d+)", last_field)
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


def read_fasta(path):
    records = []
    current_header = None
    current_seq = []
    with path.open("r") as f:
        for line in f:
            line = line.rstrip("\n")
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
    return records


def normalize_sequence(seq):
    return re.sub(r"\s+", "", seq).upper()


def get_invalid_chars(seq):
    return sorted(set(seq) - VALID_ESM_AA)


def write_fasta_record(handle, header, seq, width=80):
    handle.write(f">{header}\n")
    for i in range(0, len(seq), width):
        handle.write(seq[i:i+width] + "\n")

records = read_fasta(input_fasta)
if not records:
    raise SystemExit("ERROR: No FASTA records found.")

invalid_records = []
for idx, (header, seq) in enumerate(records, start=1):
    seq_clean = normalize_sequence(seq)
    invalid_chars = get_invalid_chars(seq_clean)
    if invalid_chars:
        invalid_records.append({
            "index": idx,
            "header": header,
            "sequence": seq_clean,
            "invalid_chars": "".join(invalid_chars),
            "sequence_length": len(seq_clean),
        })

total_records = len(records)
invalid_count = len(invalid_records)
invalid_percent = (invalid_count / total_records * 100) if total_records else 0.0

print(f"Total protein sequences: {total_records}")
print(f"Sequences with invalid ESM tokens: {invalid_count}")
print(f"Invalid-token percentage: {invalid_percent:.4f}%")
print(f"Maximum allowed invalid-token percentage for skipping: {max_invalid_percent:.4f}%")

invalid_report_csv.parent.mkdir(parents=True, exist_ok=True)
with invalid_report_csv.open("w", newline="") as csvfile:
    fieldnames = ["index", "invalid_chars", "sequence_length", "original_header", "sequence_preview"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for item in invalid_records:
        writer.writerow({
            "index": item["index"],
            "invalid_chars": item["invalid_chars"],
            "sequence_length": item["sequence_length"],
            "original_header": item["header"],
            "sequence_preview": item["sequence"][:120],
        })
print(f"Invalid token report saved to: {invalid_report_csv}")

if invalid_percent >= max_invalid_percent and invalid_count > 0:
    print("\nERROR: Invalid-token sequence percentage is too high.")
    print("The script will not remove them automatically.")
    print(f"Invalid records: {invalid_count}/{total_records} ({invalid_percent:.4f}%)")
    print(f"Allowed threshold: < {max_invalid_percent:.4f}%")
    print("Inspect the invalid token report before deciding whether to skip, truncate, or replace.")
    sys.exit(1)

invalid_indices = set(item["index"] for item in invalid_records)
used_ids = set()
mapping_rows = []
split_rows = []
long_ids = []
written_short = 0
written_long = 0
skipped_count = 0

with short_fasta.open("w") as short_out:
    for idx, (header, seq) in enumerate(records, start=1):
        info = parse_header(header)
        seq_clean = normalize_sequence(seq)
        invalid_chars = get_invalid_chars(seq_clean)

        base_clean_id = (
            f"seq_{idx:06d}_"
            f"{info['protein_id']}_"
            f"{info['gene_symbol']}_"
            f"{info['protein_length']}_"
            f"{info['status']}"
        )
        clean_id = sanitize(base_clean_id)
        original_clean_id = clean_id
        counter = 2
        while clean_id in used_ids:
            clean_id = f"{original_clean_id}_{counter}"
            counter += 1
        used_ids.add(clean_id)

        if idx in invalid_indices:
            action = "skipped_invalid_esm_token"
            route = "skipped_invalid"
            skipped_count += 1
        elif len(seq_clean) <= truncation_len:
            action = "written_short_direct"
            route = "short_direct_mean"
            written_short += 1
            write_fasta_record(short_out, clean_id, seq_clean)
        else:
            action = "written_long_parent_fasta"
            route = "long_chunk_per_tok_then_mean"
            written_long += 1
            long_ids.append(clean_id)
            parent_fasta = long_seqs_dir / f"{clean_id}.fasta"
            with parent_fasta.open("w") as parent_out:
                write_fasta_record(parent_out, clean_id, seq_clean)

        mapping_rows.append({
            "clean_id": clean_id,
            "protein_id": info["protein_id"],
            "transcript_id": info["transcript_id"],
            "gene_id": info["gene_id"],
            "gene_symbol": info["gene_symbol"],
            "protein_length_from_header": info["protein_length"],
            "actual_sequence_length": len(seq_clean),
            "status": info["status"],
            "invalid_chars": "".join(invalid_chars),
            "action": action,
            "original_header": info["original_header"],
        })
        split_rows.append({
            "clean_id": clean_id,
            "actual_sequence_length": len(seq_clean),
            "route": route,
            "invalid_chars": "".join(invalid_chars),
            "source_fasta_file": str((long_seqs_dir / f"{clean_id}.fasta") if route.startswith("long") else short_fasta),
        })

with mapping_csv.open("w", newline="") as csvfile:
    fieldnames = [
        "clean_id", "protein_id", "transcript_id", "gene_id", "gene_symbol",
        "protein_length_from_header", "actual_sequence_length", "status",
        "invalid_chars", "action", "original_header"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mapping_rows)

with split_summary_csv.open("w", newline="") as csvfile:
    fieldnames = ["clean_id", "actual_sequence_length", "route", "invalid_chars", "source_fasta_file"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(split_rows)

with long_ids_file.open("w") as f:
    for clean_id in long_ids:
        f.write(clean_id + "\n")

print("\nPreprocessing complete.")
print(f"Short FASTA saved to: {short_fasta}")
print(f"Long parent FASTA directory saved to: {long_seqs_dir}")
print(f"Long IDs file saved to: {long_ids_file}")
print(f"Header mapping CSV saved to: {mapping_csv}")
print(f"Split summary CSV saved to: {split_summary_csv}")
print(f"Input sequences: {total_records}")
print(f"Short direct proteins: {written_short}")
print(f"Long proteins requiring chunking: {written_long}")
print(f"Skipped invalid-token proteins: {skipped_count}")
PY

###############################################################################
# Step 2: Run ESM for short proteins directly with --include mean
###############################################################################

echo ""
echo "=========================================="
echo " Step 2: Short Proteins Direct Mean Extraction"
echo "=========================================="
echo ""

SHORT_COUNT=$(grep -c '^>' "$SHORT_FASTA" || true)
echo "Short proteins to process: $SHORT_COUNT"

if [ "$SHORT_COUNT" -gt 0 ]; then
    if [ "$FORCE_RERUN_SHORT" = "y" ] || [ "$FORCE_RERUN_SHORT" = "Y" ]; then
        echo "Force rerun requested. Existing short/direct final outputs may be overwritten by extract.py."
        RUN_SHORT="y"
    else
        # If no .pt exists in final dir, run; otherwise ask.
        EXISTING_FINAL_COUNT=$(find "$FINAL_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')
        if [ "$EXISTING_FINAL_COUNT" -eq 0 ]; then
            RUN_SHORT="y"
        else
            read -r -p "Final directory already has $EXISTING_FINAL_COUNT .pt files. Run short extraction anyway? y/n [n]: " RUN_SHORT
            RUN_SHORT=${RUN_SHORT:-n}
        fi
    fi

    if [[ "$RUN_SHORT" == "y" || "$RUN_SHORT" == "Y" ]]; then
        SHORT_CMD=(
            python scripts/extract.py
            "$MODEL"
            "$SHORT_FASTA"
            "$FINAL_DIR"
            --repr_layers "$REPR_LAYER"
            --include mean
            --toks_per_batch "$SHORT_TOKS_PER_BATCH"
            --truncation_seq_length "$TRUNCATION_LEN"
        )
        if [[ "$NOGPU" == "y" || "$NOGPU" == "Y" ]]; then
            SHORT_CMD+=(--nogpu)
        fi
        echo "Running short protein extraction:"
        echo "${SHORT_CMD[@]}"
        "${SHORT_CMD[@]}"
    else
        echo "Skipping short protein extraction."
    fi
else
    echo "No short proteins found."
fi

###############################################################################
# Step 3: Long proteins, one parent at a time
###############################################################################

echo ""
echo "=========================================="
echo " Step 3: Long Proteins Per-Protein Chunk Processing"
echo "=========================================="
echo ""

LONG_COUNT=$(wc -l < "$LONG_IDS_FILE" | tr -d ' ')
echo "Long proteins to process: $LONG_COUNT"

# Initialize long aggregation report
python - <<PY
from pathlib import Path
import csv
report = Path(r"$LONG_CHUNK_REPORT_CSV")
report.parent.mkdir(parents=True, exist_ok=True)
if not report.exists():
    with report.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "parent_id", "parent_length", "num_chunks", "total_residue_embeddings",
            "embedding_dim", "layer_key", "final_pt_file", "status"
        ])
        writer.writeheader()
PY

if [ "$LONG_COUNT" -gt 0 ]; then
    IDX=0
    while IFS= read -r PARENT_ID || [ -n "$PARENT_ID" ]; do
        IDX=$((IDX + 1))
        PARENT_FASTA="$LONG_SEQS_DIR/${PARENT_ID}.fasta"
        FINAL_PT="$FINAL_DIR/${PARENT_ID}.pt"

        echo ""
        echo "------------------------------------------"
        echo "Long protein $IDX / $LONG_COUNT"
        echo "Parent ID: $PARENT_ID"
        echo "------------------------------------------"

        if [ -f "$FINAL_PT" ] && [[ "$OVERWRITE_FINAL" != "y" && "$OVERWRITE_FINAL" != "Y" ]]; then
            echo "Final embedding already exists and overwrite is disabled. Skipping: $FINAL_PT"
            continue
        fi

        if [ ! -f "$PARENT_FASTA" ]; then
            echo "WARNING: Parent FASTA missing. Skipping: $PARENT_FASTA"
            continue
        fi

        PARENT_TEMP_DIR="$TEMP_DIR/$PARENT_ID"
        CHUNK_FASTA="$PARENT_TEMP_DIR/${PARENT_ID}.chunks.fasta"
        CHUNK_META="$PARENT_TEMP_DIR/${PARENT_ID}.chunks.csv"
        CHUNK_PT_DIR="$PARENT_TEMP_DIR/chunk_pt"
        rm -rf "$PARENT_TEMP_DIR"
        mkdir -p "$CHUNK_PT_DIR"

        # Create chunks for this one long parent only.
        python - <<PY
from pathlib import Path
import csv

parent_fasta = Path(r"$PARENT_FASTA")
chunk_fasta = Path(r"$CHUNK_FASTA")
chunk_meta = Path(r"$CHUNK_META")
chunk_size = int("$TRUNCATION_LEN")
parent_id = "$PARENT_ID"

header = None
seq_parts = []
with parent_fasta.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:]
        else:
            seq_parts.append(line)
seq = "".join(seq_parts)
parent_len = len(seq)
if parent_len <= chunk_size:
    raise SystemExit(f"ERROR: parent sequence is not long enough for chunking: {parent_id} len={parent_len}")

chunk_fasta.parent.mkdir(parents=True, exist_ok=True)
rows = []
with chunk_fasta.open("w") as out:
    chunk_index = 0
    for start0 in range(0, parent_len, chunk_size):
        end0 = min(start0 + chunk_size, parent_len)
        chunk_seq = seq[start0:end0]
        chunk_index += 1
        start1 = start0 + 1
        end1 = end0
        chunk_id = f"{parent_id}__chunk{chunk_index:04d}__start{start1}__end{end1}"
        out.write(f">{chunk_id}\n")
        for i in range(0, len(chunk_seq), 80):
            out.write(chunk_seq[i:i+80] + "\n")
        rows.append({
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "chunk_index": chunk_index,
            "start": start1,
            "end": end1,
            "chunk_length": len(chunk_seq),
            "parent_length": parent_len,
        })

with chunk_meta.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["chunk_id", "parent_id", "chunk_index", "start", "end", "chunk_length", "parent_length"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {len(rows)} chunks for {parent_id} length={parent_len}")
PY

        LONG_CMD=(
            python scripts/extract.py
            "$MODEL"
            "$CHUNK_FASTA"
            "$CHUNK_PT_DIR"
            --repr_layers "$REPR_LAYER"
            --include per_tok
            --toks_per_batch "$LONG_TOKS_PER_BATCH"
            --truncation_seq_length "$TRUNCATION_LEN"
        )
        if [[ "$NOGPU" == "y" || "$NOGPU" == "Y" ]]; then
            LONG_CMD+=(--nogpu)
        fi

        echo "Running ESM per-token extraction for chunks of this protein..."
        echo "${LONG_CMD[@]}"
        "${LONG_CMD[@]}"

        # Aggregate chunk per-token embeddings to one mean protein embedding, without saving concatenated matrix.
        python - <<PY
from pathlib import Path
import csv
import torch

chunk_meta = Path(r"$CHUNK_META")
chunk_pt_dir = Path(r"$CHUNK_PT_DIR")
final_pt = Path(r"$FINAL_PT")
report_csv = Path(r"$LONG_CHUNK_REPORT_CSV")
parent_id = "$PARENT_ID"
requested_layer = "$REPR_LAYER"
model_name = "$MODEL"

rows = []
with chunk_meta.open() as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise SystemExit(f"ERROR: no chunk metadata rows for {parent_id}")

running_sum = None
total_residues = 0
layer_key_used = None
embedding_dim = None

for row in rows:
    chunk_id = row["chunk_id"]
    pt_file = chunk_pt_dir / f"{chunk_id}.pt"
    if not pt_file.exists():
        raise SystemExit(f"ERROR: missing chunk pt file: {pt_file}")

    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    reps_dict = data.get("representations")
    if not reps_dict:
        raise SystemExit(f"ERROR: no per-token representations found in {pt_file}")

    # extract.py resolves -1 to the real final layer key, so use requested key if present,
    # otherwise use the only/last available layer key.
    candidate_keys = list(reps_dict.keys())
    try:
        req_int = int(requested_layer)
    except ValueError:
        req_int = None

    if req_int in reps_dict:
        layer_key = req_int
    elif len(candidate_keys) == 1:
        layer_key = candidate_keys[0]
    else:
        layer_key = sorted(candidate_keys)[-1]

    rep = reps_dict[layer_key].float()  # shape: chunk_len x dim
    if rep.ndim != 2:
        raise SystemExit(f"ERROR: unexpected representation shape in {pt_file}: {tuple(rep.shape)}")

    if running_sum is None:
        running_sum = rep.sum(dim=0)
        embedding_dim = rep.shape[1]
        layer_key_used = layer_key
    else:
        running_sum += rep.sum(dim=0)

    total_residues += rep.shape[0]

if total_residues == 0:
    raise SystemExit(f"ERROR: zero residue embeddings for {parent_id}")

final_embedding = running_sum / total_residues
parent_length = int(rows[0]["parent_length"])

final_pt.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "label": parent_id,
    "mean_representations": {layer_key_used: final_embedding.clone()},
    "source": "chunked_per_tok_mean_storage_safe",
    "model": model_name,
    "original_length": parent_length,
    "num_chunks": len(rows),
    "total_residue_embeddings": total_residues,
    "chunking": {
        "chunk_size": int("$TRUNCATION_LEN"),
        "method": "non_overlapping_chunks_then_per_token_sum_mean",
    },
}, final_pt)

with report_csv.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "parent_id", "parent_length", "num_chunks", "total_residue_embeddings",
        "embedding_dim", "layer_key", "final_pt_file", "status"
    ])
    writer.writerow({
        "parent_id": parent_id,
        "parent_length": parent_length,
        "num_chunks": len(rows),
        "total_residue_embeddings": total_residues,
        "embedding_dim": embedding_dim,
        "layer_key": layer_key_used,
        "final_pt_file": str(final_pt),
        "status": "written",
    })

print(f"Saved final long-protein mean embedding: {final_pt}")
print(f"Final shape: {tuple(final_embedding.shape)} | residues aggregated: {total_residues} | chunks: {len(rows)}")
PY

        if [[ "$DELETE_LONG_TEMP" == "y" || "$DELETE_LONG_TEMP" == "Y" ]]; then
            rm -rf "$PARENT_TEMP_DIR"
            echo "Deleted temporary chunk files for: $PARENT_ID"
        else
            echo "Kept temporary files at: $PARENT_TEMP_DIR"
        fi

    done < "$LONG_IDS_FILE"
else
    echo "No long proteins found."
fi

###############################################################################
# Step 4: Final summary
###############################################################################

echo ""
echo "=========================================="
echo " Done"
echo "=========================================="
echo "Final protein-level embeddings saved to:"
echo "  $FINAL_DIR"
echo ""
echo "Metadata saved to:"
echo "  $METADATA_DIR"
echo ""
echo "Important files:"
echo "  Header mapping:        $MAPPING_CSV"
echo "  Invalid-token report:  $INVALID_REPORT_CSV"
echo "  Split summary:         $SPLIT_SUMMARY_CSV"
echo "  Long aggregation log:  $LONG_CHUNK_REPORT_CSV"
echo ""
echo "Final output format for all proteins:"
echo "  data['mean_representations'][layer]"
echo ""
echo "Note: long proteins were embedded using temporary per-token chunk outputs,"
echo "then aggregated to one mean vector and deleted protein-by-protein."
