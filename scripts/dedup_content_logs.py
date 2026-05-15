#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "loghub-2.0" / "full_dataset"
EXACT_DEDUP_ROOT = ROOT / "deduplicated_with_dup_count"
NORMALIZED_DEDUP_ROOT = ROOT / "normalized_deduplicated"
CLUSTERED_ROOT = ROOT / "clustered"
TEMP_ROOT = ROOT / ".preprocess_tmp"
EXACT_DEDUP_SUMMARY_PATH = ROOT / "dedup_with_dup_count_summary.csv"
NORMALIZED_DEDUP_SUMMARY_PATH = ROOT / "normalized_dedup_summary.csv"
CLUSTER_DATASET_SUMMARY_PATH = ROOT / "cluster_dataset_summary.csv"
CLUSTER_SUMMARY_PATH = ROOT / "cluster_summary.csv"
SELECTED_BALANCED_ROOT = ROOT / "selected_balanced"
SELECTED_BALANCED_SUMMARY_PATH = ROOT / "selected_balanced_summary.csv"
PROGRESS_EVERY = 100_000


IPV4_RE = re.compile(r"(?<![A-Za-z0-9_])(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?![A-Za-z0-9_])")
IPV6_RE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Fa-f0-9]{1,4}:){2,}[A-Fa-f0-9]{1,4}(?::\d+)?(?![A-Za-z0-9_])")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
URL_RE = re.compile(r"\b(?:https?|ftp)://\S+")
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T\s]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)?\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b")
HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{12,}\b")
COMMON_ID_RE = re.compile(
    r"\b(?:blk|block|job|task|attempt|app|application|container|executor|stage|rdd|core|pid|tid|node|host|port)[-_:.]?[A-Za-z0-9_.:-]*\d[A-Za-z0-9_.:-]*\b",
    re.IGNORECASE,
)
PATH_RE = re.compile(r"(?<!\w)(?:/[A-Za-z0-9._+\-=:@%]+){2,}(?:/[A-Za-z0-9._+\-=:@%]*)?")
MIXED_ID_RE = re.compile(r"\b(?=[A-Za-z0-9_.:-]*\d)(?=[A-Za-z0-9_.:-]*[A-Za-z])[A-Za-z0-9_.:-]{8,}\b")
FLOAT_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])")
INT_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?![A-Za-z0-9_])")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"<[a-z_]+>|[A-Za-z_]+|[0-9]+|[^\sA-Za-z_0-9]")
PLACEHOLDER_RE = re.compile(r"^<[a-z_]+>$")


@dataclass
class Cluster:
    cluster_id: str
    token_count: int
    tokens: list[str]
    template_tokens: list[str]
    size_rows: int = 0
    size_unique: int = 0
    weight_dup_sum: int = 0
    representative_content: str = ""
    representative_normalized_content: str = ""
    representative_dup_count: int = -1
    representative_line_id: int = sys.maxsize
    member_row_ids: list[int] = field(default_factory=list)


def set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def configure_sqlite(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA cache_size=-200000")
    connection.execute("PRAGMA mmap_size=268435456")


def find_single_file(dataset_dir: Path, suffix: str) -> Path:
    matches = sorted(dataset_dir.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {suffix} in {dataset_dir}, found {len(matches)}")
    return matches[0]


def find_input_file(input_root: Path, dataset_name: str) -> Path:
    dataset_dir = input_root / dataset_name
    return find_single_file(dataset_dir, "_structured.csv")


def selected_dataset_dirs(dataset_names: list[str] | None) -> list[Path]:
    dataset_dirs = sorted(path for path in DATASET_ROOT.iterdir() if path.is_dir())
    if len(dataset_dirs) != 14:
        raise RuntimeError(f"Expected 14 datasets in {DATASET_ROOT}, found {len(dataset_dirs)}")

    if dataset_names:
        requested = set(dataset_names)
        dataset_dirs = [path for path in dataset_dirs if path.name in requested]
        missing = sorted(requested - {path.name for path in dataset_dirs})
        if missing:
            raise RuntimeError(f"Requested dataset(s) not found: {', '.join(missing)}")
    return dataset_dirs


def line_id_value(row: dict[str, str]) -> int:
    try:
        return int(row.get("LineId", sys.maxsize))
    except ValueError:
        return sys.maxsize


def should_replace_path(match: re.Match[str]) -> str:
    value = match.group(0)
    if re.search(r"\d", value):
        return " <PATH> "
    if len(value) >= 24:
        return " <PATH> "
    lowered = value.lower()
    variable_prefixes = (
        "/tmp/",
        "/var/tmp/",
        "/mnt/",
        "/data/",
        "/home/",
        "/user/",
        "/users/",
        "/export/",
    )
    if lowered.startswith(variable_prefixes):
        return " <PATH> "
    return value


def normalize_content(content: str) -> str:
    normalized = content.strip()
    replacements = [
        (URL_RE, " <URL> "),
        (EMAIL_RE, " <EMAIL> "),
        (IPV4_RE, " <IP> "),
        (IPV6_RE, " <IP> "),
        (UUID_RE, " <UUID> "),
        (DATE_RE, " <DATE> "),
        (TIME_RE, " <TIME> "),
        (HEX_RE, " <HEX> "),
        (COMMON_ID_RE, " <ID> "),
    ]
    for pattern, replacement in replacements:
        normalized = pattern.sub(replacement, normalized)

    normalized = PATH_RE.sub(should_replace_path, normalized)
    normalized = MIXED_ID_RE.sub(" <ID> ", normalized)
    normalized = FLOAT_RE.sub(" <NUM> ", normalized)
    normalized = INT_RE.sub(" <NUM> ", normalized)
    normalized = SPACE_RE.sub(" ", normalized).strip()
    return normalized.lower()


def tokenize_normalized(normalized_content: str) -> list[str]:
    return TOKEN_RE.findall(normalized_content)


def stable_tokens(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if not PLACEHOLDER_RE.match(token)]


def bucket_key(tokens: list[str]) -> tuple[int, str, str]:
    stable = stable_tokens(tokens)
    first = stable[0] if stable else ""
    second = stable[1] if len(stable) > 1 else ""
    last = stable[-1] if stable else ""
    return (len(tokens), first + "|" + second, last)


def token_similarity(tokens: list[str], template_tokens: list[str]) -> float:
    if len(tokens) != len(template_tokens):
        return 0.0
    if not tokens:
        return 1.0

    matches = 0
    comparable = 0
    for token, template_token in zip(tokens, template_tokens):
        if template_token == "<*>":
            continue
        comparable += 1
        if token == template_token or PLACEHOLDER_RE.match(token) or PLACEHOLDER_RE.match(template_token):
            matches += 1
    if comparable == 0:
        return 1.0
    return matches / comparable


def update_template_tokens(template_tokens: list[str], tokens: list[str]) -> None:
    for index, token in enumerate(tokens):
        if template_tokens[index] == token:
            continue
        if PLACEHOLDER_RE.match(template_tokens[index]) and PLACEHOLDER_RE.match(token):
            continue
        template_tokens[index] = "<*>"


def template_guess_from_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?\]\)\}])", r"\1", text)
    text = re.sub(r"([\[\(\{])\s+", r"\1", text)
    text = re.sub(r"<\s+([a-z_]+)\s+>", r"<\1>", text)
    text = re.sub(r"\s+([./:-])\s+", r"\1", text)
    text = text.replace("< * >", "<*>")
    return text


def ensure_clean_tmp(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def deduplicate_exact_content(structured_path: Path, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    ensure_clean_tmp(TEMP_ROOT)
    with tempfile.NamedTemporaryFile(
        prefix=structured_path.stem + ".seen.",
        suffix=".sqlite3",
        dir=TEMP_ROOT,
        delete=False,
    ) as temp_db:
        db_path = Path(temp_db.name)

    started_at = time.time()
    total_rows = 0
    unique_rows = 0

    connection = sqlite3.connect(db_path)
    try:
        configure_sqlite(connection)
        connection.execute(
            """
            CREATE TABLE seen_content (
                digest BLOB NOT NULL,
                content TEXT NOT NULL,
                first_seen INTEGER NOT NULL,
                dup_count INTEGER NOT NULL,
                row_json TEXT NOT NULL,
                PRIMARY KEY (digest, content)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE INDEX seen_content_first_seen_idx
            ON seen_content (first_seen)
            """
        )
        connection.commit()

        insert_cursor = connection.cursor()
        update_cursor = connection.cursor()

        with structured_path.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.reader(input_handle)
            header = next(reader)
            try:
                content_index = header.index("Content")
            except ValueError as exc:
                raise RuntimeError(f"Content column not found in {structured_path}") from exc

            for row in reader:
                if not row:
                    continue

                total_rows += 1
                content = row[content_index]
                digest = hashlib.sha256(content.encode("utf-8")).digest()
                row_json = json.dumps(row, ensure_ascii=False, separators=(",", ":"))

                insert_cursor.execute(
                    """
                    INSERT OR IGNORE INTO seen_content (
                        digest,
                        content,
                        first_seen,
                        dup_count,
                        row_json
                    ) VALUES (?, ?, ?, 1, ?)
                    """,
                    (digest, content, total_rows, row_json),
                )
                if insert_cursor.rowcount == 1:
                    unique_rows += 1
                else:
                    update_cursor.execute(
                        """
                        UPDATE seen_content
                        SET dup_count = dup_count + 1
                        WHERE digest = ? AND content = ?
                        """,
                        (digest, content),
                    )

                if total_rows % PROGRESS_EVERY == 0:
                    connection.commit()
                    elapsed = time.time() - started_at
                    print(
                        f"[{structured_path.parent.name}] exact-dedup processed={total_rows} unique={unique_rows} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

        connection.commit()

        with tmp_output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.writer(output_handle)
            writer.writerow([*header, "dup_count"])
            for row_json, dup_count in connection.execute(
                """
                SELECT row_json, dup_count
                FROM seen_content
                ORDER BY first_seen
                """
            ):
                writer.writerow([*json.loads(row_json), dup_count])
    finally:
        connection.close()

    os.replace(tmp_output_path, output_path)
    db_path.unlink(missing_ok=True)

    elapsed = time.time() - started_at
    print(
        f"[{structured_path.parent.name}] exact-dedup completed rows={total_rows} unique={unique_rows} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return total_rows, unique_rows


def deduplicate_normalized_content(input_path: Path, output_path: Path) -> tuple[int, int, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    ensure_clean_tmp(TEMP_ROOT)
    with tempfile.NamedTemporaryFile(
        prefix=input_path.stem + ".normalized.",
        suffix=".sqlite3",
        dir=TEMP_ROOT,
        delete=False,
    ) as temp_db:
        db_path = Path(temp_db.name)

    started_at = time.time()
    input_unique_contents = 0
    normalized_groups = 0
    input_weight_sum = 0

    connection = sqlite3.connect(db_path)
    try:
        configure_sqlite(connection)
        connection.execute(
            """
            CREATE TABLE normalized_seen (
                digest BLOB NOT NULL,
                normalized_content TEXT NOT NULL,
                first_seen INTEGER NOT NULL,
                normalized_unique_content_count INTEGER NOT NULL,
                dup_count INTEGER NOT NULL,
                representative_dup_count INTEGER NOT NULL,
                representative_line_id INTEGER NOT NULL,
                row_json TEXT NOT NULL,
                PRIMARY KEY (digest, normalized_content)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE INDEX normalized_seen_first_seen_idx
            ON normalized_seen (first_seen)
            """
        )
        connection.commit()

        insert_cursor = connection.cursor()
        update_cursor = connection.cursor()
        replace_cursor = connection.cursor()

        with input_path.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            if not reader.fieldnames or "Content" not in reader.fieldnames or "dup_count" not in reader.fieldnames:
                raise RuntimeError(f"Expected Content and dup_count columns in {input_path}")
            fieldnames = reader.fieldnames

            for row in reader:
                input_unique_contents += 1
                dup_count = int(row["dup_count"])
                input_weight_sum += dup_count
                normalized_content = normalize_content(row["Content"])
                digest = hashlib.sha256(normalized_content.encode("utf-8")).digest()
                row_json = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                current_line_id = line_id_value(row)

                insert_cursor.execute(
                    """
                    INSERT OR IGNORE INTO normalized_seen (
                        digest,
                        normalized_content,
                        first_seen,
                        normalized_unique_content_count,
                        dup_count,
                        representative_dup_count,
                        representative_line_id,
                        row_json
                    ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                    """,
                    (
                        digest,
                        normalized_content,
                        input_unique_contents,
                        dup_count,
                        dup_count,
                        current_line_id,
                        row_json,
                    ),
                )
                if insert_cursor.rowcount == 1:
                    normalized_groups += 1
                else:
                    existing = update_cursor.execute(
                        """
                        SELECT representative_dup_count, representative_line_id
                        FROM normalized_seen
                        WHERE digest = ? AND normalized_content = ?
                        """,
                        (digest, normalized_content),
                    ).fetchone()
                    should_replace = existing is not None and (
                        dup_count > existing[0]
                        or (dup_count == existing[0] and current_line_id < existing[1])
                    )
                    update_cursor.execute(
                        """
                        UPDATE normalized_seen
                        SET normalized_unique_content_count = normalized_unique_content_count + 1,
                            dup_count = dup_count + ?
                        WHERE digest = ? AND normalized_content = ?
                        """,
                        (dup_count, digest, normalized_content),
                    )
                    if should_replace:
                        replace_cursor.execute(
                            """
                            UPDATE normalized_seen
                            SET representative_dup_count = ?,
                                representative_line_id = ?,
                                row_json = ?
                            WHERE digest = ? AND normalized_content = ?
                            """,
                            (dup_count, current_line_id, row_json, digest, normalized_content),
                        )

                if input_unique_contents % PROGRESS_EVERY == 0:
                    connection.commit()
                    elapsed = time.time() - started_at
                    print(
                        f"[{input_path.parent.name}] normalized-dedup processed={input_unique_contents} groups={normalized_groups} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

        connection.commit()

        with tmp_output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=[
                    *fieldnames,
                    "normalized_content",
                    "normalized_unique_content_count",
                ],
            )
            writer.writeheader()
            for row_json, normalized_content, group_dup_count, unique_content_count in connection.execute(
                """
                SELECT row_json, normalized_content, dup_count, normalized_unique_content_count
                FROM normalized_seen
                ORDER BY first_seen
                """
            ):
                row = json.loads(row_json)
                row["dup_count"] = str(group_dup_count)
                row["normalized_content"] = normalized_content
                row["normalized_unique_content_count"] = str(unique_content_count)
                writer.writerow(row)
    finally:
        connection.close()

    os.replace(tmp_output_path, output_path)
    db_path.unlink(missing_ok=True)

    elapsed = time.time() - started_at
    print(
        f"[{input_path.parent.name}] normalized-dedup completed input_unique={input_unique_contents} groups={normalized_groups} weight={input_weight_sum} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return input_unique_contents, normalized_groups, input_weight_sum, input_weight_sum


def find_cluster(
    clusters: list[Cluster],
    tokens: list[str],
    similarity_threshold: float,
) -> Cluster | None:
    best_cluster = None
    best_similarity = -1.0
    for cluster in clusters:
        similarity = token_similarity(tokens, cluster.template_tokens)
        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = cluster
    if best_cluster is not None and best_similarity >= similarity_threshold:
        return best_cluster
    return None


def should_update_representative(cluster: Cluster, row: dict[str, str], dup_count: int) -> bool:
    current_line_id = line_id_value(row)
    return (
        dup_count > cluster.representative_dup_count
        or (dup_count == cluster.representative_dup_count and current_line_id < cluster.representative_line_id)
    )


def cluster_normalized_file(
    dataset_name: str,
    input_path: Path,
    output_path: Path,
    similarity_threshold: float,
    max_bucket_clusters: int,
) -> tuple[dict[str, int], list[Cluster]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    started_at = time.time()
    rows: list[dict[str, str]] = []
    buckets: dict[tuple[int, str, str], list[Cluster]] = {}
    clusters: list[Cluster] = []
    total_rows = 0
    total_unique = 0
    total_weight = 0

    with input_path.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if not reader.fieldnames or "normalized_content" not in reader.fieldnames:
            raise RuntimeError(f"Expected normalized_content column in {input_path}")
        fieldnames = reader.fieldnames

        for row in reader:
            row_id = len(rows)
            rows.append(row)
            total_rows += 1

            normalized_content = row["normalized_content"]
            tokens = tokenize_normalized(normalized_content)
            key = bucket_key(tokens)
            candidate_clusters = buckets.setdefault(key, [])
            cluster = None
            if len(candidate_clusters) <= max_bucket_clusters:
                cluster = find_cluster(candidate_clusters, tokens, similarity_threshold)

            if cluster is None:
                cluster = Cluster(
                    cluster_id=f"{dataset_name}_C{len(clusters) + 1:07d}",
                    token_count=len(tokens),
                    tokens=tokens,
                    template_tokens=list(tokens),
                )
                clusters.append(cluster)
                candidate_clusters.append(cluster)
            else:
                update_template_tokens(cluster.template_tokens, tokens)

            dup_count = int(row["dup_count"])
            unique_count = int(row.get("normalized_unique_content_count", "1"))
            cluster.size_rows += 1
            cluster.size_unique += unique_count
            cluster.weight_dup_sum += dup_count
            cluster.member_row_ids.append(row_id)
            total_unique += unique_count
            total_weight += dup_count

            if should_update_representative(cluster, row, dup_count):
                cluster.representative_content = row["Content"]
                cluster.representative_normalized_content = normalized_content
                cluster.representative_dup_count = dup_count
                cluster.representative_line_id = line_id_value(row)

            row["cluster_id"] = cluster.cluster_id

            if total_rows % PROGRESS_EVERY == 0:
                elapsed = time.time() - started_at
                print(
                    f"[{dataset_name}] cluster processed={total_rows} clusters={len(clusters)} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    cluster_lookup = {cluster.cluster_id: cluster for cluster in clusters}
    with tmp_output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(
            output_handle,
            fieldnames=[
                *fieldnames,
                "cluster_id",
                "cluster_size_unique",
                "cluster_weight_dup_sum",
                "cluster_template_guess",
            ],
        )
        writer.writeheader()
        for row in rows:
            cluster = cluster_lookup[row["cluster_id"]]
            row["cluster_size_unique"] = str(cluster.size_unique)
            row["cluster_weight_dup_sum"] = str(cluster.weight_dup_sum)
            row["cluster_template_guess"] = template_guess_from_tokens(cluster.template_tokens)
            writer.writerow(row)

    os.replace(tmp_output_path, output_path)

    elapsed = time.time() - started_at
    print(
        f"[{dataset_name}] cluster completed rows={total_rows} clusters={len(clusters)} unique={total_unique} weight={total_weight} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return (
        {
            "normalized_rows": total_rows,
            "cluster_count": len(clusters),
            "clustered_unique_content_count": total_unique,
            "clustered_dup_weight_sum": total_weight,
        },
        clusters,
    )


def retention_recommendation(cluster: Cluster) -> int:
    if cluster.weight_dup_sum >= 100_000 and cluster.size_unique >= 100:
        return 3
    if cluster.weight_dup_sum >= 1_000 and cluster.size_unique >= 10:
        return 2
    return 1


def balanced_keep_k(row: dict[str, str], cluster_size_normalized_rows: int) -> int:
    cluster_size_unique = int(row["cluster_size_unique"])
    cluster_weight = int(row["cluster_weight_dup_sum"])

    keep_k = 1
    if (
        cluster_size_unique >= 10
        or cluster_weight >= 1_000
        or cluster_size_normalized_rows >= 2
    ):
        keep_k = 2
    if (
        cluster_size_unique >= 100
        or cluster_weight >= 100_000
        or cluster_size_normalized_rows >= 5
    ):
        keep_k = 3

    # Do not duplicate near-identical variable-only examples just to satisfy keep_k.
    return min(keep_k, 3, cluster_size_normalized_rows)


def content_complexity(row: dict[str, str]) -> tuple[int, int, int, int, int]:
    content = row["Content"]
    normalized = row.get("normalized_content", "")
    normalized_tokens = tokenize_normalized(normalized)
    placeholder_count = sum(1 for token in normalized_tokens if PLACEHOLDER_RE.match(token))
    symbol_count = sum(1 for char in content if not char.isalnum() and not char.isspace())
    token_count = len(TOKEN_RE.findall(content))
    path_or_url_bonus = int("/" in content or "://" in content)
    return (len(content), token_count, symbol_count, placeholder_count, path_or_url_bonus)


def jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def row_token_set(row: dict[str, str]) -> set[str]:
    return set(tokenize_normalized(row.get("normalized_content", row["Content"].lower())))


def line_id_sort_value(row: dict[str, str]) -> int:
    return line_id_value(row)


def select_frequency_representative(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: (-int(row["dup_count"]), line_id_sort_value(row)))


def select_complex_sample(rows: list[dict[str, str]], selected_ids: set[int]) -> dict[str, str] | None:
    candidates = [row for row in rows if id(row) not in selected_ids]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (content_complexity(row), int(row["dup_count"]), -line_id_sort_value(row)),
    )


def select_diverse_sample(rows: list[dict[str, str]], selected: list[dict[str, str]]) -> dict[str, str] | None:
    selected_ids = {id(row) for row in selected}
    candidates = [row for row in rows if id(row) not in selected_ids]
    if not candidates:
        return None

    selected_token_sets = [row_token_set(row) for row in selected]
    selected_lengths = [len(row["Content"]) for row in selected]

    def diverse_key(row: dict[str, str]) -> tuple[float, int, int, int]:
        token_set = row_token_set(row)
        min_distance = min(jaccard_distance(token_set, selected_set) for selected_set in selected_token_sets)
        length_gap = min(abs(len(row["Content"]) - selected_length) for selected_length in selected_lengths)
        return (min_distance, length_gap, int(row["dup_count"]), -line_id_sort_value(row))

    return max(candidates, key=diverse_key)


def select_cluster_rows(rows: list[dict[str, str]], keep_k: int) -> list[tuple[dict[str, str], str]]:
    selected: list[tuple[dict[str, str], str]] = []
    representative = select_frequency_representative(rows)
    selected.append((representative, "frequency_representative"))

    if keep_k >= 2:
        complex_row = select_complex_sample(rows, {id(row) for row, _ in selected})
        if complex_row is not None:
            selected.append((complex_row, "complex_sample"))

    if keep_k >= 3:
        diverse_row = select_diverse_sample(rows, [row for row, _ in selected])
        if diverse_row is not None:
            selected.append((diverse_row, "diverse_sample"))

    return selected


def select_balanced_file(dataset_name: str, input_path: Path, output_path: Path) -> tuple[dict[str, int], list[tuple[str, int, int]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    clusters: dict[str, list[dict[str, str]]] = {}
    started_at = time.time()

    with input_path.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if not reader.fieldnames or "cluster_id" not in reader.fieldnames:
            raise RuntimeError(f"Expected cluster_id column in {input_path}")
        fieldnames = reader.fieldnames
        for row in reader:
            clusters.setdefault(row["cluster_id"], []).append(row)

    selected_rows: list[dict[str, str]] = []
    keep_1_count = 0
    keep_2_count = 0
    keep_3_count = 0
    selected_weight_sum = 0
    selected_unique_sum = 0
    cluster_keep_rows: list[tuple[str, int, int]] = []

    for cluster_id in sorted(clusters):
        rows = clusters[cluster_id]
        keep_k = balanced_keep_k(rows[0], len(rows))
        if keep_k == 1:
            keep_1_count += 1
        elif keep_k == 2:
            keep_2_count += 1
        else:
            keep_3_count += 1

        cluster_keep_rows.append((cluster_id, keep_k, len(rows)))
        for rank, (row, reason) in enumerate(select_cluster_rows(rows, keep_k), start=1):
            output_row = dict(row)
            output_row["cluster_keep_k"] = str(keep_k)
            output_row["selected_rank"] = str(rank)
            output_row["selection_reason"] = reason
            selected_rows.append(output_row)
            selected_weight_sum += int(row["dup_count"])
            selected_unique_sum += int(row.get("normalized_unique_content_count", "1"))

    selected_rows.sort(key=lambda row: (row["cluster_id"], int(row["selected_rank"]), line_id_sort_value(row)))

    with tmp_output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(
            output_handle,
            fieldnames=[
                *fieldnames,
                "cluster_keep_k",
                "selected_rank",
                "selection_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    os.replace(tmp_output_path, output_path)

    elapsed = time.time() - started_at
    print(
        f"[{dataset_name}] selected-balanced completed clusters={len(clusters)} selected={len(selected_rows)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return (
        {
            "cluster_count": len(clusters),
            "selected_log_count": len(selected_rows),
            "keep_1_cluster_count": keep_1_count,
            "keep_2_cluster_count": keep_2_count,
            "keep_3_cluster_count": keep_3_count,
            "selected_dup_weight_sum": selected_weight_sum,
            "selected_unique_content_sum": selected_unique_sum,
        },
        cluster_keep_rows,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LogHub preprocessing without using EventTemplate.")
    parser.add_argument(
        "--stage",
        choices=["exact-dedup", "normalized-dedup", "cluster", "all-normalized", "select-balanced"],
        default="exact-dedup",
        help="Processing stage to run.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Only process the named dataset directory. Can be repeated.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.82,
        help="Token-position similarity threshold for normalized_content clustering.",
    )
    parser.add_argument(
        "--max-bucket-clusters",
        type=int,
        default=5000,
        help="Disable expensive fuzzy matching in a bucket after this many clusters.",
    )
    return parser


def run_exact_dedup(dataset_dirs: list[Path]) -> None:
    results = []
    for dataset_dir in dataset_dirs:
        structured_path = find_single_file(dataset_dir, "_structured.csv")
        output_path = EXACT_DEDUP_ROOT / dataset_dir.name / structured_path.name
        original_log_count, deduplicated_log_count = deduplicate_exact_content(
            structured_path,
            output_path,
        )
        results.append(
            (
                dataset_dir.name,
                original_log_count,
                deduplicated_log_count,
                original_log_count - deduplicated_log_count,
            )
        )

    with EXACT_DEDUP_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_name",
                "original_log_count",
                "deduplicated_log_count",
                "removed_duplicate_count",
            ]
        )
        writer.writerows(results)
    print(f"Summary written to {EXACT_DEDUP_SUMMARY_PATH}", flush=True)


def run_normalized_dedup(dataset_dirs: list[Path]) -> None:
    results = []
    for dataset_dir in dataset_dirs:
        input_path = find_input_file(EXACT_DEDUP_ROOT, dataset_dir.name)
        output_path = NORMALIZED_DEDUP_ROOT / dataset_dir.name / input_path.name
        input_unique_count, normalized_count, input_weight_sum, output_weight_sum = deduplicate_normalized_content(
            input_path,
            output_path,
        )
        results.append(
            (
                dataset_dir.name,
                input_unique_count,
                normalized_count,
                input_unique_count - normalized_count,
                input_weight_sum,
                output_weight_sum,
            )
        )

    with NORMALIZED_DEDUP_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_name",
                "exact_unique_content_count",
                "normalized_deduplicated_count",
                "removed_by_normalization_count",
                "input_dup_weight_sum",
                "output_dup_weight_sum",
            ]
        )
        writer.writerows(results)
    print(f"Summary written to {NORMALIZED_DEDUP_SUMMARY_PATH}", flush=True)


def run_select_balanced(dataset_dirs: list[Path]) -> None:
    dataset_results = []

    for dataset_dir in dataset_dirs:
        input_path = find_input_file(CLUSTERED_ROOT, dataset_dir.name)
        output_path = SELECTED_BALANCED_ROOT / dataset_dir.name / input_path.name
        dataset_stats, _cluster_keep_rows = select_balanced_file(
            dataset_dir.name,
            input_path,
            output_path,
        )
        dataset_results.append(
            (
                dataset_dir.name,
                dataset_stats["cluster_count"],
                dataset_stats["selected_log_count"],
                dataset_stats["keep_1_cluster_count"],
                dataset_stats["keep_2_cluster_count"],
                dataset_stats["keep_3_cluster_count"],
                dataset_stats["selected_dup_weight_sum"],
                dataset_stats["selected_unique_content_sum"],
            )
        )

    with SELECTED_BALANCED_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_name",
                "cluster_count",
                "selected_log_count",
                "keep_1_cluster_count",
                "keep_2_cluster_count",
                "keep_3_cluster_count",
                "selected_dup_weight_sum",
                "selected_unique_content_sum",
            ]
        )
        writer.writerows(dataset_results)
    print(f"Selected balanced summary written to {SELECTED_BALANCED_SUMMARY_PATH}", flush=True)


def run_cluster(dataset_dirs: list[Path], similarity_threshold: float, max_bucket_clusters: int) -> None:
    dataset_results = []
    cluster_rows = []

    for dataset_dir in dataset_dirs:
        input_path = find_input_file(NORMALIZED_DEDUP_ROOT, dataset_dir.name)
        output_path = CLUSTERED_ROOT / dataset_dir.name / input_path.name
        dataset_stats, clusters = cluster_normalized_file(
            dataset_dir.name,
            input_path,
            output_path,
            similarity_threshold,
            max_bucket_clusters,
        )
        dataset_results.append(
            (
                dataset_dir.name,
                dataset_stats["normalized_rows"],
                dataset_stats["cluster_count"],
                dataset_stats["clustered_unique_content_count"],
                dataset_stats["clustered_dup_weight_sum"],
            )
        )
        for cluster in clusters:
            cluster_rows.append(
                (
                    dataset_dir.name,
                    cluster.cluster_id,
                    cluster.size_rows,
                    cluster.size_unique,
                    cluster.weight_dup_sum,
                    retention_recommendation(cluster),
                    template_guess_from_tokens(cluster.template_tokens),
                    cluster.representative_content,
                    cluster.representative_normalized_content,
                )
            )

    with CLUSTER_DATASET_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_name",
                "normalized_rows",
                "cluster_count",
                "clustered_unique_content_count",
                "clustered_dup_weight_sum",
            ]
        )
        writer.writerows(dataset_results)
    print(f"Dataset cluster summary written to {CLUSTER_DATASET_SUMMARY_PATH}", flush=True)

    with CLUSTER_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_name",
                "cluster_id",
                "cluster_size_normalized_rows",
                "cluster_size_unique",
                "cluster_weight_dup_sum",
                "recommended_keep_k_balanced",
                "cluster_template_guess",
                "representative_content",
                "representative_normalized_content",
            ]
        )
        writer.writerows(cluster_rows)
    print(f"Cluster summary written to {CLUSTER_SUMMARY_PATH}", flush=True)


def main() -> int:
    args = build_argument_parser().parse_args()
    set_csv_field_size_limit()
    dataset_dirs = selected_dataset_dirs(args.datasets)

    if args.stage == "exact-dedup":
        run_exact_dedup(dataset_dirs)
    elif args.stage == "normalized-dedup":
        run_normalized_dedup(dataset_dirs)
    elif args.stage == "cluster":
        run_cluster(dataset_dirs, args.cluster_threshold, args.max_bucket_clusters)
    elif args.stage == "all-normalized":
        run_normalized_dedup(dataset_dirs)
        run_cluster(dataset_dirs, args.cluster_threshold, args.max_bucket_clusters)
    elif args.stage == "select-balanced":
        run_select_balanced(dataset_dirs)
    else:
        raise RuntimeError(f"Unsupported stage: {args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
