"""Bidirectional schema linker, BM25/LCS value RAG, and FK graph closure for Assay.

Implements:
  1. Lexical cell retrieval: BM25 candidate shortlist + RapidFuzz normalized LCS /
     token sort ratio to ground query literals.
  2. Dense column retrieval: cosine similarity via nomic-embed-text.
  3. Bidirectional Schema Linking: Table-First ∪ Column-First set union.
  4. Foreign-Key Graph Closure: shortest-path traversal to restore required bridge tables.
  5. Compact M-Schema formatting with backtick columns, types, sample values,
     explicit NULL flags, and dialect-specific rules.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from rapidfuzz import fuzz

from src.llm import default_client
from src.schema.indexer import ColumnInfo, DatabaseIndex, TableInfo, load_index, tokenize


@dataclass
class MatchedValue:
    table: str
    column: str
    value: str
    query_term: str
    score: float


@dataclass
class LinkedSchema:
    db_id: str
    selected_tables: List[str]
    selected_columns: Dict[str, List[str]]  # table -> list of col names
    matched_values: List[MatchedValue]
    fk_relationships: List[Tuple[str, str, str, str]]  # [(t1, c1, t2, c2), ...]
    m_schema_str: str


class SchemaLinker:
    """Performs schema linking, cell value RAG, FK closure, and M-Schema generation."""

    def __init__(self, index: Union[DatabaseIndex, str, Path], index_dir: str = "data/benchmarks/indices"):
        if isinstance(index, DatabaseIndex):
            self.index = index
        else:
            self.index = load_index(str(index), index_dir)

    def extract_query_phrases(self, text: str) -> List[str]:
        """Extract high-confidence candidate literal phrases (quoted strings, proper nouns) from text."""
        if not text:
            return []

        phrases: List[str] = []

        # 1. Quoted strings: '...', "..."
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
        for q in quoted:
            q_clean = q.strip()
            if q_clean and not q_clean.isdigit():
                phrases.append(q_clean)

        # 2. Capitalized phrases (proper nouns, entity names)
        schema_stopwords = {
            "the", "and", "or", "what", "which", "how", "many", "give", "list", "show",
            "find", "name", "names", "all", "each", "every", "with", "from", "that", "this",
            "than", "more", "less", "where", "have", "been", "were", "their", "there",
            "between", "most", "least", "highest", "lowest", "total", "number", "count",
            "sum", "average", "avg", "max", "min", "greater", "fewer", "equal", "order",
            "by", "asc", "desc", "limit", "top", "database", "table", "column", "record",
            "school", "schools", "game", "games", "team", "teams", "card", "cards",
            "club", "clubs", "user", "users", "student", "students", "year", "date",
            "county", "counties", "city", "cities", "state", "states", "please", "whose",
            "first", "last", "id", "ids", "type", "types", "code", "codes", "status",
        }

        # Extract capitalized entities like "Alameda County", "Formula 1", "Michael Schumacher"
        for cap in re.findall(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b", text):
            cap_clean = cap.strip()
            if cap_clean.lower() not in schema_stopwords and len(cap_clean) >= 3:
                phrases.append(cap_clean)
                for part in cap_clean.split():
                    if part.lower() not in schema_stopwords and len(part) >= 4:
                        phrases.append(part)

        # Deduplicate while preserving order
        seen = set()
        unique_phrases = []
        for p in phrases:
            p_lower = p.lower()
            if p_lower not in seen:
                seen.add(p_lower)
                unique_phrases.append(p)

        return unique_phrases

    def retrieve_cell_values(
        self,
        question: str,
        evidence: str = "",
        top_k: int = 5,
        min_score: float = 85.0,
    ) -> List[MatchedValue]:
        """Lexical cell-value retrieval: BM25 candidate shortlist + RapidFuzz matching."""
        if not self.index.bm25_values or not self.index.value_entries:
            return []

        combined_text = f"{question} {evidence}".strip()
        query_phrases = self.extract_query_phrases(combined_text)
        if not query_phrases:
            return []

        matched_list: List[MatchedValue] = []
        seen_matches: Set[Tuple[str, str, str]] = set()

        for phrase in query_phrases:
            p_tokens = tokenize(phrase)
            if not p_tokens:
                continue

            # Query BM25 values
            scores = self.index.bm25_values.get_scores(p_tokens)
            # Shortlist top 15 candidates
            top_indices = np.argsort(scores)[::-1][:15]

            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                t_name, c_name, val_str = self.index.value_entries[idx]
                key = (t_name, c_name, val_str.lower())
                if key in seen_matches:
                    continue

                # Don't match if value is > 2.5x length of phrase unless exact match
                if len(val_str) > len(phrase) * 2.5 and phrase.lower() != val_str.lower():
                    continue

                # Fuzzy string scoring
                ratio1 = fuzz.token_sort_ratio(phrase.lower(), val_str.lower())
                ratio2 = 100.0 if phrase.lower() == val_str.lower() else 0.0
                best_ratio = max(ratio1, ratio2)

                if best_ratio >= min_score:
                    seen_matches.add(key)
                    matched_list.append(
                        MatchedValue(
                            table=t_name,
                            column=c_name,
                            value=val_str,
                            query_term=phrase,
                            score=float(best_ratio),
                        )
                    )

        # Sort by score descending and keep top_k
        matched_list.sort(key=lambda m: m.score, reverse=True)
        return matched_list[:top_k]

    def score_tables(self, question: str, evidence: str = "") -> Dict[str, float]:
        """Compute table relevance scores using BM25 and token overlap."""
        scores: Dict[str, float] = {t: 0.0 for t in self.index.table_names_list}
        query_tokens = tokenize(f"{question} {evidence}")
        if not query_tokens:
            return scores

        if self.index.bm25_tables:
            bm25_raw = self.index.bm25_tables.get_scores(query_tokens)
            max_s = max(bm25_raw) if len(bm25_raw) > 0 and max(bm25_raw) > 0 else 1.0
            for t_name, s in zip(self.index.table_names_list, bm25_raw):
                scores[t_name] += max(0.0, s / max_s)

        # Exact table name mention bonus
        q_lower = f"{question} {evidence}".lower()
        for t_name in self.index.table_names_list:
            t_lower = t_name.lower()
            if t_lower in q_lower or t_lower.replace("_", " ") in q_lower:
                scores[t_name] += 1.5

        return scores

    def score_columns(
        self,
        question: str,
        evidence: str = "",
        use_embeddings: bool = True,
    ) -> Dict[Tuple[str, str], float]:
        """Compute column relevance scores using Dense embeddings, BM25, and lexical matches."""
        scores: Dict[Tuple[str, str], float] = {k: 0.0 for k in self.index.column_keys_list}
        combined_text = f"{question} {evidence}".strip()
        query_tokens = tokenize(combined_text)
        if not query_tokens:
            return scores

        # 1. BM25 Column Scores
        if self.index.bm25_columns:
            bm25_raw = self.index.bm25_columns.get_scores(query_tokens)
            max_s = max(bm25_raw) if len(bm25_raw) > 0 and max(bm25_raw) > 0 else 1.0
            for key, s in zip(self.index.column_keys_list, bm25_raw):
                scores[key] += 0.35 * max(0.0, s / max_s)

        # 2. Dense Cosine Similarity
        if use_embeddings and self.index.column_embeddings:
            try:
                q_vec = default_client.get_embedding(combined_text)[0]
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 1e-6:
                    q_arr = np.array(q_vec) / q_norm
                    for key in self.index.column_keys_list:
                        if key in self.index.column_embeddings:
                            col_vec = np.array(self.index.column_embeddings[key])
                            c_norm = np.linalg.norm(col_vec)
                            if c_norm > 1e-6:
                                cos_sim = float(np.dot(q_arr, col_vec / c_norm))
                                # Normalize [-1, 1] to [0, 1]
                                sim_norm = max(0.0, (cos_sim + 1.0) / 2.0)
                                scores[key] += 0.45 * sim_norm
            except Exception:
                pass

        # 3. Lexical / String Matching Bonus
        q_lower = combined_text.lower()
        for (t_name, c_name), _ in scores.items():
            c_lower = c_name.lower()
            c_clean = c_lower.replace("_", " ")
            if c_lower in q_lower or c_clean in q_lower:
                scores[(t_name, c_name)] += 1.0

        return scores

    def foreign_key_graph_closure(
        self,
        selected_tables: Set[str],
    ) -> Tuple[Set[str], List[Tuple[str, str, str, str]]]:
        """Restore missing bridge tables by computing shortest paths on FK graph.
        
        Returns:
            Tuple of (closed_tables_set, list_of_active_fks)
        """
        if len(selected_tables) <= 1:
            return selected_tables, []

        all_tables = list(self.index.tables.keys())
        # Build undirected graph: node -> set of neighbor nodes
        adj: Dict[str, Set[str]] = {t: set() for t in all_tables}
        for t_name, edges in self.index.fk_graph.items():
            for target_tbl, _, _ in edges:
                if t_name in adj and target_tbl in adj:
                    adj[t_name].add(target_tbl)
                    adj[target_tbl].add(t_name)

        # Find shortest path between pairs using BFS
        def shortest_path(src: str, dst: str) -> Optional[List[str]]:
            if src == dst:
                return [src]
            queue = collections.deque([[src]])
            visited = {src}
            while queue:
                path = queue.popleft()
                node = path[-1]
                if node == dst:
                    return path
                for neighbor in adj.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(path + [neighbor])
            return None

        closed_tables = set(selected_tables)
        tbl_list = sorted(list(selected_tables))

        # Check connectivity between all pairs of selected tables
        for i in range(len(tbl_list)):
            for j in range(i + 1, len(tbl_list)):
                t1, t2 = tbl_list[i], tbl_list[j]
                path = shortest_path(t1, t2)
                if path:
                    # If path exists and adds at most 2 bridge tables, add them
                    if len(path) <= 4:
                        for bridge in path:
                            closed_tables.add(bridge)

        # Collect active foreign keys between closed_tables
        active_fks: List[Tuple[str, str, str, str]] = []
        seen_fks = set()
        for t_name in closed_tables:
            t_obj = self.index.tables[t_name]
            for c_from, t_target, c_target in t_obj.foreign_keys:
                if t_target in closed_tables:
                    fk_pair = tuple(sorted([(t_name, c_from), (t_target, c_target)]))
                    if fk_pair not in seen_fks:
                        seen_fks.add(fk_pair)
                        active_fks.append((t_name, c_from, t_target, c_target))

        return closed_tables, active_fks

    def link_and_prune(
        self,
        question: str,
        evidence: str = "",
        max_tables: int = 4,
        max_columns_per_table: int = 10,
        use_embeddings: bool = True,
    ) -> LinkedSchema:
        """Perform bidirectional linking, value RAG, FK closure, and M-Schema formatting."""
        # 1. Retrieve cell values
        matched_values = self.retrieve_cell_values(question, evidence)
        cell_tables = {m.table for m in matched_values if m.table in self.index.tables}

        # 2. Table-First scoring
        table_scores = self.score_tables(question, evidence)
        for t in cell_tables:
            table_scores[t] = table_scores.get(t, 0.0) + 1.2

        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top tables with positive relevance
        table_first_tables = set()
        for t, s in sorted_tables[:max_tables]:
            if s > 0.15 or len(table_first_tables) == 0:
                table_first_tables.add(t)

        # 3. Column-First scoring
        col_scores = self.score_columns(question, evidence, use_embeddings=use_embeddings)
        sorted_cols = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)

        # Include tables for top 8 highest-scoring columns
        col_first_tables: Set[str] = set()
        for (t_name, _), s in sorted_cols[:8]:
            if s > 0.25:
                col_first_tables.add(t_name)

        # 4. Bidirectional Union: Table-First ∪ Column-First ∪ Cell-Value Tables
        candidate_tables = table_first_tables.union(col_first_tables).union(cell_tables)

        # 5. Foreign-Key Graph Closure
        closed_tables, active_fks = self.foreign_key_graph_closure(candidate_tables)

        # 6. Column Selection per Table
        selected_columns: Dict[str, List[str]] = {}
        for t_name in sorted(list(closed_tables)):
            if t_name not in self.index.tables:
                continue
            t_obj = self.index.tables[t_name]
            all_cols = list(t_obj.columns.keys())

            # If table has <= max_columns_per_table, keep all columns
            if len(all_cols) <= max_columns_per_table:
                selected_columns[t_name] = all_cols
                continue

            # Otherwise, prioritize:
            # a) Primary keys
            # b) Foreign keys
            # c) Matched cell value columns
            # d) Highest scoring columns
            kept: Set[str] = set()
            for pk in t_obj.primary_keys:
                if pk in t_obj.columns:
                    kept.add(pk)
            for c_from, _, _ in t_obj.foreign_keys:
                if c_from in t_obj.columns:
                    kept.add(c_from)
            for m in matched_values:
                if m.table == t_name and m.column in t_obj.columns:
                    kept.add(m.column)

            # Add highest scoring columns for this table
            table_col_scores = [
                (c, col_scores.get((t_name, c), 0.0)) for c in all_cols if c not in kept
            ]
            table_col_scores.sort(key=lambda x: x[1], reverse=True)
            for c, _ in table_col_scores:
                if len(kept) >= max_columns_per_table:
                    break
                kept.add(c)

            # Preserve original table column ordering
            selected_columns[t_name] = [c for c in all_cols if c in kept]

        # 7. Format Compact M-Schema Prompt
        m_schema_str = self.format_m_schema(
            closed_tables=closed_tables,
            selected_columns=selected_columns,
            matched_values=matched_values,
            active_fks=active_fks,
        )

        return LinkedSchema(
            db_id=self.index.db_id,
            selected_tables=sorted(list(closed_tables)),
            selected_columns=selected_columns,
            matched_values=matched_values,
            fk_relationships=active_fks,
            m_schema_str=m_schema_str,
        )

    def format_m_schema(
        self,
        closed_tables: Set[str],
        selected_columns: Dict[str, List[str]],
        matched_values: List[MatchedValue],
        active_fks: List[Tuple[str, str, str, str]],
    ) -> str:
        """Format clean, compact DDL M-Schema with backtick columns, types, samples, and NULL flags."""
        lines: List[str] = ["【Compact Database Schema】"]

        for t_name in sorted(list(closed_tables)):
            if t_name not in self.index.tables:
                continue
            t_obj = self.index.tables[t_name]
            t_desc = f" -- {t_obj.description}" if t_obj.description else ""
            lines.append(f"# Table: `{t_name}`{t_desc}")

            cols_to_render = selected_columns.get(t_name, list(t_obj.columns.keys()))
            for c_name in cols_to_render:
                c_obj = t_obj.columns[c_name]
                meta_tags = [c_obj.data_type]
                if c_obj.is_pk:
                    meta_tags.append("PRIMARY KEY")
                for tgt_tbl, tgt_col in c_obj.foreign_keys:
                    if tgt_tbl in closed_tables:
                        meta_tags.append(f"FK -> `{tgt_tbl}`.`{tgt_col}`")

                tag_str = " ".join(meta_tags)
                col_line = f"  `{c_name}` {tag_str}"

                # Compact single-line comment for description and samples
                comments = []
                if c_obj.description:
                    clean_desc = re.sub(r"[\r\n\t•]+", " ", c_obj.description).strip()
                    clean_desc = re.sub(r"\s+", " ", clean_desc)
                    # Exclude long descriptions or code maps
                    if len(clean_desc) > 60:
                        clean_desc = clean_desc[:60].rsplit(" ", 1)[0] + "..."
                    if clean_desc and clean_desc.lower() != c_name.lower():
                        comments.append(clean_desc)

                if c_obj.samples:
                    sample_strs = [
                        repr(str(s)[:25]) if isinstance(s, str) else str(s)
                        for s in c_obj.samples[:3]
                    ]
                    comments.append(f"e.g., [{', '.join(sample_strs)}]")

                if comments:
                    col_line += f" -- {', '.join(comments)}"

                lines.append(col_line)

            lines.append("")

        # Foreign Key Relationships Section
        if active_fks:
            lines.append("【Foreign Key Relationships】")
            for t1, c1, t2, c2 in active_fks:
                lines.append(f"`{t1}`.`{c1}` = `{t2}`.`{c2}`")
            lines.append("")

        # Matched Cell Values Section
        if matched_values:
            lines.append("【Matched Cell Values (Use exact strings for filters)】")
            for m in matched_values:
                val_repr = repr(m.value)
                lines.append(f"- `{m.table}`.`{m.column}` = {val_repr}")
            lines.append("")

        return "\n".join(lines).strip()
