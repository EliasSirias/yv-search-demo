import re
from pathlib import Path
from collections import defaultdict


def format_source_name(filename: str) -> str:
    name = Path(filename).stem

    for prefix in ("TS_", "HT_", "CFG_", "REP_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]

    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()

    return name.title()


def format_category_name(category: str) -> str:
    return category.replace("_", " ").replace("-", " ").strip().title()


def format_knowledge_base_name(knowledge_base: str) -> str:
    return knowledge_base.replace("_", " ").replace("-", " ").strip().title()


def group_results_by_path(results: list[dict]) -> dict:
    grouped = defaultdict(list)

    for item in results:
        relative_path = item["relative_path"]

        grouped[relative_path].append(
            {
                "text": item["text"],
                "score": item["score"],
                "source": item["source"],
                "knowledge_base": item["knowledge_base"],
                "category": item["category"],
            }
        )

    return grouped
