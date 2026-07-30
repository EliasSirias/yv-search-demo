import csv
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_FILE = BASE_DIR / "feedback" / "search_feedback.csv"


def save_feedback(
    question: str,
    source: str,
    relative_path: str,
    rank: int,
    confidence: str,
    helpful: bool,
) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = FEEDBACK_FILE.exists()

    with FEEDBACK_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "question",
                "source",
                "relative_path",
                "rank",
                "confidence",
                "helpful",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "source": source,
                "relative_path": relative_path,
                "rank": rank,
                "confidence": confidence,
                "helpful": helpful,
            }
        )
