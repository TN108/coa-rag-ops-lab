from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentService:
    def __init__(
        self,
        output_directory: str = "backend/data/experiments",
    ) -> None:
        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def generate_experiment_id() -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_uuid = uuid.uuid4().hex[:4]

        return f"EXP-{timestamp}-{short_uuid}"

    def save_experiment(
        self,
        experiment_id: str,
        configuration: dict[str, Any],
        results: dict[str, Any],
        experiment_type: str = "single_run",
    ) -> Path:
        experiment_record = {
            "experiment_id": experiment_id,
            "experiment_type": experiment_type,
            "created_at": datetime.now().isoformat(),
            "configuration": configuration,
            "results": results,
        }

        output_path = (
            self.output_directory
            / f"{experiment_id}.json"
        )

        try:
            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    experiment_record,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to save experiment to {output_path}"
            ) from exc

        return output_path


experiment_service = ExperimentService()