from fastapi import (
    APIRouter,
    HTTPException,
)
from pydantic import (
    BaseModel,
    Field,
)

from app.services.coa_evaluation_service import (
    COAEvaluationService,
)


router = APIRouter(
    tags=["COA Evaluation"],
)


class EvaluationRequest(BaseModel):
    dataset_path: str = Field(
        default=(
            "data/evaluation/"
            "evaluation_dataset.json"
        ),
        min_length=1,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    min_retrieval_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
    )

    save_report: bool = True


@router.post(
    "/coa/evaluation/run"
)
def run_coa_evaluation(
    request: EvaluationRequest,
):
    try:
        service = COAEvaluationService(
            dataset_path=(
                request.dataset_path
            )
        )

        return service.run(
            top_k=request.top_k,
            min_retrieval_score=(
                request.min_retrieval_score
            ),
            save_report=(
                request.save_report
            ),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "COA evaluation failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
        ) from error