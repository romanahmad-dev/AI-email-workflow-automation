"""FastAPI application entry point.

Exposes a single POST endpoint, `/process-email`, which accepts an
email body and returns the workflow decision (category, priority,
recommended action, extracted keywords).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from workflow import EmailWorkflow


# Path to the static UI shipped with the project.
INDEX_HTML = Path(__file__).parent / "index.html"

# Optional URL prefix when running behind a reverse proxy (e.g. in a
# Replit workspace where this service is mounted at "/email-app").
# Defaults to "" so local development works at "/" with no surprises.
BASE_PATH = os.getenv("BASE_PATH", "").rstrip("/")


# Configure root logger once at import time. A simple format keeps
# the logs readable in a terminal and easy to ship to a log collector.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("email-workflow")


# Request and response models. Defining them with Pydantic gives us
# automatic validation, clear error messages, and OpenAPI docs.
class EmailRequest(BaseModel):
    email_text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Raw email body to classify.",
    )

    @field_validator("email_text")
    @classmethod
    def _strip_and_check(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("email_text must not be empty or whitespace only")
        return stripped


class EmailResponse(BaseModel):
    category: str
    priority: str
    recommended_action: str
    keywords: List[str]


# The workflow is created once during the lifespan of the app so the
# scikit-learn model is trained a single time, not per request.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: training email classifier")
    app.state.workflow = EmailWorkflow()
    yield
    logger.info("Shutting down email workflow service")


app = FastAPI(
    title="AI Email Workflow",
    description="Classify incoming emails and recommend an action.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=f"{BASE_PATH}/docs",
    openapi_url=f"{BASE_PATH}/openapi.json",
)


"""All routes live on a router so they can be mounted under an optional
  prefix (BASE_PATH). This keeps the app reusable both standalone and
  behind a reverse proxy that does not strip the path prefix."""
router = APIRouter()


@router.get("/", include_in_schema=False)
def read_root() -> FileResponse:
    """Serve the simple HTML UI for trying the API in a browser."""
    return FileResponse(INDEX_HTML)


@router.get("/info", tags=["health"])
def service_info() -> dict:
    """Small JSON description of the service (handy for scripts)."""
    return {
        "service": "AI Email Workflow",
        "status": "ok",
        "endpoint": "POST /process-email",
    }


@router.get("/health", tags=["health"])
def health_check() -> dict:
    """Lightweight health check for uptime monitors."""
    return {"status": "ok"}


@router.post(
    "/process-email",
    response_model=EmailResponse,
    tags=["workflow"],
    summary="Classify an email and return the recommended action.",
)
def process_email(payload: EmailRequest, request: Request) -> EmailResponse:
    """Run the email workflow and return a structured decision."""
    workflow: EmailWorkflow = request.app.state.workflow

    try:
        result = workflow.generate_response(payload.email_text)
    except ValueError as error:
        # Raised by the workflow when the cleaned text is empty.
        logger.warning("Rejected email: %s", error)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:  # pragma: no cover - defensive
        logger.exception("Unexpected error while processing email")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while processing the email",
        ) from error

    logger.info(
        "Processed email | category=%s priority=%s keywords=%s",
        result["category"],
        result["priority"],
        result["keywords"],
    )
    return EmailResponse(**result)


# Mount the router. `prefix=""` is a no-op locally; in the workspace it
# becomes "/email-app" so the proxy and FastAPI agree on the URL shape.
app.include_router(router, prefix=BASE_PATH)


@app.exception_handler(ValidationError)
async def handle_validation_error(_: Request, exc: ValidationError) -> JSONResponse:
    """Return a clean 422 payload when Pydantic validation fails."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


if __name__ == "__main__":
    # Convenience entry point for `python main.py`. In production you
    # would typically run `uvicorn main:app` from a process manager.
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
