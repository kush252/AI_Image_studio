"""Uniform error envelope. See PLAN.md Section 4."""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for every error we deliberately surface to the client."""

    code = "INTERNAL_ERROR"
    status = 500

    def __init__(self, message: str, *, field: str | None = None, detail: Any = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.detail = detail

    def envelope(self) -> dict:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            err["field"] = self.field
        if self.detail is not None:
            err["detail"] = self.detail
        return {"error": err}


class ImageTooLarge(AppError):
    code = "IMAGE_TOO_LARGE"
    status = 413


class UnsupportedFormat(AppError):
    code = "UNSUPPORTED_FORMAT"
    status = 415


class DecodeFailed(AppError):
    code = "DECODE_FAILED"
    status = 400


class ImageNotFound(AppError):
    code = "IMAGE_NOT_FOUND"
    status = 404


class RenderNotFound(AppError):
    code = "RENDER_NOT_FOUND"
    status = 404


class UnknownOperation(AppError):
    code = "UNKNOWN_OPERATION"
    status = 404


class ParamOutOfRange(AppError):
    code = "PARAM_OUT_OF_RANGE"
    status = 422


class ParamInvalid(AppError):
    code = "PARAM_INVALID"
    status = 422


class OpRequiresGrayscale(AppError):
    code = "OP_REQUIRES_GRAYSCALE"
    status = 422


class OpRequiresColor(AppError):
    code = "OP_REQUIRES_COLOR"
    status = 422


class SecondImageRequired(AppError):
    code = "SECOND_IMAGE_REQUIRED"
    status = 422


class Cancelled(AppError):
    code = "CANCELLED"
    status = 409


class JobNotFound(AppError):
    code = "JOB_NOT_FOUND"
    status = 404
