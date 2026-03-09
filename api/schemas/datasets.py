"""Schemas for the datasets API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    """Summary of a prepared dataset."""

    name: str
    source_file: str
    target_column: str
    n_rows: int
    date_range: list[str]
    creation_date: str
    station_column: Optional[str] = None
    stations: list[str] = Field(default_factory=list)


class DatasetDetail(DatasetSummary):
    """Full dataset details including preprocessing config."""

    covariate_columns: list[str] = Field(default_factory=list)
    preprocessing: dict[str, Any] = Field(default_factory=dict)


class DatasetPreview(BaseModel):
    """Preview of dataset rows."""

    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int


class DatasetProfile(BaseModel):
    """Statistical profile of a dataset."""

    columns: dict[str, dict[str, Any]]
    shape: list[int]
    dtypes: dict[str, str]
    missing: dict[str, int]


class ImportDBRequest(BaseModel):
    """Request to import data from the BRGM database."""

    table_name: str
    schema_name: str = "gold"
    columns: list[str]
    date_column: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    filters: Optional[dict[str, Any]] = None
    limit: Optional[int] = None
    dataset_name: Optional[str] = None


class DatasetCreateRequest(BaseModel):
    """Request to create a dataset from uploaded CSV."""

    name: str
    target_column: str
    covariate_columns: list[str] = Field(default_factory=list)
    station_column: Optional[str] = None
    stations: list[str] = Field(default_factory=list)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
