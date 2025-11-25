"""
Kaggle Integration Module

Provides Kaggle competition data fetching, problem parsing, etc.
"""
from .data_fetcher import KaggleDataFetcher, CompetitionInfo
from .submission_validator import SubmissionValidator

__all__ = ["KaggleDataFetcher", "CompetitionInfo", "SubmissionValidator"]

