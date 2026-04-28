"""
============================================================
Violations Log Routes -- /api/violations
============================================================
"""

from fastapi import APIRouter, Query
from typing import Optional

from app.database import get_violations, get_violation_stats, get_history

router = APIRouter()


@router.get("/violations")
async def list_violations(
    student_id: Optional[str] = Query(default=None),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=200),
):
    """
    List violations with optional filters.
    
    Query params:
    - student_id: Filter by student
    - start_date: Filter from date (YYYY-MM-DD)
    - end_date: Filter to date (YYYY-MM-DD)
    - limit: Max results (default 50)
    """
    violations = get_violations(student_id, start_date, end_date, limit)
    return {"violations": violations, "total": len(violations)}


@router.get("/violations/stats")
async def violation_stats():
    """Get violation summary statistics."""
    stats = get_violation_stats()
    return stats


@router.get("/history")
async def analysis_history(limit: int = Query(default=100, le=500)):
    """Get full history of all image analyses performed."""
    history = get_history(limit=limit)
    return {"history": history, "total": len(history)}
