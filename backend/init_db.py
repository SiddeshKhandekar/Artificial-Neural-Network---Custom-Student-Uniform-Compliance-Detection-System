"""
============================================================
Database Initialization & Demo Data Seeder
============================================================
Run this script once to create the database tables and seed
5 demo student records for testing.

Usage:
    cd backend
    python init_db.py
============================================================
"""

import sys
import os
import io

# Fix Windows console encoding for Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import init_database, create_student, get_all_students
from app.config import FACES_DIR


DEMO_STUDENTS = [
    {
        "student_id": "STU001",
        "name": "Aarav Sharma",
        "department": "Computer Science",
        "year": 3,
    },
    {
        "student_id": "STU002",
        "name": "Priya Patel",
        "department": "Information Technology",
        "year": 3,
    },
    {
        "student_id": "STU003",
        "name": "Rahul Verma",
        "department": "Computer Science",
        "year": 3,
    },
    {
        "student_id": "STU004",
        "name": "Sneha Gupta",
        "department": "Electronics",
        "year": 3,
    },
    {
        "student_id": "STU005",
        "name": "Vikram Singh",
        "department": "Mechanical Engineering",
        "year": 3,
    },
]


def seed_database():
    """Initialize tables and insert demo student records."""
    print("=" * 60)
    print("  Student Uniform Compliance System -- DB Initializer")
    print("=" * 60)

    # Step 1: Create tables
    print("\n[1/3] Creating database tables...")
    init_database()
    print("      [OK] Tables created successfully")

    # Step 2: Create student face directories
    print("\n[2/3] Creating face enrollment directories...")
    for student in DEMO_STUDENTS:
        student_dir = FACES_DIR / student["student_id"]
        student_dir.mkdir(parents=True, exist_ok=True)
        print(f"      [OK] Created: {student_dir}")

    # Step 3: Seed demo students
    print("\n[3/3] Seeding demo students...")
    existing = get_all_students()
    existing_ids = {s["student_id"] for s in existing}

    for student in DEMO_STUDENTS:
        if student["student_id"] in existing_ids:
            print(f"      [SKIP] Already exists: {student['student_id']} - {student['name']}")
        else:
            create_student(**student)
            print(f"      [OK] Created: {student['student_id']} - {student['name']}")

    # Summary
    all_students = get_all_students()
    print(f"\n{'=' * 60}")
    print(f"  Database ready! {len(all_students)} students registered.")
    print(f"{'=' * 60}")
    print(f"\n  Next steps:")
    print(f"  1. Add face photos to: {FACES_DIR}/<STUDENT_ID>/")
    print(f"     (3-5 clear face photos per student)")
    print(f"  2. Start the backend: uvicorn app.main:app --reload")
    print(f"  3. Register faces via: POST /api/students/<id>/enroll")
    print()


if __name__ == "__main__":
    seed_database()
