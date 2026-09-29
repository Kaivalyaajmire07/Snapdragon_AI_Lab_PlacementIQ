# ==================================================
# IMPORT REQUIRED LIBRARIES
# ==================================================
import time
import os
import json
import re
import base64
import hashlib
import hmac
import secrets
import tempfile
import uuid
from pathlib import Path
from html import escape as html_escape
from typing import Optional, List

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from google import genai
from google.genai import types

import joblib
import pandas as pd
import psycopg2
import shap
import numpy as np


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# CREATE FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Student Placement Readiness API",
    description=(
        "Student Placement Readiness Prediction System with "
        "Random Forest, K-Means Clustering, SHAP Explainability, "
        "Skill Gap Analysis, Analytics and Gemini AI Assistant"
    )
)


# ==================================================
# CORS CONFIGURATION
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# GEMINI CONFIGURATION - DUAL ASSISTANT
# ==================================================
#
# .env:
# GEMINI_API_KEY_AI=...
# GEMINI_API_KEY_DATA=...
# GEMINI_API_KEY_BACKUP=...
#
# IMPORTANT:
# API-key limits are project-level. Multiple keys from the
# same project share that project's quota. Use only keys from
# projects you are authorized to use.
#
# AI assistant:
#   /ai-chat     -> general career / learning assistant
#
# Data assistant:
#   /data-chat   -> NLQ -> Gemini tool call -> PostgreSQL/ML
#
# /chat remains as a backward-compatible alias for /data-chat.

GEMINI_KEYS = {
    "ai": (
        os.getenv("GEMINI_API_KEY_AI")
        or os.getenv("GEMINI_API_KEY")
    ),
    "data": os.getenv("GEMINI_API_KEY_DATA"),
    "backup": os.getenv("GEMINI_API_KEY_BACKUP"),
}

# Gemini is optional in the Snapdragon build.
# If no Gemini key is configured, /ai-chat and /data-chat can use
# a local Ollama-compatible model when LOCAL_AI_ENABLED=true.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

gemini_clients = {
    name: genai.Client(api_key=key)
    for name, key in GEMINI_KEYS.items()
    if key
}



# ==================================================
# LOCAL SNAPDRAGON AI FALLBACK
# ==================================================
# The final Snapdragon deployment can point this to an on-device
# Ollama-compatible open-source model. No cloud key is required.
LOCAL_AI_ENABLED = os.getenv("LOCAL_AI_ENABLED", "true").lower() == "true"
LOCAL_AI_URL = os.getenv("LOCAL_AI_URL", "http://127.0.0.1:11434/api/chat")
LOCAL_AI_MODEL = os.getenv("LOCAL_AI_MODEL", "qwen2.5:3b")

class _LocalAIResponse:
    def __init__(self, text):
        self.text = text

def _local_ai_response(contents):
    if not LOCAL_AI_ENABLED:
        raise RuntimeError("Local AI is disabled.")

    # Flatten the Gemini-style contents into a prompt.
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        prompt = "\n".join(parts)
    else:
        prompt = str(contents)

    payload = json.dumps({
        "model": LOCAL_AI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are PlacementIQ, a concise student placement and "
                    "career assistant. Use only the information supplied "
                    "in the prompt. Do not invent student facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }).encode("utf-8")

    from urllib.request import Request, urlopen
    request = Request(
        LOCAL_AI_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))

    message = data.get("message", {})
    answer = message.get("content") or data.get("response")
    if not answer:
        raise RuntimeError("Local AI returned no text.")
    return _LocalAIResponse(answer)

def _gemini_retryable(error):
    """Return True when a fallback key/model may help."""
    text = str(error).lower()

    return any(
        token in text
        for token in (
            "429",
            "resource_exhausted",
            "quota",
            "503",
            "unavailable",
            "high demand",
            "temporarily",
        )
    )


def generate_gemini_response(
    assistant_type,
    contents,
    config,
):
    """
    Use the primary key for the selected assistant, then try
    the configured backup keys for retryable quota/capacity errors.

    This does not create extra quota when keys belong to the same
    Google Cloud project.
    """

    if assistant_type == "ai":
        order = ["ai", "backup", "data"]
    else:
        order = ["data", "backup", "ai"]

    # Prefer configured Gemini when available; otherwise use local AI.
    if not gemini_clients and LOCAL_AI_ENABLED:
        return _local_ai_response(contents)

    last_error = None

    for key_name in order:

        client = gemini_clients.get(key_name)

        if not client:
            continue

        for attempt in range(2):

            try:

                print(
                    f"[Gemini] {assistant_type.upper()} "
                    f"using {key_name}, attempt {attempt + 1}"
                )

                return client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config,
                )

            except Exception as error:

                last_error = error

                print(
                    f"[Gemini] {key_name} failed: {error}"
                )

                if not _gemini_retryable(error):
                    raise

                if attempt == 0:
                    time.sleep(1)

    if LOCAL_AI_ENABLED:
        try:
            return _local_ai_response(contents)
        except Exception as local_error:
            raise RuntimeError(
                f"All configured Gemini clients and local AI failed: "
                f"{last_error}; local error: {local_error}"
            )
    raise RuntimeError(
        f"All configured Gemini clients failed: {last_error}"
    )



# ==================================================
# FILE PATHS
# ==================================================

_BASE_DIR = Path(__file__).resolve().parent
_TRAINED_DIR = _BASE_DIR / "trained_models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_TRAINED_DIR / "placement_model_sem6.pkl")))
KMEANS_MODEL_PATH = Path(os.getenv("KMEANS_MODEL_PATH", str(_TRAINED_DIR / "kmeans_model.pkl")))
KMEANS_SCALER_PATH = Path(os.getenv("KMEANS_SCALER_PATH", str(_TRAINED_DIR / "kmeans_scaler.pkl")))
CLUSTER_LABEL_PATH = Path(os.getenv("CLUSTER_LABEL_PATH", str(_TRAINED_DIR / "cluster_labels.pkl")))


# ==================================================
# LOAD RANDOM FOREST MODEL
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Placement model not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


# ==================================================
# LOAD K-MEANS MODEL
# ==================================================

for _artifact in (KMEANS_MODEL_PATH, KMEANS_SCALER_PATH, CLUSTER_LABEL_PATH):
    if not _artifact.exists():
        raise FileNotFoundError(f"K-Means artifact not found: {_artifact}")
kmeans_model = joblib.load(KMEANS_MODEL_PATH)
kmeans_scaler = joblib.load(KMEANS_SCALER_PATH)
cluster_labels = joblib.load(CLUSTER_LABEL_PATH)


# ==================================================
# CREATE SHAP EXPLAINER
# ==================================================

_shap_model = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model
shap_explainer = shap.TreeExplainer(_shap_model)


# ==================================================
# MODEL FEATURES
# ORDER MUST MATCH TRAINING DATA
# ==================================================

FEATURES = [

    "CGPA",
    "Attendance_Percent",
    "Technical_Skill_Score",
    "Aptitude_Score",
    "Communication_Score",
    "Projects_Count",
    "Internships_Count",
    "Certifications_Count",
    "Coding_Assessment_Score",
    "Backlogs"

]


# ==================================================
# POSTGRESQL DATABASE CONNECTION
# ==================================================

def get_db_connection():

    return psycopg2.connect(

        host=os.getenv(
            "DB_HOST",
            "localhost"
        ),

        database=os.getenv(
            "DB_NAME",
            "postgres"
        ),

        user=os.getenv(
            "DB_USER",
            "postgres"
        ),

        password=os.getenv(
            "DB_PASSWORD"
        ),

        port=os.getenv(
            "DB_PORT",
            "5432"
        )

    )


# ==================================================
# PYDANTIC MODELS
# ==================================================

class JobDescriptionRequest(BaseModel):

    job_description: str


class AnalyticsQueryRequest(BaseModel):

    question: str


class ChatRequest(BaseModel):

    message: str



class LoginRequest(BaseModel):
    username: str
    password: str


class FacultyProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    new_password: Optional[str] = None


class RegisterStudentRequest(BaseModel):
    username: str
    password: str
    student_name: str


class StudentProfileUpdate(BaseModel):
    student_name: Optional[str] = None
    contact_email: Optional[str] = None
    phone: Optional[str] = None
    objective: Optional[str] = None

    cgpa: Optional[float] = None
    tenth_percentage: Optional[float] = None
    twelfth_percentage: Optional[float] = None
    sem_1_percentage: Optional[float] = None
    sem_1_sgpa: Optional[float] = None
    sem_2_percentage: Optional[float] = None
    sem_2_sgpa: Optional[float] = None
    sem_3_percentage: Optional[float] = None
    sem_3_sgpa: Optional[float] = None
    sem_4_percentage: Optional[float] = None
    sem_4_sgpa: Optional[float] = None
    sem_5_percentage: Optional[float] = None
    sem_5_sgpa: Optional[float] = None
    sem_6_percentage: Optional[float] = None
    sem_6_sgpa: Optional[float] = None
    sem_7_percentage: Optional[float] = None
    sem_7_sgpa: Optional[float] = None
    sem_8_percentage: Optional[float] = None
    sem_8_sgpa: Optional[float] = None
    attendance_percent: Optional[float] = None
    technical_skill_score: Optional[float] = None
    aptitude_score: Optional[float] = None
    communication_score: Optional[float] = None
    projects_count: Optional[int] = None
    internships_count: Optional[int] = None
    certifications_count: Optional[int] = None
    coding_assessment_score: Optional[float] = None
    backlogs: Optional[int] = None

    # Skills used by local Skill Gap Analysis.
    skills: Optional[List[str]] = None
    soft_skills: Optional[List[str]] = None


class ProjectRequest(BaseModel):
    title: str
    description: str
    technologies: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class InternshipRequest(BaseModel):
    company_name: str
    role: str
    description: str
    technologies: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CertificationRequest(BaseModel):
    name: str
    issuer: str = ""
    issue_date: str = ""
    credential_url: str = ""


class AchievementRequest(BaseModel):
    title: str
    description: str = ""
    achievement_date: str = ""


class ResumeGenerateRequest(BaseModel):
    target_role: str
    job_description: str = ""


class ObjectiveGenerateRequest(BaseModel):
    target_role: str = ""


class PlacementIntelligenceRequest(BaseModel):
    target_role: str = ""
    job_description: str = ""


class ResumeExportRequest(BaseModel):
    """Export an already-generated resume without another Gemini call."""
    target_role: str
    resume: dict


# ==================================================
# AUTHENTICATION
# ==================================================

AUTH_SECRET = os.getenv(
    "AUTH_SECRET",
    "change-this-auth-secret",
)

AUTH_TOKEN_TTL_SECONDS = int(
    os.getenv("AUTH_TOKEN_TTL_SECONDS", "28800")
)


def _hash_password(password: str) -> str:

    salt = secrets.token_bytes(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120000,
    )

    return (
        base64.urlsafe_b64encode(salt).decode("ascii")
        + "$"
        + base64.urlsafe_b64encode(digest).decode("ascii")
    )


def _verify_password(password: str, stored_hash: str) -> bool:

    try:
        salt_b64, digest_b64 = stored_hash.split("$", 1)

        salt = base64.urlsafe_b64decode(
            salt_b64.encode("ascii")
        )

        expected = base64.urlsafe_b64decode(
            digest_b64.encode("ascii")
        )

        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            120000,
        )

        return hmac.compare_digest(
            actual,
            expected,
        )

    except Exception:
        return False


def _create_auth_token(user: dict) -> str:

    payload = {
        "user_id": int(user["user_id"]),
        "username": user["username"],
        "role": user["role"],
        "student_id": user.get("student_id"),
        "exp": int(time.time()) + AUTH_TOKEN_TTL_SECONDS,
    }

    raw = json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")

    encoded = (
        base64.urlsafe_b64encode(raw)
        .decode("ascii")
        .rstrip("=")
    )

    signature = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        encoded.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()

    return f"{encoded}.{signature}"


def _read_auth_token(token: str) -> dict:

    try:
        encoded, signature = token.split(".", 1)

        expected = hmac.new(
            AUTH_SECRET.encode("utf-8"),
            encoded.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(
            signature,
            expected,
        ):
            raise ValueError("Invalid signature.")

        padded = encoded + "=" * (
            (-len(encoded)) % 4
        )

        payload = json.loads(
            base64.urlsafe_b64decode(
                padded.encode("ascii")
            ).decode("utf-8")
        )

        if int(payload["exp"]) < int(time.time()):
            raise ValueError("Token expired.")

        return payload

    except Exception as error:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid or expired token: {error}",
        )


def _split_technology_text(value: str) -> list[str]:

    if not value:
        return []

    parts = re.split(r"[,;\\n|]+", value)

    cleaned = []
    seen = set()

    for part in parts:
        skill = part.strip()

        if not skill:
            continue

        key = skill.lower()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(skill)

    return cleaned


def _ensure_student_detail_tables(connection, cursor):

    # Add new profile columns without deleting existing data.
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS contact_email TEXT")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS phone VARCHAR(40)")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS objective TEXT")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS soft_skills TEXT NOT NULL DEFAULT ''")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS profile_image_path TEXT")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS sem_7_percentage DOUBLE PRECISION")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS sem_7_sgpa DOUBLE PRECISION")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS sem_8_percentage DOUBLE PRECISION")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS sem_8_sgpa DOUBLE PRECISION")


    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS student_projects (
            project_id SERIAL PRIMARY KEY,
            student_id INTEGER NOT NULL,
            title VARCHAR(200) NOT NULL,
            description TEXT NOT NULL,
            technologies TEXT NOT NULL,
            start_date VARCHAR(30),
            end_date VARCHAR(30),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS student_internships (
            internship_id SERIAL PRIMARY KEY,
            student_id INTEGER NOT NULL,
            company_name VARCHAR(200) NOT NULL,
            role VARCHAR(200) NOT NULL,
            description TEXT NOT NULL,
            technologies TEXT NOT NULL,
            start_date VARCHAR(30),
            end_date VARCHAR(30),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Safely upgrade either portfolio schema without referencing columns that may not exist.
    cursor.execute("ALTER TABLE student_projects ADD COLUMN IF NOT EXISTS title VARCHAR(200)")
    cursor.execute("ALTER TABLE student_projects ADD COLUMN IF NOT EXISTS description TEXT")
    cursor.execute("ALTER TABLE student_projects ADD COLUMN IF NOT EXISTS technologies TEXT")
    cursor.execute("ALTER TABLE student_projects ADD COLUMN IF NOT EXISTS start_date VARCHAR(30)")
    cursor.execute("ALTER TABLE student_projects ADD COLUMN IF NOT EXISTS end_date VARCHAR(30)")
    cursor.execute("ALTER TABLE student_projects ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='student_projects'")
    project_columns = {row[0] for row in cursor.fetchall()}
    if 'project_title' in project_columns:
        cursor.execute("UPDATE student_projects SET title=COALESCE(NULLIF(title,''),project_title) WHERE title IS NULL OR title=''")
        cursor.execute("ALTER TABLE student_projects ALTER COLUMN project_title DROP NOT NULL")
    if 'project_description' in project_columns:
        cursor.execute("UPDATE student_projects SET description=COALESCE(NULLIF(description,''),project_description) WHERE description IS NULL OR description=''")
        cursor.execute("ALTER TABLE student_projects ALTER COLUMN project_description DROP NOT NULL")
    if 'key_technologies' in project_columns:
        cursor.execute("UPDATE student_projects SET technologies=COALESCE(NULLIF(technologies,''),key_technologies,'') WHERE technologies IS NULL OR technologies=''")
        cursor.execute("ALTER TABLE student_projects ALTER COLUMN key_technologies DROP NOT NULL")

    cursor.execute("ALTER TABLE student_internships ADD COLUMN IF NOT EXISTS role VARCHAR(200)")
    cursor.execute("ALTER TABLE student_internships ADD COLUMN IF NOT EXISTS description TEXT")
    cursor.execute("ALTER TABLE student_internships ADD COLUMN IF NOT EXISTS technologies TEXT")
    cursor.execute("ALTER TABLE student_internships ADD COLUMN IF NOT EXISTS start_date VARCHAR(30)")
    cursor.execute("ALTER TABLE student_internships ADD COLUMN IF NOT EXISTS end_date VARCHAR(30)")
    cursor.execute("ALTER TABLE student_internships ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='student_internships'")
    internship_columns = {row[0] for row in cursor.fetchall()}
    if 'role_title' in internship_columns:
        cursor.execute("UPDATE student_internships SET role=COALESCE(NULLIF(role,''),role_title) WHERE role IS NULL OR role=''")
        cursor.execute("ALTER TABLE student_internships ALTER COLUMN role_title DROP NOT NULL")
    if 'internship_description' in internship_columns:
        cursor.execute("UPDATE student_internships SET description=COALESCE(NULLIF(description,''),internship_description) WHERE description IS NULL OR description=''")
        cursor.execute("ALTER TABLE student_internships ALTER COLUMN internship_description DROP NOT NULL")
    if 'key_technologies' in internship_columns:
        cursor.execute("UPDATE student_internships SET technologies=COALESCE(NULLIF(technologies,''),key_technologies,'') WHERE technologies IS NULL OR technologies=''")
        cursor.execute("ALTER TABLE student_internships ALTER COLUMN key_technologies DROP NOT NULL")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS student_manual_skills (
            student_id INTEGER NOT NULL,
            skill_name VARCHAR(150) NOT NULL,
            PRIMARY KEY (student_id, skill_name)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS student_skills (
            student_id INTEGER NOT NULL,
            skill_name VARCHAR(150) NOT NULL,
            PRIMARY KEY (student_id, skill_name)
        )
        """
    )

    # New portfolio collections. No 10-item hard limit is imposed.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS student_certifications (
            certification_id SERIAL PRIMARY KEY,
            student_id INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
            name VARCHAR(250) NOT NULL,
            issuer VARCHAR(250) NOT NULL DEFAULT '',
            issue_date VARCHAR(50) NOT NULL DEFAULT '',
            credential_url TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS student_achievements (
            achievement_id SERIAL PRIMARY KEY,
            student_id INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
            title VARCHAR(250) NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            achievement_date VARCHAR(50) NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()


def _refresh_student_skill_index(student_id, cursor):
    """Rebuild the local skill index from every portfolio/profile source."""
    source_texts = []

    cursor.execute("SELECT skill_name FROM student_manual_skills WHERE student_id = %s", (student_id,))
    source_texts.extend([row[0] for row in cursor.fetchall() if row[0]])

    cursor.execute("SELECT title, description, technologies FROM student_projects WHERE student_id = %s", (student_id,))
    for row in cursor.fetchall():
        source_texts.extend([value for value in row if value])

    cursor.execute("SELECT company_name, role, description, technologies FROM student_internships WHERE student_id = %s", (student_id,))
    for row in cursor.fetchall():
        source_texts.extend([value for value in row if value])

    cursor.execute("SELECT name, issuer FROM student_certifications WHERE student_id = %s", (student_id,))
    for row in cursor.fetchall():
        source_texts.extend([value for value in row if value])

    cursor.execute("SELECT title, description FROM student_achievements WHERE student_id = %s", (student_id,))
    for row in cursor.fetchall():
        source_texts.extend([value for value in row if value])

    detected, seen = [], set()
    for text in source_texts:
        for skill in extract_skills(str(text)):
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                detected.append(skill)

    cursor.execute("DELETE FROM student_skills WHERE student_id = %s", (student_id,))
    for skill in detected:
        cursor.execute(
            "INSERT INTO student_skills (student_id, skill_name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (student_id, skill),
        )


def _ensure_auth_table():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_users (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role VARCHAR(20) NOT NULL
                    CHECK (role IN ('faculty', 'student')),
                student_id INTEGER NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS display_name VARCHAR(150)")
        cursor.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS email VARCHAR(200)")

        faculty_username = os.getenv(
            "FACULTY_USERNAME",
            "faculty",
        )
        faculty_password = os.getenv(
            "FACULTY_PASSWORD",
            "faculty123",
        )

        student_username = os.getenv(
            "STUDENT_USERNAME",
            "student1",
        )
        student_password = os.getenv(
            "STUDENT_PASSWORD",
            "student123",
        )

        try:
            student_id = int(
                os.getenv("STUDENT_ID", "1")
            )
        except ValueError:
            student_id = 1

        cursor.execute(
            """
            INSERT INTO auth_users
                (username, password_hash, role)
            VALUES (%s, %s, 'faculty')
            ON CONFLICT (username) DO NOTHING
            """,
            (
                faculty_username,
                _hash_password(faculty_password),
            ),
        )

        cursor.execute(
            """
            INSERT INTO auth_users
                (username, password_hash, role, student_id)
            VALUES (%s, %s, 'student', %s)
            ON CONFLICT (username) DO NOTHING
            """,
            (
                student_username,
                _hash_password(student_password),
                student_id,
            ),
        )

        _ensure_student_detail_tables(
            connection,
            cursor,
        )

        connection.commit()

    except Exception:
        if connection:
            connection.rollback()
        raise

    finally:
        close_db(cursor, connection)


def get_current_user(
    authorization: Optional[str] = Header(default=None),
):

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Use Bearer authentication.",
        )

    return _read_auth_token(token)


def require_faculty(
    user: dict = Depends(get_current_user),
):

    if user.get("role") != "faculty":
        raise HTTPException(
            status_code=403,
            detail="Faculty access required.",
        )

    return user


def require_student(
    user: dict = Depends(get_current_user),
):

    if user.get("role") != "student":
        raise HTTPException(
            status_code=403,
            detail="Student access required.",
        )

    if user.get("student_id") is None:
        raise HTTPException(
            status_code=403,
            detail="This account is not linked to a student profile.",
        )

    return user


def require_student_or_faculty(
    student_id: int,
    user: dict = Depends(get_current_user),
):

    if user.get("role") == "faculty":
        return user

    if user.get("role") == "student" and int(user.get("student_id")) == int(student_id):
        return user

    raise HTTPException(
        status_code=403,
        detail="You can only access your own student profile.",
    )


# ==================================================
# LOCAL SKILL DICTIONARY / NORMALIZATION
# ==================================================

SKILL_ALIASES = {
    "Python": ["python", "python3", "python programming"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c sharp"],
    "R": ["r programming", "r language"],
    "Go": ["golang", "go language"],

    "SQL": ["sql", "structured query language"],
    "PostgreSQL": ["postgresql", "postgres", "psql"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo db"],
    "Redis": ["redis"],
    "Firebase": ["firebase"],
    "Supabase": ["supabase"],
    "SQLite": ["sqlite"],
    "NoSQL": ["nosql", "no sql"],
    "Database Design": ["database design", "database modeling", "database modelling"],
    "Data Warehousing": ["data warehouse", "data warehousing"],

    "React": ["react", "reactjs", "react.js"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "REST API": ["rest api", "restful api", "rest services", "rest apis"],
    "GraphQL": ["graphql"],
    "API Development": ["api development", "backend api development"],
    "HTML": ["html"],
    "CSS": ["css"],
    "Tailwind CSS": ["tailwind", "tailwind css"],
    "JWT": ["jwt", "json web token", "json web tokens"],
    "OAuth": ["oauth", "oauth2"],

    "Git": ["git"],
    "GitHub": ["github"],
    "GitLab": ["gitlab"],
    "Docker": ["docker", "containerization", "containerisation"],
    "Kubernetes": ["kubernetes", "k8s"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "Linux": ["linux", "ubuntu"],
    "MLOps": ["mlops", "machine learning operations"],
    "Postman": ["postman"],

    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "EC2": ["ec2", "amazon ec2"],
    "S3": ["s3", "amazon s3"],
    "AWS Lambda": ["aws lambda", "lambda function", "lambda functions"],
    "Cloud Computing": ["cloud computing", "cloud platforms", "cloud technology"],

    "Data Analysis": ["data analysis", "data analytics", "data analyst"],
    "Data Science": ["data science", "data scientist"],
    "Data Cleaning": ["data cleaning", "data cleansing"],
    "Data Preprocessing": ["data preprocessing", "data pre-processing", "data preparation"],
    "Exploratory Data Analysis": ["exploratory data analysis", "eda"],
    "Data Visualization": ["data visualization", "data visualisation", "data viz"],
    "Statistics": ["statistics", "statistical analysis", "statistical modeling", "statistical modelling"],
    "Probability": ["probability", "probability theory"],
    "Business Intelligence": ["business intelligence", "business intelligence tools"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "microsoft excel"],
    "Power Query": ["power query"],
    "ETL": ["etl", "extract transform load", "extract-transform-load"],
    "Data Engineering": ["data engineering", "data engineer"],
    "Big Data": ["big data"],
    "Apache Spark": ["apache spark", "spark"],
    "Apache Airflow": ["apache airflow", "airflow"],
    "dbt": ["dbt"],
    "Snowflake": ["snowflake"],
    "Databricks": ["databricks"],

    "Machine Learning": ["machine learning", "machine-learning", "ml"],
    "Deep Learning": ["deep learning", "deep-learning", "dl"],
    "Artificial Intelligence": ["artificial intelligence", "ai"],
    "Supervised Learning": ["supervised learning"],
    "Unsupervised Learning": ["unsupervised learning"],
    "Regression": ["regression", "regression analysis"],
    "Classification": ["classification", "classification models"],
    "Clustering": ["clustering", "cluster analysis"],
    "Time Series": ["time series", "time-series"],
    "Forecasting": ["forecasting", "predictive forecasting"],
    "Feature Engineering": ["feature engineering", "feature extraction"],
    "Model Evaluation": ["model evaluation", "model validation", "model assessment"],
    "Model Deployment": ["model deployment", "ml model deployment", "machine learning deployment"],
    "Hyperparameter Tuning": ["hyperparameter tuning", "hyperparameter optimization"],
    "Cross Validation": ["cross validation", "cross-validation"],
    "Random Forest": ["random forest", "randomforest"],
    "XGBoost": ["xgboost"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy", "num py"],
    "SciPy": ["scipy"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Keras": ["keras"],
    "OpenCV": ["opencv", "open cv"],
    "SHAP": ["shap", "shap explainability", "explainable ai", "xai"],

    "Natural Language Processing": ["natural language processing", "nlp"],
    "Computer Vision": ["computer vision"],
    "Generative AI": ["generative ai", "genai", "generative artificial intelligence"],
    "LLM": ["llm", "large language model", "large language models"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "LangChain": ["langchain"],
    "Hugging Face": ["hugging face", "huggingface"],
    "Transformers": ["transformers", "transformer models"],
    "BERT": ["bert"],
    "Prompt Engineering": ["prompt engineering", "prompt design"],
    "OpenAI": ["openai", "openai api"],
    "Gemini": ["gemini", "gemini api"],

    "Jupyter": ["jupyter", "jupyter notebook", "jupyter notebooks"],
    "Streamlit": ["streamlit"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "Plotly": ["plotly"],
    "BeautifulSoup": ["beautifulsoup", "beautiful soup"],
    "Selenium": ["selenium"],

    "Problem Solving": ["problem solving", "problem-solving"],
    "Communication": ["communication", "communication skills"],
    "Teamwork": ["teamwork", "team collaboration", "collaboration"],
    "Agile": ["agile", "agile methodology", "scrum"],
    "Testing": ["software testing", "unit testing", "integration testing"],
    "Documentation": ["technical documentation", "documentation"],
}


KNOWN_SKILLS = list(SKILL_ALIASES.keys())


def _normalize_text_for_skill_matching(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^a-z0-9+#.\- ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _contains_skill(text: str, alias: str) -> bool:
    alias_norm = _normalize_text_for_skill_matching(alias)
    if not alias_norm:
        return False
    escaped = re.escape(alias_norm).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9+#]){escaped}(?![a-z0-9+#])", text) is not None


def extract_skills(text: str) -> list[str]:
    normalized = _normalize_text_for_skill_matching(text or "")
    detected = []

    for canonical, aliases in SKILL_ALIASES.items():
        if any(_contains_skill(normalized, alias) for alias in aliases):
            detected.append(canonical)

    return detected


def _collect_student_skill_evidence(student_id: int, cursor) -> dict:
    """Build an explainable skill profile from all student sources."""

    evidence = {}

    def add_skills(text: str, source: str):
        for skill in extract_skills(text or ""):
            evidence.setdefault(skill, set()).add(source)

    cursor.execute(
        """
        SELECT skill_name
        FROM student_manual_skills
        WHERE student_id = %s
        ORDER BY skill_name
        """,
        (student_id,),
    )

    for row in cursor.fetchall():
        if row[0]:
            add_skills(row[0], "Profile skills")

    cursor.execute(
        """
        SELECT title, description, technologies
        FROM student_projects
        WHERE student_id = %s
        ORDER BY created_at DESC, project_id DESC
        """,
        (student_id,),
    )

    for title, description, technologies in cursor.fetchall():
        add_skills(title, "Project title")
        add_skills(description, "Project description")
        add_skills(technologies, "Project technologies")

    cursor.execute(
        """
        SELECT company_name, role, description, technologies
        FROM student_internships
        WHERE student_id = %s
        ORDER BY created_at DESC, internship_id DESC
        """,
        (student_id,),
    )

    for company, role, description, technologies in cursor.fetchall():
        add_skills(company, "Internship company")
        add_skills(role, "Internship role")
        add_skills(description, "Internship description")
        add_skills(technologies, "Internship technologies")

    cursor.execute(
        """
        SELECT skill_name
        FROM student_skills
        WHERE student_id = %s
        ORDER BY skill_name
        """,
        (student_id,),
    )

    for row in cursor.fetchall():
        if row[0]:
            add_skills(row[0], "Skill index")

    try:
        cursor.execute("SELECT skills, project_description, internship_description, programming_language_interest_1, programming_language_interest_2, area_of_interest FROM students WHERE student_id=%s", (student_id,))
        live = cursor.fetchone()
        if live:
            add_skills(live[0], "Live profile skills")
            add_skills(live[1], "Live project description")
            add_skills(live[2], "Live internship description")
            add_skills(live[3], "Programming language interest")
            add_skills(live[4], "Programming language interest")
            add_skills(live[5], "Area of interest")
    except Exception:
        pass

    return evidence


# ==================================================
# HELPER
# CLOSE DATABASE RESOURCES
# ==================================================

def close_db(cursor, connection):

    if cursor:

        cursor.close()

    if connection:

        connection.close()



# ==================================================
# STUDENT REGISTRATION
# ==================================================

@app.post("/auth/register/student")
def register_student(request: RegisterStudentRequest):

    username = request.username.strip()
    student_name = request.student_name.strip()

    if len(username) < 3:
        raise HTTPException(
            status_code=400,
            detail="Username must contain at least 3 characters.",
        )

    if len(request.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters.",
        )

    if not student_name:
        raise HTTPException(
            status_code=400,
            detail="Student name is required.",
        )

    try:
        _ensure_auth_table()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Authentication setup error: {error}",
        )

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        _ensure_student_detail_tables(connection, cursor)

        cursor.execute(
            "SELECT 1 FROM auth_users WHERE username = %s",
            (username,),
        )
        if cursor.fetchone():
            raise HTTPException(
                status_code=409,
                detail="Username already exists.",
            )

        # Registration links authentication to an existing live-dataset student.
        # It never creates a blank/fake student record.
        live_student = _find_live_student(student_name, request.password.strip())
        if not live_student:
            raise HTTPException(
                status_code=404,
                detail="Student not found in the live dataset. Use your exact Student Name and Roll Number.",
            )

        student_id = int(live_student[0])

        cursor.execute(
            """
            INSERT INTO auth_users
                (username, password_hash, role, student_id)
            VALUES (%s, %s, 'student', %s)
            """,
            (
                username,
                _hash_password(request.password),
                student_id,
            ),
        )

        connection.commit()

        cursor.execute(
            """
            SELECT user_id, username, role, student_id
            FROM auth_users
            WHERE username = %s
            """,
            (username,),
        )

        user_row = cursor.fetchone()
        user = {
            "user_id": int(user_row[0]),
            "username": user_row[1],
            "role": user_row[2],
            "student_id": int(user_row[3]),
        }

        return {
            "access_token": _create_auth_token(user),
            "token_type": "bearer",
            "user": user,
            "message": "Student account created successfully.",
        }

    except HTTPException:
        if connection:
            connection.rollback()
        raise

    except Exception as error:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Student registration error: {error}",
        )

    finally:
        close_db(cursor, connection)


# ==================================================
# LOGIN
# ==================================================

@app.post("/auth/login")
def auth_login(request: LoginRequest):

    try:
        _ensure_auth_table()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Authentication setup error: {error}",
        )

    connection = None
    cursor = None

    try:
        # Accept live-dataset student login: full Student_Name + Roll_Number.
        live_student = _find_live_student(request.username.strip(), request.password.strip())
        if live_student:
            student_id = int(live_student[0])
            student_name = live_student[1]
            roll_number = live_student[2]
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT user_id FROM auth_users WHERE student_id = %s LIMIT 1", (student_id,))
            existing = cursor.fetchone()
            if existing:
                user_id = int(existing[0])
            else:
                cursor.execute("INSERT INTO auth_users (username, password_hash, role, student_id) VALUES (%s, %s, 'student', %s) RETURNING user_id", (student_name, _hash_password(roll_number), student_id))
                user_id = int(cursor.fetchone()[0])
            connection.commit()
            user = {"user_id": user_id, "username": student_name, "role": "student", "student_id": student_id}
            return {"access_token": _create_auth_token(user), "token_type": "bearer", "user": user}

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                user_id,
                username,
                password_hash,
                role,
                student_id
            FROM auth_users
            WHERE username = %s
            """,
            (request.username.strip(),),
        )

        row = cursor.fetchone()

        if row is None or not _verify_password(
            request.password,
            row[2],
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password.",
            )

        user = {
            "user_id": int(row[0]),
            "username": row[1],
            "role": row[3],
            "student_id": (
                int(row[4])
                if row[4] is not None
                else None
            ),
        }

        return {
            "access_token": _create_auth_token(user),
            "token_type": "bearer",
            "user": user,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Login error: {error}",
        )

    finally:
        close_db(cursor, connection)


@app.get("/auth/me")
def auth_me(
    user: dict = Depends(get_current_user),
):
    return {"user": user}


@app.put("/faculty/me/profile")
def update_faculty_profile(request: FacultyProfileUpdate, user: dict = Depends(require_faculty)):
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor()
        updates = []
        values = []
        if request.display_name is not None:
            name = request.display_name.strip()
            if not name: raise HTTPException(status_code=400, detail="Display name cannot be empty.")
            updates.append("display_name=%s"); values.append(name)
        if request.email is not None:
            updates.append("email=%s"); values.append(request.email.strip())
        if request.new_password is not None:
            if len(request.new_password) < 6: raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
            updates.append("password_hash=%s"); values.append(_hash_password(request.new_password))
        if updates:
            values.append(int(user["user_id"]))
            cur.execute(f"UPDATE auth_users SET {', '.join(updates)} WHERE user_id=%s", tuple(values))
        cur.execute("SELECT user_id, username, role, student_id, display_name, email FROM auth_users WHERE user_id=%s", (int(user["user_id"]),))
        row = cur.fetchone(); conn.commit()
        updated = {"user_id": int(row[0]), "username": row[1], "role": row[2], "student_id": row[3], "display_name": row[4] or row[1], "email": row[5] or ""}
        return {"user": updated}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Faculty profile update error: {error}")
    finally:
        close_db(cur, conn)


# ==================================================
# STUDENT PORTFOLIO TABLES
# ==================================================

def _ensure_portfolio_tables():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS student_projects (
                project_id SERIAL PRIMARY KEY,
                student_id INTEGER NOT NULL
                    REFERENCES students(student_id)
                    ON DELETE CASCADE,
                project_title VARCHAR(200) NOT NULL,
                project_description TEXT NOT NULL,
                key_technologies TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS student_internships (
                internship_id SERIAL PRIMARY KEY,
                student_id INTEGER NOT NULL
                    REFERENCES students(student_id)
                    ON DELETE CASCADE,
                company_name VARCHAR(200) NOT NULL,
                role_title VARCHAR(200) NOT NULL,
                internship_description TEXT NOT NULL,
                key_technologies TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.commit()

    except Exception:

        if connection:
            connection.rollback()

        raise

    finally:

        close_db(
            cursor,
            connection,
        )



# ==================================================
# HOME API
# ==================================================

# ==================================================
# GET MY PORTFOLIO SKILLS
# ==================================================

@app.get("/my/portfolio-skills")
def get_my_portfolio_skills(
    user: dict = Depends(get_current_user),
):

    if user.get("role") != "student":
        raise HTTPException(
            status_code=403,
            detail="Student access required.",
        )

    _ensure_portfolio_tables()

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT key_technologies
            FROM student_projects
            WHERE student_id = %s

            UNION ALL

            SELECT key_technologies
            FROM student_internships
            WHERE student_id = %s
            """,
            (
                int(user["student_id"]),
                int(user["student_id"]),
            ),
        )

        values = []
        for row in cursor.fetchall():
            if row[0]:
                values.extend(
                    [
                        item.strip()
                        for item in row[0].split(",")
                        if item.strip()
                    ]
                )

        unique = []
        seen = set()

        for value in values:
            key = value.lower()
            if key not in seen:
                seen.add(key)
                unique.append(value)

        return {
            "skills": unique
        }

    finally:

        close_db(cursor, connection)



# ==================================================
# STUDENT PROJECTS
# ==================================================

@app.get("/my/projects")
def get_my_projects(
    user: dict = Depends(get_current_user),
):

    if user.get("role") != "student":
        raise HTTPException(
            status_code=403,
            detail="Student access required.",
        )

    _ensure_portfolio_tables()

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                project_id,
                project_title,
                project_description,
                key_technologies
            FROM student_projects
            WHERE student_id = %s
            ORDER BY created_at DESC, project_id DESC
            """,
            (int(user["student_id"]),),
        )

        rows = cursor.fetchall()

        return {
            "projects": [
                {
                    "project_id": int(row[0]),
                    "project_title": row[1],
                    "project_description": row[2],
                    "key_technologies": row[3],
                }
                for row in rows
            ]
        }

    finally:

        close_db(cursor, connection)


@app.post("/my/projects")
def create_my_project(
    request: ProjectRequest,
    user: dict = Depends(get_current_user),
):

    if user.get("role") != "student":
        raise HTTPException(
            status_code=403,
            detail="Student access required.",
        )

    if not request.title.strip():
        raise HTTPException(
            status_code=400,
            detail="Project title is required.",
        )

    if not request.description.strip():
        raise HTTPException(
            status_code=400,
            detail="Project description is required.",
        )

    _ensure_portfolio_tables()

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO student_projects
                (
                    student_id,
                    project_title,
                    project_description,
                    key_technologies
                )
            VALUES (%s, %s, %s, %s)
            RETURNING
                project_id,
                project_title,
                project_description,
                key_technologies
            """,
            (
                int(user["student_id"]),
                request.title.strip(),
                request.description.strip(),
                request.technologies.strip(),
            ),
        )

        row = cursor.fetchone()
        connection.commit()

        return {
            "project": {
                "project_id": int(row[0]),
                "project_title": row[1],
                "project_description": row[2],
                "key_technologies": row[3],
            }
        }

    except HTTPException:
        raise

    except Exception as error:

        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Project creation error: {error}",
        )

    finally:

        close_db(cursor, connection)


@app.delete("/my/projects/{project_id}")
def delete_my_project(
    project_id: int,
    user: dict = Depends(get_current_user),
):

    if user.get("role") != "student":
        raise HTTPException(
            status_code=403,
            detail="Student access required.",
        )

    _ensure_portfolio_tables()

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM student_projects
            WHERE project_id = %s
              AND student_id = %s
            RETURNING project_id
            """,
            (
                project_id,
                int(user["student_id"]),
            ),
        )

        row = cursor.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Project not found.",
            )

        connection.commit()

        return {
            "message": "Project deleted successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:

        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Project deletion error: {error}",
        )

    finally:

        close_db(cursor, connection)


# ==================================================
# STUDENT INTERNSHIPS
# ==================================================

@app.get("/my/internships")
def get_my_internships(
    user: dict = Depends(get_current_user),
):

    if user.get("role") != "student":
        raise HTTPException(
            status_code=403,
            detail="Student access required.",
        )

    _ensure_portfolio_tables()

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                internship_id,
                company_name,
                role_title,
                internship_description,
                key_technologies
            FROM student_internships
            WHERE student_id = %s
            ORDER BY created_at DESC, internship_id DESC
            """,
            (int(user["student_id"]),),
        )

        rows = cursor.fetchall()

        return {
            "internships": [
                {
                    "internship_id": int(row[0]),
                    "company_name": row[1],
                    "role_title": row[2],
                    "internship_description": row[3],
                    "key_technologies": row[4],
                }
                for row in rows
            ]
        }

    finally:

        close_db(cursor, connection)


@app.post("/my/internships")
def create_my_internship(
    request: InternshipRequest,
    user: dict = Depends(get_current_user),
):

    if user.get("role") != "student":
        raise HTTPException(
            status_code=403,
            detail="Student access required.",
        )

    required_values = [
        request.company_name,
        request.role,
        request.description,
    ]

    if any(
        not value.strip()
        for value in required_values
    ):
        raise HTTPException(
            status_code=400,
            detail="Company, role, and internship description are required.",
        )

    _ensure_portfolio_tables()

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO student_internships
                (
                    student_id,
                    company_name,
                    role_title,
                    internship_description,
                    key_technologies
                )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING
                internship_id,
                company_name,
                role_title,
                internship_description,
                key_technologies
            """,
            (
                int(user["student_id"]),
                request.company_name.strip(),
                request.role.strip(),
                request.description.strip(),
                request.technologies.strip(),
            ),
        )

        row = cursor.fetchone()
        connection.commit()

        return {
            "internship": {
                "internship_id": int(row[0]),
                "company_name": row[1],
                "role_title": row[2],
                "internship_description": row[3],
                "key_technologies": row[4],
            }
        }

    except Exception as error:

        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Internship creation error: {error}",
        )

    finally:

        close_db(cursor, connection)


@app.delete("/my/internships/{internship_id}")
def delete_my_internship(
    internship_id: int,
    user: dict = Depends(get_current_user),
):

    if user.get("role") != "student":
        raise HTTPException(
            status_code=403,
            detail="Student access required.",
        )

    _ensure_portfolio_tables()

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM student_internships
            WHERE internship_id = %s
              AND student_id = %s
            RETURNING internship_id
            """,
            (
                internship_id,
                int(user["student_id"]),
            ),
        )

        row = cursor.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Internship not found.",
            )

        connection.commit()

        return {
            "message": "Internship deleted successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:

        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Internship deletion error: {error}",
        )

    finally:

        close_db(cursor, connection)



@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "student-placement-api",
    }


@app.get("/")
def home():

    return {

        "message":
        "Student Placement Readiness API is running"

    }


# ==================================================
# GET ALL STUDENTS
# ==================================================

@app.get("/students", dependencies=[Depends(require_faculty)])
def get_students():
    return _live_list_students()


    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                student_id,
                student_name,
                cgpa,
                attendance_percent,
                technical_skill_score,
                aptitude_score,
                communication_score,
                projects_count,
                internships_count,
                certifications_count,
                coding_assessment_score,
                backlogs
            FROM students
            ORDER BY student_id
            """
        )

        rows = cursor.fetchall()

        students = []

        for row in rows:

            students.append([

                row[0],
                row[1],
                _db_float(row[2]),
                _db_float(row[3]),
                _db_float(row[4]),
                _db_float(row[5]),
                _db_float(row[6]),
                _db_int(row[7]),
                _db_int(row[8]),
                _db_int(row[9]),
                _db_float(row[10]),
                _db_int(row[11])

            ])

        return {

            "students": students

        }

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=f"Database error: {str(error)}"

        )

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# GET SINGLE STUDENT
# ==================================================

@app.get("/students/{student_id}", dependencies=[Depends(require_student_or_faculty)])
def get_student(student_id: int):
    return _live_get_student_profile(student_id)


    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                student_id,
                student_name,
                cgpa,
                attendance_percent,
                technical_skill_score,
                aptitude_score,
                communication_score,
                projects_count,
                internships_count,
                certifications_count,
                coding_assessment_score,
                backlogs
            FROM students
            WHERE student_id = %s
            """,
            (student_id,)
        )

        row = cursor.fetchone()

        if row is None:

            raise HTTPException(

                status_code=404,

                detail="Student not found"

            )

        # Apply the extended-schema migration before reading the new columns.
        _ensure_student_detail_tables(
            connection,
            cursor,
        )

        cursor.execute(
            """
            SELECT contact_email, phone, objective, soft_skills, profile_image_path
            FROM students
            WHERE student_id = %s
            """,
            (student_id,),
        )
        profile_meta = cursor.fetchone() or ("", "", "", "", None)

        cursor.execute(
            """
            SELECT
                project_id,
                title,
                description,
                technologies,
                start_date,
                end_date
            FROM student_projects
            WHERE student_id = %s
            ORDER BY created_at DESC, project_id DESC
            """,
            (student_id,),
        )
        project_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT
                internship_id,
                company_name,
                role,
                description,
                technologies,
                start_date,
                end_date
            FROM student_internships
            WHERE student_id = %s
            ORDER BY created_at DESC, internship_id DESC
            """,
            (student_id,),
        )
        internship_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT skill_name
            FROM student_manual_skills
            WHERE student_id = %s
            ORDER BY skill_name
            """,
            (student_id,),
        )
        manual_skill_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT certification_id, name, issuer, issue_date, credential_url
            FROM student_certifications
            WHERE student_id = %s
            ORDER BY created_at DESC, certification_id DESC
            """,
            (student_id,),
        )
        certification_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT achievement_id, title, description, achievement_date
            FROM student_achievements
            WHERE student_id = %s
            ORDER BY created_at DESC, achievement_id DESC
            """,
            (student_id,),
        )
        achievement_rows = cursor.fetchall()

        projects = [
            {
                "project_id": int(item[0]),
                "title": item[1],
                "description": item[2],
                "technologies": item[3],
                "start_date": item[4],
                "end_date": item[5],
            }
            for item in project_rows
        ]

        internships = [
            {
                "internship_id": int(item[0]),
                "company_name": item[1],
                "role": item[2],
                "description": item[3],
                "technologies": item[4],
                "start_date": item[5],
                "end_date": item[6],
            }
            for item in internship_rows
        ]

        manual_skills = [
            item[0]
            for item in manual_skill_rows
        ]

        certifications = [
            {
                "certification_id": int(item[0]),
                "name": item[1],
                "issuer": item[2],
                "issue_date": item[3],
                "credential_url": item[4],
            }
            for item in certification_rows
        ]

        achievements = [
            {
                "achievement_id": int(item[0]),
                "title": item[1],
                "description": item[2],
                "achievement_date": item[3],
            }
            for item in achievement_rows
        ]

        return {

            "student_id": row[0],

            "student_name": row[1],
            "contact_email": profile_meta[0] or "",
            "phone": profile_meta[1] or "",
            "objective": profile_meta[2] or "",
            "soft_skills": [
                item.strip()
                for item in str(profile_meta[3] or "").split(",")
                if item.strip()
            ],
            "profile_image_path": profile_meta[4],

            "cgpa": _db_float(row[2]),

            "attendance_percent": _db_float(row[3]),

            "technical_skill_score": _db_float(row[4]),

            "aptitude_score": _db_float(row[5]),

            "communication_score": _db_float(row[6]),

            "projects_count": _db_int(row[7]),

            "internships_count": _db_int(row[8]),

            "certifications_count": _db_int(row[9]),

            "coding_assessment_score": _db_float(row[10]),

            "backlogs": _db_int(row[11]),

            "projects": projects,
            "internships": internships,
            "skills": manual_skills,
            "certifications": certifications,
            "achievements": achievements,

        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=f"Database error: {str(error)}"

        )

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# STUDENT PROFILE EDITING
# ==================================================

@app.put("/student/me/profile")
def update_my_profile(
    request: StudentProfileUpdate,
    user: dict = Depends(require_student),
):

    student_id = int(user["student_id"])

    fields = {
        "student_name": request.student_name,
        "contact_email": request.contact_email,
        "phone": request.phone,
        "objective": request.objective,
        "cgpa": request.cgpa,
        "tenth_percentage": request.tenth_percentage,
        "twelfth_percentage": request.twelfth_percentage,
        "sem_1_percentage": request.sem_1_percentage,
        "sem_1_sgpa": request.sem_1_sgpa,
        "sem_2_percentage": request.sem_2_percentage,
        "sem_2_sgpa": request.sem_2_sgpa,
        "sem_3_percentage": request.sem_3_percentage,
        "sem_3_sgpa": request.sem_3_sgpa,
        "sem_4_percentage": request.sem_4_percentage,
        "sem_4_sgpa": request.sem_4_sgpa,
        "sem_5_percentage": request.sem_5_percentage,
        "sem_5_sgpa": request.sem_5_sgpa,
        "sem_6_percentage": request.sem_6_percentage,
        "sem_6_sgpa": request.sem_6_sgpa,
        "sem_7_percentage": request.sem_7_percentage,
        "sem_7_sgpa": request.sem_7_sgpa,
        "sem_8_percentage": request.sem_8_percentage,
        "sem_8_sgpa": request.sem_8_sgpa,
        "attendance_percent": request.attendance_percent,
        "technical_skill_score": request.technical_skill_score,
        "aptitude_score": request.aptitude_score,
        "communication_score": request.communication_score,
        "projects_count": request.projects_count,
        "internships_count": request.internships_count,
        "certifications_count": request.certifications_count,
        "coding_assessment_score": request.coding_assessment_score,
        "backlogs": request.backlogs,
    }

    allowed = {
        key: value
        for key, value in fields.items()
        if value is not None
    }

    if "student_name" in allowed:
        allowed["student_name"] = str(
            allowed["student_name"]
        ).strip()

    for key in (
        "cgpa",
        "attendance_percent",
        "technical_skill_score",
        "aptitude_score",
        "communication_score",
        "coding_assessment_score",
    ):
        if key in allowed:
            if not (0 <= float(allowed[key]) <= (10 if key == "cgpa" else 100)):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {key}.",
                )

    for key, value in allowed.items():
        if key.endswith("_percentage") and value is not None and not (0 <= float(value) <= 100):
            raise HTTPException(status_code=400, detail=f"Invalid value for {key}.")
        if key.endswith("_sgpa") and value is not None and not (0 <= float(value) <= 10):
            raise HTTPException(status_code=400, detail=f"Invalid value for {key}.")

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        _ensure_student_detail_tables(connection, cursor)

        if allowed:
            assignments = [
                f"{column} = %s"
                for column in allowed
            ]
            values = list(allowed.values())
            values.append(student_id)

            cursor.execute(
                f"""
                UPDATE students
                SET {', '.join(assignments)}
                WHERE student_id = %s
                """,
                values,
            )

        if request.soft_skills is not None:
            cursor.execute(
                "UPDATE students SET soft_skills = %s WHERE student_id = %s",
                (
                    ", ".join(dict.fromkeys(
                        str(item).strip() for item in request.soft_skills if str(item).strip()
                    )),
                    student_id,
                ),
            )

        if request.skills is not None:
            cursor.execute(
                "DELETE FROM student_manual_skills WHERE student_id = %s",
                (student_id,),
            )

            for skill in request.skills:
                clean = str(skill).strip()
                if not clean:
                    continue

                cursor.execute(
                    """
                    INSERT INTO student_manual_skills
                        (student_id, skill_name)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (student_id, clean),
                )

        _refresh_student_skill_index(
            student_id,
            cursor,
        )

        connection.commit()

        return get_student(student_id)

    except HTTPException:
        if connection:
            connection.rollback()
        raise

    except Exception as error:
        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Profile update error: {error}",
        )

    finally:
        close_db(cursor, connection)



# ==================================================
# PROFILE EXTRAS: CERTIFICATIONS, ACHIEVEMENTS, PHOTO
# ==================================================

PROFILE_IMAGE_DIR = Path(__file__).resolve().parent / "uploads" / "profile_images"
PROFILE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
MAX_IMAGE_SIZE = 5 * 1024 * 1024


@app.post("/student/me/certifications")
def add_certification(request: CertificationRequest, user: dict = Depends(require_student)):
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Certification name is required.")
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor(); _ensure_student_detail_tables(conn, cur)
        cur.execute(
            """INSERT INTO student_certifications (student_id, name, issuer, issue_date, credential_url)
               VALUES (%s,%s,%s,%s,%s) RETURNING certification_id""",
            (int(user["student_id"]), request.name.strip(), request.issuer.strip(), request.issue_date.strip(), request.credential_url.strip()),
        )
        item_id = int(cur.fetchone()[0])
        cur.execute(
            """UPDATE students SET certifications_count = (SELECT COUNT(*) FROM student_certifications WHERE student_id=%s)
               WHERE student_id=%s""",
            (int(user["student_id"]), int(user["student_id"])),
        )
        _refresh_student_skill_index(int(user["student_id"]), cur); conn.commit()
        return {"certification_id": item_id, "message": "Certification added successfully."}
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Certification error: {error}")
    finally:
        close_db(cur, conn)


@app.put("/student/me/certifications/{certification_id}")
def update_certification(certification_id: int, request: CertificationRequest, user: dict = Depends(require_student)):
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute(
            """UPDATE student_certifications
               SET name=%s, issuer=%s, issue_date=%s, credential_url=%s
               WHERE certification_id=%s AND student_id=%s""",
            (request.name.strip(), request.issuer.strip(), request.issue_date.strip(), request.credential_url.strip(), certification_id, int(user["student_id"])),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Certification not found.")
        _refresh_student_skill_index(int(user["student_id"]), cur); conn.commit()
        return {"message": "Certification updated successfully."}
    except HTTPException:
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Certification update error: {error}")
    finally:
        close_db(cur, conn)


@app.delete("/student/me/certifications/{certification_id}")
def delete_certification(certification_id: int, user: dict = Depends(require_student)):
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute(
            "DELETE FROM student_certifications WHERE certification_id=%s AND student_id=%s",
            (certification_id, int(user["student_id"])),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Certification not found.")
        cur.execute(
            """UPDATE students SET certifications_count=(SELECT COUNT(*) FROM student_certifications WHERE student_id=%s)
               WHERE student_id=%s""",
            (int(user["student_id"]), int(user["student_id"])),
        )
        _refresh_student_skill_index(int(user["student_id"]), cur); conn.commit()
        return {"message": "Certification deleted successfully."}
    except HTTPException:
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Certification delete error: {error}")
    finally:
        close_db(cur, conn)


@app.post("/student/me/achievements")
def add_achievement(request: AchievementRequest, user: dict = Depends(require_student)):
    if not request.title.strip():
        raise HTTPException(status_code=400, detail="Achievement title is required.")
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor(); _ensure_student_detail_tables(conn, cur)
        cur.execute(
            """INSERT INTO student_achievements (student_id, title, description, achievement_date)
               VALUES (%s,%s,%s,%s) RETURNING achievement_id""",
            (int(user["student_id"]), request.title.strip(), request.description.strip(), request.achievement_date.strip()),
        )
        item_id = int(cur.fetchone()[0]); _refresh_student_skill_index(int(user["student_id"]), cur); conn.commit()
        return {"achievement_id": item_id, "message": "Achievement added successfully."}
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Achievement error: {error}")
    finally:
        close_db(cur, conn)


@app.put("/student/me/achievements/{achievement_id}")
def update_achievement(achievement_id: int, request: AchievementRequest, user: dict = Depends(require_student)):
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute(
            """UPDATE student_achievements
               SET title=%s, description=%s, achievement_date=%s
               WHERE achievement_id=%s AND student_id=%s""",
            (request.title.strip(), request.description.strip(), request.achievement_date.strip(), achievement_id, int(user["student_id"])),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Achievement not found.")
        _refresh_student_skill_index(int(user["student_id"]), cur); conn.commit()
        return {"message": "Achievement updated successfully."}
    except HTTPException:
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Achievement update error: {error}")
    finally:
        close_db(cur, conn)


@app.delete("/student/me/achievements/{achievement_id}")
def delete_achievement(achievement_id: int, user: dict = Depends(require_student)):
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("DELETE FROM student_achievements WHERE achievement_id=%s AND student_id=%s", (achievement_id, int(user["student_id"])))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Achievement not found.")
        _refresh_student_skill_index(int(user["student_id"]), cur); conn.commit()
        return {"message": "Achievement deleted successfully."}
    except HTTPException:
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Achievement delete error: {error}")
    finally:
        close_db(cur, conn)


@app.post("/student/me/profile-image")
async def upload_profile_image(file: UploadFile = File(...), user: dict = Depends(require_student)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Upload PNG, JPG/JPEG, or WEBP image.")
    payload = await file.read()
    if len(payload) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Profile image must be 5 MB or smaller.")
    student_id = int(user["student_id"])
    path = PROFILE_IMAGE_DIR / f"student_{student_id}_{uuid.uuid4().hex}{ALLOWED_IMAGE_TYPES[file.content_type]}"
    path.write_bytes(payload)
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT profile_image_path FROM students WHERE student_id=%s", (student_id,))
        old = cur.fetchone()
        cur.execute("UPDATE students SET profile_image_path=%s WHERE student_id=%s", (str(path), student_id)); conn.commit()
        if old and old[0]:
            try:
                oldp = Path(old[0]);
                if oldp.exists() and oldp != path: oldp.unlink()
            except OSError: pass
        return {"message": "Profile image uploaded successfully."}
    except Exception as error:
        if conn: conn.rollback()
        try: path.unlink()
        except OSError: pass
        raise HTTPException(status_code=500, detail=f"Profile image error: {error}")
    finally:
        close_db(cur, conn)


def _serve_profile_image(student_id: int):
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT profile_image_path FROM students WHERE student_id=%s", (student_id,)); row = cur.fetchone()
        if not row or not row[0]: raise HTTPException(status_code=404, detail="Profile image not found.")
        path = Path(row[0])
        if not path.exists(): raise HTTPException(status_code=404, detail="Profile image file not found.")
        mime = "image/png" if path.suffix.lower()==".png" else "image/webp" if path.suffix.lower()==".webp" else "image/jpeg"
        return FileResponse(str(path), media_type=mime)
    finally:
        close_db(cur, conn)


@app.get("/students/{student_id}/profile-image")
def get_student_profile_image(student_id: int, user: dict = Depends(require_student_or_faculty)):
    return _serve_profile_image(student_id)


@app.get("/student/me/profile-image")
def get_my_profile_image(user: dict = Depends(require_student)):
    return _serve_profile_image(int(user["student_id"]))


@app.delete("/student/me/profile-image")
def delete_profile_image(user: dict = Depends(require_student)):
    student_id = int(user["student_id"]); conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor(); cur.execute("SELECT profile_image_path FROM students WHERE student_id=%s", (student_id,)); row=cur.fetchone()
        cur.execute("UPDATE students SET profile_image_path=NULL WHERE student_id=%s", (student_id,)); conn.commit()
        if row and row[0]:
            try:
                p=Path(row[0]);
                if p.exists(): p.unlink()
            except OSError: pass
        return {"message":"Profile image removed successfully."}
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Profile image deletion error: {error}")
    finally:
        close_db(cur, conn)


# ==================================================
# FACULTY STUDENT PROFILE MANAGEMENT
# ==================================================

@app.put("/faculty/students/{student_id}/profile")
def faculty_update_student_profile(student_id: int, request: StudentProfileUpdate, user: dict = Depends(require_faculty)):
    if not isinstance(student_id, int) or student_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid student id.")
    _live_get_student_feature_dict(student_id)
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor(); _ensure_student_detail_tables(conn, cur)
        fields = {
            "student_name": request.student_name, "contact_email": request.contact_email, "phone": request.phone,
            "objective": request.objective, "cgpa": request.cgpa, "tenth_percentage": request.tenth_percentage,
            "twelfth_percentage": request.twelfth_percentage, "sem_1_percentage": request.sem_1_percentage, "sem_1_sgpa": request.sem_1_sgpa,
            "sem_2_percentage": request.sem_2_percentage, "sem_2_sgpa": request.sem_2_sgpa, "sem_3_percentage": request.sem_3_percentage, "sem_3_sgpa": request.sem_3_sgpa,
            "sem_4_percentage": request.sem_4_percentage, "sem_4_sgpa": request.sem_4_sgpa, "sem_5_percentage": request.sem_5_percentage, "sem_5_sgpa": request.sem_5_sgpa,
            "sem_6_percentage": request.sem_6_percentage, "sem_6_sgpa": request.sem_6_sgpa,
            "sem_7_percentage": request.sem_7_percentage, "sem_7_sgpa": request.sem_7_sgpa,
            "sem_8_percentage": request.sem_8_percentage, "sem_8_sgpa": request.sem_8_sgpa, "backlogs": request.backlogs,
        }
        allowed = {k:v for k,v in fields.items() if v is not None}
        for key, value in allowed.items():
            if key == "cgpa" and not 0 <= float(value) <= 10: raise HTTPException(status_code=400, detail="CGPA must be between 0 and 10.")
            if key.endswith("_percentage") and not 0 <= float(value) <= 100: raise HTTPException(status_code=400, detail=f"{key} must be between 0 and 100.")
            if key.endswith("_sgpa") and not 0 <= float(value) <= 10: raise HTTPException(status_code=400, detail=f"{key} must be between 0 and 10.")
        if allowed:
            cur.execute(f"UPDATE students SET {', '.join(f'{k}=%s' for k in allowed)} WHERE student_id=%s", (*allowed.values(), student_id))
        if request.skills is not None:
            cur.execute("DELETE FROM student_manual_skills WHERE student_id=%s", (student_id,))
            for skill in request.skills:
                clean=str(skill).strip()
                if clean: cur.execute("INSERT INTO student_manual_skills(student_id,skill_name) VALUES(%s,%s) ON CONFLICT DO NOTHING", (student_id,clean))
        if request.soft_skills is not None:
            cur.execute("UPDATE students SET soft_skills=%s WHERE student_id=%s", (", ".join(dict.fromkeys(str(x).strip() for x in request.soft_skills if str(x).strip())), student_id))
        _refresh_student_skill_index(student_id, cur); conn.commit()
        return get_student(student_id)
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Faculty profile update error: {error}")
    finally:
        close_db(cur, conn)


def _faculty_target_student(student_id: int, user: dict):
    if user.get("role") != "faculty": raise HTTPException(status_code=403, detail="Faculty access required.")
    _live_get_student_feature_dict(student_id)
    return int(student_id)

@app.post("/faculty/students/{student_id}/projects")
def faculty_add_project(student_id: int, request: ProjectRequest, user: dict = Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user)
    if not request.title.strip() or not request.description.strip(): raise HTTPException(status_code=400, detail="Project title and description are required.")
    conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("INSERT INTO student_projects(student_id,title,description,technologies,start_date,end_date) VALUES(%s,%s,%s,%s,%s,%s) RETURNING project_id",(sid,request.title.strip(),request.description.strip(),request.technologies.strip(),request.start_date,request.end_date)); pid=int(cur.fetchone()[0])
        cur.execute("UPDATE students SET projects_count=projects_count+1 WHERE student_id=%s",(sid,)); _refresh_student_skill_index(sid,cur); conn.commit(); return {"project_id":pid,"message":"Project added successfully."}
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Project creation error: {error}")
    finally: close_db(cur,conn)

@app.post("/faculty/students/{student_id}/internships")
def faculty_add_internship(student_id: int, request: InternshipRequest, user: dict = Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user)
    if not request.company_name.strip() or not request.role.strip() or not request.description.strip(): raise HTTPException(status_code=400, detail="Company, role and description are required.")
    conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("INSERT INTO student_internships(student_id,company_name,role,description,technologies,start_date,end_date) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING internship_id",(sid,request.company_name.strip(),request.role.strip(),request.description.strip(),request.technologies.strip(),request.start_date,request.end_date)); iid=int(cur.fetchone()[0])
        cur.execute("UPDATE students SET internships_count=internships_count+1 WHERE student_id=%s",(sid,)); _refresh_student_skill_index(sid,cur); conn.commit(); return {"internship_id":iid,"message":"Internship added successfully."}
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Internship creation error: {error}")
    finally: close_db(cur,conn)

@app.post("/faculty/students/{student_id}/certifications")
def faculty_add_certification(student_id: int, request: CertificationRequest, user: dict = Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user)
    if not request.name.strip(): raise HTTPException(status_code=400, detail="Certification name is required.")
    conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("INSERT INTO student_certifications(student_id,name,issuer,issue_date,credential_url) VALUES(%s,%s,%s,%s,%s) RETURNING certification_id",(sid,request.name.strip(),request.issuer.strip(),request.issue_date.strip(),request.credential_url.strip())); cid=int(cur.fetchone()[0])
        cur.execute("UPDATE students SET certifications_count=(SELECT COUNT(*) FROM student_certifications WHERE student_id=%s) WHERE student_id=%s",(sid,sid)); _refresh_student_skill_index(sid,cur); conn.commit(); return {"certification_id":cid,"message":"Certification added successfully."}
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Certification error: {error}")
    finally: close_db(cur,conn)

@app.post("/faculty/students/{student_id}/achievements")
def faculty_add_achievement(student_id: int, request: AchievementRequest, user: dict = Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user)
    if not request.title.strip(): raise HTTPException(status_code=400, detail="Achievement title is required.")
    conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("INSERT INTO student_achievements(student_id,title,description,achievement_date) VALUES(%s,%s,%s,%s) RETURNING achievement_id",(sid,request.title.strip(),request.description.strip(),request.achievement_date.strip())); aid=int(cur.fetchone()[0]); _refresh_student_skill_index(sid,cur); conn.commit(); return {"achievement_id":aid,"message":"Achievement added successfully."}
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Achievement error: {error}")
    finally: close_db(cur,conn)

@app.post("/student/me/generate-objective")
def generate_student_objective(request: ObjectiveGenerateRequest, user: dict = Depends(require_student)):
    profile=get_student(int(user["student_id"]))
    role=request.target_role.strip() or "a suitable entry-level role"
    payload={"name":profile.get("student_name"),"branch":profile.get("Branch"),"area_of_interest":profile.get("Area_of_Interest"),"skills":profile.get("skills") or [],"project_descriptions":[x.get("description","") for x in profile.get("projects",[])],"internship_descriptions":[x.get("description","") for x in profile.get("internships",[])]}
    schema={"type":"object","properties":{"objective":{"type":"string"}},"required":["objective"]}
    prompt=f"Write a concise 2-3 sentence career objective for the target role '{role}' using only the supplied verified student profile. Do not invent experience, companies, projects, skills, achievements or qualifications. If evidence is limited, keep the objective factual. PROFILE={json.dumps(payload,ensure_ascii=False)}"
    try:
        response=generate_gemini_response("ai",[types.Content(role="user",parts=[types.Part.from_text(text=prompt)])],types.GenerateContentConfig(system_instruction="You are a factual career objective writer. Use only supplied evidence.",max_output_tokens=350,response_mime_type="application/json",response_json_schema=schema))
        result=parse_final_json(response.text)
        objective=str(result.get("objective","")).strip() if isinstance(result,dict) else ""
        if not objective: raise ValueError("Gemini returned an empty objective.")
        return {"objective":objective,"target_role":role,"source":"Gemini"}
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Gemini objective generation unavailable: {error}")


@app.put("/faculty/students/{student_id}/projects/{project_id}")
def faculty_update_project(student_id:int, project_id:int, request:ProjectRequest, user:dict=Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user)
    if not request.title.strip() or not request.description.strip(): raise HTTPException(status_code=400,detail="Project title and description are required.")
    conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("UPDATE student_projects SET title=%s,description=%s,technologies=%s,start_date=%s,end_date=%s WHERE project_id=%s AND student_id=%s RETURNING project_id",(request.title.strip(),request.description.strip(),request.technologies.strip(),request.start_date,request.end_date,project_id,sid))
        if cur.fetchone() is None: raise HTTPException(status_code=404,detail="Project not found.")
        _refresh_student_skill_index(sid,cur); conn.commit(); return {"message":"Project updated successfully."}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500,detail=f"Project update error: {error}")
    finally: close_db(cur,conn)

@app.delete("/faculty/students/{student_id}/projects/{project_id}")
def faculty_delete_project(student_id:int, project_id:int, user:dict=Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user); conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("DELETE FROM student_projects WHERE project_id=%s AND student_id=%s",(project_id,sid))
        if cur.rowcount==0: raise HTTPException(status_code=404,detail="Project not found.")
        cur.execute("UPDATE students SET projects_count=GREATEST(projects_count-1,0) WHERE student_id=%s",(sid,)); _refresh_student_skill_index(sid,cur); conn.commit(); return {"message":"Project deleted successfully."}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500,detail=f"Project delete error: {error}")
    finally: close_db(cur,conn)

@app.put("/faculty/students/{student_id}/internships/{internship_id}")
def faculty_update_internship(student_id:int, internship_id:int, request:InternshipRequest, user:dict=Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user)
    if not request.company_name.strip() or not request.role.strip() or not request.description.strip(): raise HTTPException(status_code=400,detail="Company, role and description are required.")
    conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("UPDATE student_internships SET company_name=%s,role=%s,description=%s,technologies=%s,start_date=%s,end_date=%s WHERE internship_id=%s AND student_id=%s RETURNING internship_id",(request.company_name.strip(),request.role.strip(),request.description.strip(),request.technologies.strip(),request.start_date,request.end_date,internship_id,sid))
        if cur.fetchone() is None: raise HTTPException(status_code=404,detail="Internship not found.")
        _refresh_student_skill_index(sid,cur); conn.commit(); return {"message":"Internship updated successfully."}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500,detail=f"Internship update error: {error}")
    finally: close_db(cur,conn)

@app.delete("/faculty/students/{student_id}/internships/{internship_id}")
def faculty_delete_internship(student_id:int, internship_id:int, user:dict=Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user); conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("DELETE FROM student_internships WHERE internship_id=%s AND student_id=%s",(internship_id,sid))
        if cur.rowcount==0: raise HTTPException(status_code=404,detail="Internship not found.")
        cur.execute("UPDATE students SET internships_count=GREATEST(internships_count-1,0) WHERE student_id=%s",(sid,)); _refresh_student_skill_index(sid,cur); conn.commit(); return {"message":"Internship deleted successfully."}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500,detail=f"Internship delete error: {error}")
    finally: close_db(cur,conn)

@app.put("/faculty/students/{student_id}/certifications/{certification_id}")
def faculty_update_certification(student_id:int, certification_id:int, request:CertificationRequest, user:dict=Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user); conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("UPDATE student_certifications SET name=%s,issuer=%s,issue_date=%s,credential_url=%s WHERE certification_id=%s AND student_id=%s RETURNING certification_id",(request.name.strip(),request.issuer.strip(),request.issue_date.strip(),request.credential_url.strip(),certification_id,sid))
        if cur.fetchone() is None: raise HTTPException(status_code=404,detail="Certification not found.")
        _refresh_student_skill_index(sid,cur); conn.commit(); return {"message":"Certification updated successfully."}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500,detail=f"Certification update error: {error}")
    finally: close_db(cur,conn)

@app.delete("/faculty/students/{student_id}/certifications/{certification_id}")
def faculty_delete_certification(student_id:int, certification_id:int, user:dict=Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user); conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("DELETE FROM student_certifications WHERE certification_id=%s AND student_id=%s",(certification_id,sid))
        if cur.rowcount==0: raise HTTPException(status_code=404,detail="Certification not found.")
        cur.execute("UPDATE students SET certifications_count=(SELECT COUNT(*) FROM student_certifications WHERE student_id=%s) WHERE student_id=%s",(sid,sid)); _refresh_student_skill_index(sid,cur); conn.commit(); return {"message":"Certification deleted successfully."}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500,detail=f"Certification delete error: {error}")
    finally: close_db(cur,conn)

@app.put("/faculty/students/{student_id}/achievements/{achievement_id}")
def faculty_update_achievement(student_id:int, achievement_id:int, request:AchievementRequest, user:dict=Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user); conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("UPDATE student_achievements SET title=%s,description=%s,achievement_date=%s WHERE achievement_id=%s AND student_id=%s RETURNING achievement_id",(request.title.strip(),request.description.strip(),request.achievement_date.strip(),achievement_id,sid))
        if cur.fetchone() is None: raise HTTPException(status_code=404,detail="Achievement not found.")
        _refresh_student_skill_index(sid,cur); conn.commit(); return {"message":"Achievement updated successfully."}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500,detail=f"Achievement update error: {error}")
    finally: close_db(cur,conn)

@app.delete("/faculty/students/{student_id}/achievements/{achievement_id}")
def faculty_delete_achievement(student_id:int, achievement_id:int, user:dict=Depends(require_faculty)):
    sid=_faculty_target_student(student_id,user); conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur)
        cur.execute("DELETE FROM student_achievements WHERE achievement_id=%s AND student_id=%s",(achievement_id,sid))
        if cur.rowcount==0: raise HTTPException(status_code=404,detail="Achievement not found.")
        _refresh_student_skill_index(sid,cur); conn.commit(); return {"message":"Achievement deleted successfully."}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as error:
        if conn: conn.rollback()
        raise HTTPException(status_code=500,detail=f"Achievement delete error: {error}")
    finally: close_db(cur,conn)

@app.post("/faculty/students/{student_id}/generate-objective")
def generate_faculty_student_objective(student_id: int, request: ObjectiveGenerateRequest, user: dict = Depends(require_faculty)):
    profile=get_student(_faculty_target_student(student_id,user))
    role=request.target_role.strip() or "a suitable entry-level role"
    payload={"name":profile.get("student_name"),"branch":profile.get("Branch"),"area_of_interest":profile.get("Area_of_Interest"),"skills":profile.get("skills") or [],"project_descriptions":[x.get("description","") for x in profile.get("projects",[])],"internship_descriptions":[x.get("description","") for x in profile.get("internships",[])]}
    schema={"type":"object","properties":{"objective":{"type":"string"}},"required":["objective"]}
    prompt=f"Write a concise 2-3 sentence career objective for the target role '{role}' using only the supplied verified student profile. Do not invent experience, companies, projects, skills, achievements or qualifications. If evidence is limited, keep the objective factual. PROFILE={json.dumps(payload,ensure_ascii=False)}"
    try:
        response=generate_gemini_response("ai",[types.Content(role="user",parts=[types.Part.from_text(text=prompt)])],types.GenerateContentConfig(system_instruction="You are a factual career objective writer. Use only supplied evidence.",max_output_tokens=350,response_mime_type="application/json",response_json_schema=schema))
        result=parse_final_json(response.text); objective=str(result.get("objective","")).strip() if isinstance(result,dict) else ""
        if not objective: raise ValueError("Gemini returned an empty objective.")
        return {"objective":objective,"target_role":role,"source":"Gemini"}
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Gemini objective generation unavailable: {error}")

# ==================================================
# REPORT GENERATORS
# ==================================================
# REPORT EXPORT IS LOCAL:
# PDF/DOCX generation uses existing profile/report data only.
# These endpoints MUST NOT call Gemini.

def _ensure_report_deps():
    """Verify local PDF/DOCX libraries before document rendering."""
    missing = []

    try:
        import reportlab  # noqa: F401
    except ImportError:
        missing.append("reportlab")

    try:
        import docx  # noqa: F401
    except ImportError:
        missing.append("python-docx")

    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                "Document export is unavailable because these packages are missing: "
                + ", ".join(missing)
                + ". Run inside the active .venv: pip install "
                + " ".join(missing)
            ),
        )


def _build_pdf(profile: dict, path: str, mode: str):
    _ensure_report_deps()
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib import colors

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=36, bottomMargin=36)
    styles=getSampleStyleSheet()
    title=ParagraphStyle("ResumeTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22)
    section=ParagraphStyle("ResumeSection", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=10, spaceAfter=5)
    body=ParagraphStyle("ResumeBody", parent=styles["BodyText"], fontSize=9, leading=13)
    story=[]
    img=profile.get("profile_image_path")
    if img and Path(img).exists():
        try: story += [Image(img, width=.8*inch, height=.8*inch), Spacer(1,4)]
        except Exception: pass
    story.append(Paragraph(html_escape(str(profile.get("student_name","Student"))), title))
    contact=" | ".join(v for v in [profile.get("contact_email",""), profile.get("phone","")] if v)
    if contact: story.append(Paragraph(html_escape(contact), body))
    if profile.get("objective"): story += [Paragraph("Career Objective", section), Paragraph(html_escape(str(profile["objective"])), body)]
    for label,key in [("Technical Skills","skills"),("Soft Skills","soft_skills")]:
        raw_vals=profile.get(key) or []
        vals=raw_vals if isinstance(raw_vals,list) else [x.strip() for x in str(raw_vals).split(",") if x.strip()]
        if vals: story += [Paragraph(label, section), Paragraph(html_escape(", ".join(map(str, vals))), body)]
    if mode=="faculty":
        story.append(Paragraph("Academic & Placement Metrics", section))
        metric_values=[("CGPA",profile.get("cgpa")),("Attendance",profile.get("attendance_percent")),("Technical Score",profile.get("technical_skill_score")),("Aptitude",profile.get("aptitude_score")),("Communication",profile.get("communication_score")),("Coding Assessment",profile.get("coding_assessment_score")),("Backlogs",profile.get("backlogs")),("Projects",profile.get("projects_count")),("Internships",profile.get("internships_count")),("Certifications",profile.get("certifications_count"))]
        data=[["Metric","Value"]]+[[label,(f"{value}%" if label=="Attendance" and value is not None else str(value))] for label,value in metric_values if value is not None]
        t=Table(data,colWidths=[220,180]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#12284A")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey)])); story += [t]
        if profile.get("predicted_readiness"):
            story += [Paragraph("Placement Readiness", section), Paragraph(str(profile["predicted_readiness"]), body)]
        if profile.get("shap_explanation"):
            story += [Paragraph("Reason for Placement Readiness", section), Paragraph(html_escape(str(profile["shap_explanation"])), body)]
    for heading,key in [("Projects","projects"),("Internships","internships")]:
        items=profile.get(key) or []
        if not items:
            continue
        story.append(Paragraph(heading,section))
        for item in items:
            if key=="projects": text=f"<b>{html_escape(str(item.get('title','')))}</b><br/>{html_escape(str(item.get('description','')))}<br/><b>Technologies:</b> {html_escape(str(item.get('technologies','')))}"
            else: text=f"<b>{html_escape(str(item.get('role','')))} — {html_escape(str(item.get('company_name','')))}</b><br/>{html_escape(str(item.get('description','')))}<br/><b>Technologies:</b> {html_escape(str(item.get('technologies','')))}"
            story += [Paragraph(text,body), Spacer(1,4)]
    if profile.get("certifications"):
        story.append(Paragraph("Certifications",section))
        for item in profile.get("certifications") or []: story.append(Paragraph(f"<b>{html_escape(str(item.get('name','')))}</b> — {html_escape(str(item.get('issuer','')))}",body))
    if profile.get("achievements"):
        story.append(Paragraph("Achievements",section))
        for item in profile.get("achievements") or []: story.append(Paragraph(f"<b>{html_escape(str(item.get('title','')))}</b> — {html_escape(str(item.get('description','')))}",body))
    doc.build(story)


def _build_docx(profile: dict, path: str, mode: str):
    _ensure_report_deps()
    from docx import Document
    from docx.shared import Inches
    doc=Document(); doc.add_heading("Student Resume" if mode=="resume" else "Student Placement Assessment Report",0); doc.add_paragraph(profile.get("student_name","Student"))
    img=profile.get("profile_image_path")
    if img and Path(img).exists():
        try: doc.add_picture(img,width=Inches(1.0))
        except Exception: pass
    if profile.get("contact_email"): doc.add_paragraph(f"Email: {profile['contact_email']}")
    if profile.get("phone"): doc.add_paragraph(f"Phone: {profile['phone']}")
    if profile.get("objective"): doc.add_heading("Career Objective",1); doc.add_paragraph(profile["objective"])
    if mode=="faculty":
        doc.add_heading("Academic & Placement Metrics",1)
        for label,value in [("CGPA",profile.get("cgpa")), ("Attendance",f"{profile.get('attendance_percent')}%"), ("Technical Skill Score",profile.get("technical_skill_score")), ("Aptitude",profile.get("aptitude_score")), ("Communication",profile.get("communication_score")), ("Coding Assessment",profile.get("coding_assessment_score")), ("Backlogs",profile.get("backlogs"))]: doc.add_paragraph(f"{label}: {value}")
        if profile.get("predicted_readiness"):
            doc.add_heading("Placement Readiness",1); doc.add_paragraph(str(profile["predicted_readiness"]))
        if profile.get("shap_explanation"):
            doc.add_heading("Reason for Placement Readiness",1); doc.add_paragraph(str(profile["shap_explanation"]))
    if profile.get("skills"): doc.add_heading("Technical Skills",1); doc.add_paragraph(", ".join(profile["skills"]))
    if profile.get("soft_skills"): doc.add_heading("Soft Skills",1); doc.add_paragraph(", ".join(profile["soft_skills"]))
    for heading,key in [("Projects","projects"),("Internships","internships")]:
        items=profile.get(key) or []
        if not items:
            continue
        doc.add_heading(heading,1)
        for item in items:
            if key=="projects": doc.add_heading(item.get("title","Project"),2)
            else: doc.add_heading(f"{item.get('role','')} — {item.get('company_name','')}",2)
            doc.add_paragraph(item.get("description","")); doc.add_paragraph(f"Technologies: {item.get('technologies','')}")
    if profile.get("certifications"):
        doc.add_heading("Certifications",1)
        for item in profile.get("certifications") or []: doc.add_paragraph(f"{item.get('name','')} — {item.get('issuer','')}")
    if profile.get("achievements"):
        doc.add_heading("Achievements",1)
        for item in profile.get("achievements") or []: doc.add_paragraph(f"{item.get('title','')} — {item.get('description','')}")
    doc.save(path)


def _augment_with_ml_results(profile: dict, student_id: int) -> dict:
    """Add the existing ML readiness and SHAP explanation to faculty reports."""
    enriched = dict(profile)
    try:
        prediction_result = predict_student(student_id)
        enriched["predicted_readiness"] = prediction_result.get("predicted_readiness")
    except Exception:
        enriched["predicted_readiness"] = None

    try:
        explanation_result = explain_student(student_id)
        enriched["shap_explanation"] = explanation_result.get("explanation")
    except Exception:
        enriched["shap_explanation"] = None

    return enriched


@app.get("/student/me/report/pdf")
def student_report_pdf(user: dict = Depends(require_student)):
    p=get_student(int(user["student_id"])); path=Path(tempfile.gettempdir())/f"student_{uuid.uuid4().hex}.pdf"; _build_pdf(p,str(path),"resume"); return FileResponse(str(path),media_type="application/pdf",filename=f"{p['student_name']}_Resume.pdf")

@app.get("/student/me/report/docx")
def student_report_docx(user: dict = Depends(require_student)):
    p=get_student(int(user["student_id"])); path=Path(tempfile.gettempdir())/f"student_{uuid.uuid4().hex}.docx"; _build_docx(p,str(path),"resume"); return FileResponse(str(path),media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",filename=f"{p['student_name']}_Resume.docx")

@app.get("/students/{student_id}/report/pdf")
def faculty_report_pdf(student_id:int,user:dict=Depends(require_faculty)):
    p=_augment_with_ml_results(get_student(student_id), student_id); path=Path(tempfile.gettempdir())/f"faculty_{uuid.uuid4().hex}.pdf"; _build_pdf(p,str(path),"faculty"); return FileResponse(str(path),media_type="application/pdf",filename=f"{p['student_name']}_Placement_Report.pdf")

@app.get("/students/{student_id}/report/docx")
def faculty_report_docx(student_id:int,user:dict=Depends(require_faculty)):
    p=_augment_with_ml_results(get_student(student_id), student_id); path=Path(tempfile.gettempdir())/f"faculty_{uuid.uuid4().hex}.docx"; _build_docx(p,str(path),"faculty"); return FileResponse(str(path),media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",filename=f"{p['student_name']}_Placement_Report.docx")


# ==================================================
# AI RESUME GENERATOR AGENT
# ==================================================
ROLE_TARGETS={
    "data analyst":["SQL","Python","Pandas","Power BI","Tableau","Excel","Data Analysis","Data Visualization","Statistics"],
    "data scientist":["Python","SQL","Pandas","NumPy","Machine Learning","Scikit-learn","Statistics","Feature Engineering","Model Evaluation","Data Science"],
    "ml engineer":["Python","Machine Learning","Scikit-learn","TensorFlow","PyTorch","FastAPI","Docker","Model Deployment","MLOps"],
    "frontend developer":["React","JavaScript","TypeScript","HTML","CSS","Tailwind CSS","Git"],
    "backend developer":["Python","FastAPI","Django","Flask","SQL","PostgreSQL","REST API","Docker","Git"],
    "power bi developer":["Power BI","Power Query","SQL","Excel","Data Visualization","Business Intelligence"],
}

def _target_role_skills(role:str)->set:
    r=role.lower().strip()
    for name,skills in ROLE_TARGETS.items():
        if name in r or r in name: return set(skills)
    return set(extract_skills(role))

def _role_score(text:str,target:set,role:str)->int:
    score=len(set(extract_skills(text)).intersection(target)); rt=set(re.findall(r"[a-zA-Z]{3,}",role.lower())); tt=set(re.findall(r"[a-zA-Z]{3,}",text.lower())); return score+len(rt.intersection(tt))

@app.post("/student/me/generate-resume")
def generate_student_resume(request:ResumeGenerateRequest,user:dict=Depends(require_student)):
    role=request.target_role.strip()
    if not role: raise HTTPException(status_code=400,detail="Target role is required.")
    profile=get_student(int(user["student_id"])); target=_target_role_skills(role)
    def pick(items, builder, limit=5):
        ranked=sorted([(_role_score(builder(x),target,role),x) for x in items], key=lambda z:z[0], reverse=True); chosen=[x for s,x in ranked[:limit] if s>0]; return chosen or list(items[:min(limit,3)])
    projects=list(profile.get("projects") or [])
    legacy_project=str(profile.get("Project_Description") or "").strip()
    if legacy_project and not any(legacy_project.lower() == str(x.get("description") or "").strip().lower() for x in projects):
        projects.append({"title":"Verified Project","description":legacy_project,"technologies":", ".join(extract_skills(legacy_project))})
    projects=pick(projects,lambda x:" ".join([x.get("title",""),x.get("description",""),x.get("technologies","")]),5)
    internships=list(profile.get("internships") or [])
    legacy_internship=str(profile.get("Internship_Description") or "").strip()
    if legacy_internship and not any(legacy_internship.lower() == str(x.get("description") or "").strip().lower() for x in internships):
        internships.append({"company_name":"","role":str(profile.get("Internship_Type") or "Internship"),"description":legacy_internship,"technologies":", ".join(extract_skills(legacy_internship))})
    internships=pick(internships,lambda x:" ".join([x.get("company_name",""),x.get("role",""),x.get("description",""),x.get("technologies","")]),5)
    certs=pick(list(profile.get("certifications") or []),lambda x:" ".join([x.get("name",""),x.get("issuer","")]),8)
    achievements=list(profile.get("achievements") or [])
    if not achievements and str(profile.get("Achievements") or "").strip():
        raw=str(profile.get("Achievements") or "").strip()
        achievements=[{"title":part.strip(),"description":""} for part in re.split(r"\s*[|;]\s*",raw) if part.strip()]
    achievements=pick(achievements,lambda x:" ".join([x.get("title",""),x.get("description","")]),5)
    technology_corpus = " ".join([
        " ".join(profile.get("skills") or []),
        str(profile.get("Project_Description") or ""),
        str(profile.get("Internship_Description") or ""),
        " ".join(x.get("technologies", "") for x in projects),
        " ".join(x.get("technologies", "") for x in internships),
    ])
    selected_skills = sorted(
        set(profile.get("skills") or []).union(
            target.intersection(set(extract_skills(technology_corpus)))
        )
    )
    payload={"target_role":role,"student_name":profile.get("student_name",""),"objective":profile.get("objective",""),"technical_skills":selected_skills,"soft_skills":profile.get("soft_skills") or [],"projects":projects,"internships":internships,"certifications":certs,"achievements":achievements}
    schema={"type":"object","properties":{"headline":{"type":"string"},"objective":{"type":"string"},"technical_skills":{"type":"array","items":{"type":"string"}},"soft_skills":{"type":"array","items":{"type":"string"}},"projects":{"type":"array","items":{"type":"object","properties":{"title":{"type":"string"},"description":{"type":"string"},"technologies":{"type":"string"}},"required":["title","description","technologies"]}},"internships":{"type":"array","items":{"type":"object","properties":{"company_name":{"type":"string"},"role":{"type":"string"},"description":{"type":"string"},"technologies":{"type":"string"}},"required":["company_name","role","description","technologies"]}},"certifications":{"type":"array","items":{"type":"string"}},"achievements":{"type":"array","items":{"type":"string"}}},"required":["headline","objective","technical_skills","soft_skills","projects","internships","certifications","achievements"]}
    prompt=f"You are a factual resume-writing agent. Create a concise resume for {role} using only the supplied data. Never invent experience, dates, employers, projects, certifications, achievements or technologies. Prioritize relevant evidence. PROFILE={json.dumps(payload,ensure_ascii=False)}"
    try:
        response=generate_gemini_response("ai",[types.Content(role="user",parts=[types.Part.from_text(text=prompt)])],types.GenerateContentConfig(system_instruction="Use only supplied facts and produce concise professional resume wording. Never create a section or item that is absent from the supplied profile.",max_output_tokens=1800,response_mime_type="application/json",response_json_schema=schema))
        result=parse_final_json(response.text)
        if not isinstance(result,dict) or not result.get("headline"): raise ValueError("Incomplete AI resume response")
        result["technical_skills"]=selected_skills
        result["soft_skills"]=profile.get("soft_skills") or []
        result["certifications"]=[x.get("name","") for x in certs if x.get("name")]
        result["achievements"]=[x.get("title","") for x in achievements if x.get("title")]
        if not projects: result["projects"]=[]
        if not internships: result["internships"]=[]
        if not certs: result["certifications"]=[]
        if not achievements: result["achievements"]=[]
        status="AI-generated wording using locally selected factual content."
    except Exception as error:
        result={"headline":role,"objective":profile.get("objective","") or "","technical_skills":selected_skills,"soft_skills":profile.get("soft_skills") or [],"projects":projects,"internships":internships,"certifications":[x.get("name","") for x in certs if x.get("name")],"achievements":[x.get("title","") for x in achievements if x.get("title")]}; status=f"Local factual fallback used: {error}"
    return {"target_role":role,"resume":result,"selected_counts":{"projects":len(projects),"internships":len(internships),"certifications":len(certs),"achievements":len(achievements)},"status":status}



# ==================================================
# ROLE-SPECIFIC RESUME EXPORTS
# ==================================================
# IMPORTANT: PDF/DOCX export only renders the already-generated resume.
# It MUST NOT call Gemini again.

def _resume_list(value):
    if isinstance(value,list):
        return [x for x in value if x not in (None, "")]
    if value in (None, ""):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _build_generated_resume_pdf(resume: dict, profile: dict, path: str):
    _ensure_report_deps()
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

    styles=getSampleStyleSheet()
    title=ParagraphStyle("PlacementIQResumeTitle",parent=styles["Title"],alignment=TA_CENTER,fontSize=18,leading=22,spaceAfter=8)
    section=ParagraphStyle("PlacementIQResumeSection",parent=styles["Heading2"],fontSize=11,leading=14,spaceBefore=10,spaceAfter=5)
    body=ParagraphStyle("PlacementIQResumeBody",parent=styles["BodyText"],fontSize=9.5,leading=13)
    doc=SimpleDocTemplate(path,pagesize=A4,leftMargin=40,rightMargin=40,topMargin=36,bottomMargin=36)
    story=[]
    img=profile.get("profile_image_path")
    if img and Path(img).exists():
        try: story += [Image(img,width=.8*inch,height=.8*inch),Spacer(1,4)]
        except Exception: pass
    headline=str(resume.get("headline") or profile.get("student_name") or "Resume")
    story.append(Paragraph(html_escape(headline),title))
    target_role=str(resume.get("target_role") or "").strip()
    if target_role: story.append(Paragraph(html_escape(target_role),body))
    contact=" | ".join(v for v in [str(profile.get("contact_email") or "").strip(),str(profile.get("phone") or "").strip()] if v)
    if contact: story.append(Paragraph(html_escape(contact),body))
    objective=str(resume.get("objective") or "").strip()
    if objective: story += [Paragraph("Career Objective",section),Paragraph(html_escape(objective),body)]
    education=[]
    if profile.get("Branch"): education.append(f"B.Tech — {profile.get('Branch')}")
    if profile.get("CGPA") not in (None, ""): education.append(f"CGPA: {profile.get('CGPA')}/10")
    if profile.get("12th_Percentage") not in (None, ""): education.append(f"12th: {profile.get('12th_Percentage')}%")
    if profile.get("10th_Percentage") not in (None, ""): education.append(f"10th: {profile.get('10th_Percentage')}%")
    if education: story += [Paragraph("Education",section),Paragraph(html_escape(" | ".join(education)),body)]
    for label,key in [("Technical Skills","technical_skills"),("Soft Skills","soft_skills")]:
        vals=_resume_list(resume.get(key))
        if vals: story += [Paragraph(label,section),Paragraph(html_escape(", ".join(map(str,vals))),body)]
    projects=resume.get("projects") or []
    if projects:
        story.append(Paragraph("Projects",section))
        for item in projects:
            title_text=str(item.get("title") or "Project")
            desc=str(item.get("description") or "").strip()
            tech=str(item.get("technologies") or "").strip()
            story.append(Paragraph(f"<b>{html_escape(title_text)}</b>",body))
            if desc: story.append(Paragraph(html_escape(desc),body))
            if tech: story.append(Paragraph(f"<b>Technologies:</b> {html_escape(tech)}",body))
            story.append(Spacer(1,5))
    internships=resume.get("internships") or []
    if internships:
        story.append(Paragraph("Internships",section))
        for item in internships:
            role=str(item.get("role") or "Internship")
            company=str(item.get("company_name") or "").strip()
            desc=str(item.get("description") or "").strip()
            tech=str(item.get("technologies") or "").strip()
            heading=f"{role} — {company}" if company else role
            story.append(Paragraph(f"<b>{html_escape(heading)}</b>",body))
            if desc: story.append(Paragraph(html_escape(desc),body))
            if tech: story.append(Paragraph(f"<b>Technologies:</b> {html_escape(tech)}",body))
            story.append(Spacer(1,5))
    certs=_resume_list(resume.get("certifications"))
    if certs:
        story.append(Paragraph("Certifications",section))
        for item in certs: story.append(Paragraph(html_escape(str(item)),body))
    achievements=_resume_list(resume.get("achievements"))
    if achievements:
        story.append(Paragraph("Achievements",section))
        for item in achievements: story.append(Paragraph(html_escape(str(item)),body))
    doc.build(story)


def _build_generated_resume_docx(resume: dict, profile: dict, path: str):
    _ensure_report_deps()
    from docx import Document
    from docx.shared import Inches
    doc=Document()
    doc.add_heading(resume.get("headline") or profile.get("student_name") or "Resume",0)
    if resume.get("target_role"): doc.add_paragraph(str(resume.get("target_role")))
    img=profile.get("profile_image_path")
    if img and Path(img).exists():
        try: doc.add_picture(img,width=Inches(1.0))
        except Exception: pass
    contact=" | ".join(v for v in [str(profile.get("contact_email") or "").strip(),str(profile.get("phone") or "").strip()] if v)
    if contact: doc.add_paragraph(contact)
    objective=str(resume.get("objective") or "").strip()
    if objective: doc.add_heading("Career Objective",1); doc.add_paragraph(objective)
    education=[]
    if profile.get("Branch"): education.append(f"B.Tech — {profile.get('Branch')}")
    if profile.get("CGPA") not in (None, ""): education.append(f"CGPA: {profile.get('CGPA')}/10")
    if profile.get("12th_Percentage") not in (None, ""): education.append(f"12th: {profile.get('12th_Percentage')}%")
    if profile.get("10th_Percentage") not in (None, ""): education.append(f"10th: {profile.get('10th_Percentage')}%")
    if education: doc.add_heading("Education",1); doc.add_paragraph(" | ".join(education))
    technical=_resume_list(resume.get("technical_skills"))
    if technical: doc.add_heading("Technical Skills",1); doc.add_paragraph(", ".join(map(str,technical)))
    soft=_resume_list(resume.get("soft_skills"))
    if soft: doc.add_heading("Soft Skills",1); doc.add_paragraph(", ".join(map(str,soft)))
    projects=resume.get("projects") or []
    if projects:
        doc.add_heading("Projects",1)
        for item in projects:
            doc.add_heading(item.get("title") or "Project",2)
            if item.get("description"): doc.add_paragraph(item.get("description"))
            if item.get("technologies"): doc.add_paragraph(f"Technologies: {item.get('technologies')}")
    internships=resume.get("internships") or []
    if internships:
        doc.add_heading("Internships",1)
        for item in internships:
            heading=f"{item.get('role') or 'Internship'} — {item.get('company_name')}" if item.get("company_name") else (item.get("role") or "Internship")
            doc.add_heading(heading,2)
            if item.get("description"): doc.add_paragraph(item.get("description"))
            if item.get("technologies"): doc.add_paragraph(f"Technologies: {item.get('technologies')}")
    certs=_resume_list(resume.get("certifications"))
    if certs:
        doc.add_heading("Certifications",1)
        for item in certs: doc.add_paragraph(str(item))
    achievements=_resume_list(resume.get("achievements"))
    if achievements:
        doc.add_heading("Achievements",1)
        for item in achievements: doc.add_paragraph(str(item))
    doc.save(path)


@app.post("/student/me/generate-resume/pdf")
def generate_student_resume_pdf(
    request: ResumeExportRequest,
    user: dict = Depends(require_student),
):
    """Render an already-generated resume. No Gemini call happens here."""

    if not request.target_role.strip():
        raise HTTPException(status_code=400, detail="Target role is required.")

    path = (
        Path(tempfile.gettempdir())
        / f"role_resume_{uuid.uuid4().hex}.pdf"
    )

    profile = get_student(int(user["student_id"]))
    export_resume = dict(request.resume or {})
    export_resume["target_role"] = request.target_role.strip()
    export_resume["headline"] = profile.get("student_name") or request.resume.get("headline") or "Resume"
    _build_generated_resume_pdf(
        export_resume,
        profile,
        str(path),
    )

    safe_role = (
        re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            request.target_role.strip(),
        ).strip("_")
        or "Resume"
    )

    return FileResponse(
        str(path),
        media_type="application/pdf",
        filename=f"{safe_role}_Resume.pdf",
    )


@app.post("/student/me/generate-resume/docx")
def generate_student_resume_docx(
    request: ResumeExportRequest,
    user: dict = Depends(require_student),
):
    """Render an already-generated resume. No Gemini call happens here."""

    if not request.target_role.strip():
        raise HTTPException(status_code=400, detail="Target role is required.")

    path = (
        Path(tempfile.gettempdir())
        / f"role_resume_{uuid.uuid4().hex}.docx"
    )

    profile = get_student(int(user["student_id"]))
    export_resume = dict(request.resume or {})
    export_resume["target_role"] = request.target_role.strip()
    export_resume["headline"] = profile.get("student_name") or request.resume.get("headline") or "Resume"
    _build_generated_resume_docx(
        export_resume,
        profile,
        str(path),
    )

    safe_role = (
        re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            request.target_role.strip(),
        ).strip("_")
        or "Resume"
    )

    return FileResponse(
        str(path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{safe_role}_Resume.docx",
    )

# ==================================================
# PLACEMENT INTELLIGENCE AGENT
# ==================================================
def _local_skill_gap_for_agent(student_id:int,jd:str)->dict:
    conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); required=extract_skills(jd); evidence=_collect_student_skill_evidence(student_id,cur); have=set(evidence.keys()); matched=[s for s in required if s in have]; missing=[s for s in required if s not in have]; pct=round(len(matched)/len(required)*100,2) if required else 0; return {"required_skills":required,"matched_skills":matched,"missing_skills":missing,"skill_match_percent":pct}
    finally: close_db(cur,conn)

@app.post("/students/{student_id}/placement-intelligence",dependencies=[Depends(require_student_or_faculty)])
def placement_intelligence_agent(student_id:int,request:PlacementIntelligenceRequest):
    profile=get_student(student_id); jd=request.job_description.strip(); gap=_local_skill_gap_for_agent(student_id,jd) if jd else None
    try:
        prediction = predict_student(student_id)
    except Exception:
        prediction = {}
    try:
        shap_data = explain_student(student_id)
    except Exception:
        shap_data = {}
    payload={"target_role":request.target_role or "general placement","student":profile,"skill_gap":gap,"prediction":prediction,"shap_explanation":shap_data}
    schema={"type":"object","properties":{"summary":{"type":"string"},"strengths":{"type":"array","items":{"type":"string"}},"priority_gaps":{"type":"array","items":{"type":"string"}},"action_plan":{"type":"array","items":{"type":"string"}},"job_fit_explanation":{"type":"string"}},"required":["summary","strengths","priority_gaps","action_plan","job_fit_explanation"]}
    prompt=f"You are a placement intelligence agent. Use only the supplied verified profile and analysis. Do not invent facts. Give concise personalized guidance for {request.target_role or 'general placement'}. DATA={json.dumps(payload,ensure_ascii=False)}"
    try:
        response=generate_gemini_response("ai",[types.Content(role="user",parts=[types.Part.from_text(text=prompt)])],types.GenerateContentConfig(system_instruction="Use only supplied facts.",max_output_tokens=1400,response_mime_type="application/json",response_json_schema=schema)); result=parse_final_json(response.text)
        return {"agent":"Placement Intelligence Agent","target_role":request.target_role,"result":result,"skill_gap":gap}
    except Exception as error:
        raise HTTPException(status_code=500,detail=f"Placement Intelligence Agent error: {error}")

# ==================================================
# ADD PROJECT
# ==================================================

@app.post("/student/me/projects")
def add_my_project(
    request: ProjectRequest,
    user: dict = Depends(require_student),
):

    if not request.title.strip() or not request.description.strip():
        raise HTTPException(
            status_code=400,
            detail="Project title and description are required.",
        )

    student_id = int(user["student_id"])

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        _ensure_student_detail_tables(connection, cursor)

        cursor.execute(
            """
            INSERT INTO student_projects
                (student_id, title, description, technologies, start_date, end_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING project_id
            """,
            (
                student_id,
                request.title.strip(),
                request.description.strip(),
                request.technologies.strip(),
                request.start_date,
                request.end_date,
            ),
        )

        project_id = int(cursor.fetchone()[0])

        cursor.execute(
            "UPDATE students SET projects_count = projects_count + 1 WHERE student_id = %s",
            (student_id,),
        )

        _refresh_student_skill_index(
            student_id,
            cursor,
        )

        connection.commit()

        return {
            "project_id": project_id,
            "message": "Project added successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Project creation error: {error}",
        )

    finally:
        close_db(cursor, connection)


# ==================================================
# ADD INTERNSHIP
# ==================================================

@app.post("/student/me/internships")
def add_my_internship(
    request: InternshipRequest,
    user: dict = Depends(require_student),
):

    if (
        not request.company_name.strip()
        or not request.role.strip()
        or not request.description.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="Company, role and description are required.",
        )

    student_id = int(user["student_id"])

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        _ensure_student_detail_tables(connection, cursor)

        cursor.execute(
            """
            INSERT INTO student_internships
                (student_id, company_name, role, description, technologies, start_date, end_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING internship_id
            """,
            (
                student_id,
                request.company_name.strip(),
                request.role.strip(),
                request.description.strip(),
                request.technologies.strip(),
                request.start_date,
                request.end_date,
            ),
        )

        internship_id = int(cursor.fetchone()[0])

        cursor.execute(
            "UPDATE students SET internships_count = internships_count + 1 WHERE student_id = %s",
            (student_id,),
        )

        _refresh_student_skill_index(
            student_id,
            cursor,
        )

        connection.commit()

        return {
            "internship_id": internship_id,
            "message": "Internship added successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Internship creation error: {error}",
        )

    finally:
        close_db(cursor, connection)


# ==================================================
# DELETE PROJECT
# ==================================================


@app.put("/student/me/projects/{project_id}")
def update_my_project(
    project_id: int,
    request: ProjectRequest,
    user: dict = Depends(require_student),
):

    if (
        not request.title.strip()
        or not request.description.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Project title and description are required."
            ),
        )

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE student_projects
            SET
                title = %s,
                description = %s,
                technologies = %s,
                start_date = %s,
                end_date = %s
            WHERE project_id = %s
              AND student_id = %s
            RETURNING
                project_id,
                title,
                description,
                technologies,
                start_date,
                end_date
            """,
            (
                request.title.strip(),
                request.description.strip(),
                request.technologies.strip(),
                request.start_date,
                request.end_date,
                project_id,
                int(user["student_id"]),
            ),
        )

        row = cursor.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Project not found.",
            )

        _refresh_student_skill_index(
            int(user["student_id"]),
            cursor,
        )

        connection.commit()

        return {
            "project": {
                "project_id": int(row[0]),
                "title": row[1],
                "description": row[2],
                "technologies": row[3],
                "start_date": row[4],
                "end_date": row[5],
            },
            "message": "Project updated successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:

        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Project update error: {error}",
        )

    finally:

        close_db(
            cursor,
            connection,
        )


@app.delete("/student/me/projects/{project_id}")
def delete_my_project(
    project_id: int,
    user: dict = Depends(require_student),
):

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        _ensure_student_detail_tables(connection, cursor)

        cursor.execute(
            """
            DELETE FROM student_projects
            WHERE project_id = %s
              AND student_id = %s
            """,
            (project_id, int(user["student_id"])),
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Project not found.",
            )

        cursor.execute(
            """
            UPDATE students
            SET projects_count = GREATEST(projects_count - 1, 0)
            WHERE student_id = %s
            """,
            (int(user["student_id"]),),
        )

        _refresh_student_skill_index(
            int(user["student_id"]),
            cursor,
        )

        connection.commit()

        return {
            "message": "Project deleted successfully."
        }

    except HTTPException:
        if connection:
            connection.rollback()
        raise

    except Exception as error:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Project deletion error: {error}",
        )

    finally:
        close_db(cursor, connection)


# ==================================================
# DELETE INTERNSHIP
# ==================================================


@app.put("/student/me/internships/{internship_id}")
def update_my_internship(
    internship_id: int,
    request: InternshipRequest,
    user: dict = Depends(require_student),
):

    if (
        not request.company_name.strip()
        or not request.role.strip()
        or not request.description.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Company, role and description are required."
            ),
        )

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE student_internships
            SET
                company_name = %s,
                role = %s,
                description = %s,
                technologies = %s,
                start_date = %s,
                end_date = %s
            WHERE internship_id = %s
              AND student_id = %s
            RETURNING
                internship_id,
                company_name,
                role,
                description,
                technologies,
                start_date,
                end_date
            """,
            (
                request.company_name.strip(),
                request.role.strip(),
                request.description.strip(),
                request.technologies.strip(),
                request.start_date,
                request.end_date,
                internship_id,
                int(user["student_id"]),
            ),
        )

        row = cursor.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Internship not found.",
            )

        _refresh_student_skill_index(
            int(user["student_id"]),
            cursor,
        )

        connection.commit()

        return {
            "internship": {
                "internship_id": int(row[0]),
                "company_name": row[1],
                "role": row[2],
                "description": row[3],
                "technologies": row[4],
                "start_date": row[5],
                "end_date": row[6],
            },
            "message": "Internship updated successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:

        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Internship update error: {error}",
        )

    finally:

        close_db(
            cursor,
            connection,
        )


@app.delete("/student/me/internships/{internship_id}")
def delete_my_internship(
    internship_id: int,
    user: dict = Depends(require_student),
):

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        _ensure_student_detail_tables(connection, cursor)

        cursor.execute(
            """
            DELETE FROM student_internships
            WHERE internship_id = %s
              AND student_id = %s
            """,
            (internship_id, int(user["student_id"])),
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Internship not found.",
            )

        cursor.execute(
            """
            UPDATE students
            SET internships_count = GREATEST(internships_count - 1, 0)
            WHERE student_id = %s
            """,
            (int(user["student_id"]),),
        )

        _refresh_student_skill_index(
            int(user["student_id"]),
            cursor,
        )

        connection.commit()

        return {
            "message": "Internship deleted successfully."
        }

    except HTTPException:
        if connection:
            connection.rollback()
        raise

    except Exception as error:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Internship deletion error: {error}",
        )

    finally:
        close_db(cursor, connection)


# ==================================================
# PREDICT PLACEMENT READINESS
# ==================================================

@app.post("/predict/{student_id}", dependencies=[Depends(require_student_or_faculty)])
def predict_student(student_id: int):
    return _live_predict_student(student_id)

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                student_id,
                student_name,
                cgpa,
                attendance_percent,
                technical_skill_score,
                aptitude_score,
                communication_score,
                projects_count,
                internships_count,
                certifications_count,
                coding_assessment_score,
                backlogs
            FROM students
            WHERE student_id = %s
            """,
            (student_id,)
        )

        student = cursor.fetchone()

        if student is None:

            raise HTTPException(

                status_code=404,

                detail="Student not found"

            )

        feature_values = [[

            student[2],
            student[3],
            student[4],
            student[5],
            student[6],
            student[7],
            student[8],
            student[9],
            student[10],
            student[11]

        ]]

        input_data = pd.DataFrame(

            feature_values,

            columns=FEATURES

        )

        prediction = model.predict(
            input_data
        )[0]

        probability = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_data
            )[0]

            probability = float(
                max(probabilities)
            )

        cursor.execute(
            """
            INSERT INTO predictions
            (
                student_id,
                predicted_readiness
            )
            VALUES (%s, %s)
            """,
            (
                student_id,
                str(prediction)
            )
        )

        connection.commit()

        return {

            "student_id":
            student[0],

            "student_name":
            student[1],

            "predicted_readiness":
            str(prediction),

            "confidence":
            round(
                probability * 100,
                2
            )
            if probability
            else None

        }

    except HTTPException:

        raise

    except Exception as error:

        if connection:

            connection.rollback()

        raise HTTPException(

            status_code=500,

            detail=f"Prediction error: {str(error)}"

        )

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# K-MEANS STUDENT CLUSTERING
# ==================================================

@app.post("/cluster/{student_id}", dependencies=[Depends(require_student_or_faculty)])
def cluster_student(student_id: int):
    return _live_cluster_student(student_id)

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                cgpa,
                attendance_percent,
                technical_skill_score,
                aptitude_score,
                communication_score,
                projects_count,
                internships_count,
                certifications_count,
                coding_assessment_score,
                backlogs
            FROM students
            WHERE student_id = %s
            """,
            (student_id,)
        )

        student = cursor.fetchone()

        if student is None:

            raise HTTPException(

                status_code=404,

                detail="Student not found"

            )

        student_data = pd.DataFrame(

            [[

                student[0],
                student[1],
                student[2],
                student[3],
                student[4],
                student[5],
                student[6],
                student[7],
                student[8],
                student[9]

            ]],

            columns=FEATURES

        )

        scaled_data = kmeans_scaler.transform(
            student_data
        )

        cluster_number = int(
            kmeans_model.predict(
                scaled_data
            )[0]
        )

        cluster_name = cluster_labels.get(

            cluster_number,

            f"Cluster {cluster_number}"

        )

        return {

            "student_id":
            student_id,

            "cluster_number":
            cluster_number,

            "cluster_name":
            cluster_name

        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=f"Cluster error: {str(error)}"

        )

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# SHAP EXPLAINABILITY
# ==================================================

@app.get("/explain/{student_id}", dependencies=[Depends(require_student_or_faculty)])
def explain_student(student_id: int):
    return _live_explain_student(student_id)

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                student_id,
                student_name,
                cgpa,
                attendance_percent,
                technical_skill_score,
                aptitude_score,
                communication_score,
                projects_count,
                internships_count,
                certifications_count,
                coding_assessment_score,
                backlogs
            FROM students
            WHERE student_id = %s
            """,
            (student_id,)
        )

        student = cursor.fetchone()

        if student is None:

            raise HTTPException(

                status_code=404,

                detail="Student not found"

            )

        input_data = pd.DataFrame(

            [[

                student[2],
                student[3],
                student[4],
                student[5],
                student[6],
                student[7],
                student[8],
                student[9],
                student[10],
                student[11]

            ]],

            columns=FEATURES

        )

        prediction = model.predict(
            input_data
        )[0]

        shap_values = shap_explainer.shap_values(
            input_data
        )

        shap_array = np.asarray(
            shap_values
        )

        if isinstance(
            shap_values,
            list
        ):

            values = np.asarray(
                shap_values[0]
            )[0]

        elif shap_array.ndim == 3:

            values = shap_array[0, :, 0]

        elif shap_array.ndim == 2:

            values = shap_array[0]

        else:

            values = shap_array

        feature_explanations = []

        for index, feature in enumerate(
            FEATURES
        ):

            shap_value = float(
                values[index]
            )

            impact = (

                "Positive"
                if shap_value >= 0
                else "Negative"

            )

            feature_explanations.append({

                "feature":
                feature,

                "value":
                float(
                    input_data.iloc[
                        0,
                        index
                    ]
                ),

                "shap_value":
                round(
                    shap_value,
                    4
                ),

                "impact":
                impact,

                "importance":
                round(
                    abs(shap_value),
                    4
                )

            })

        feature_explanations = sorted(

            feature_explanations,

            key=lambda item:
            item["importance"],

            reverse=True

        )

        label_map = {
            "CGPA": "CGPA",
            "Attendance_Percent": "attendance",
            "Technical_Skill_Score": "technical skill score",
            "Aptitude_Score": "aptitude score",
            "Communication_Score": "communication score",
            "Projects_Count": "projects",
            "Internships_Count": "internships",
            "Certifications_Count": "certifications",
            "Coding_Assessment_Score": "coding assessment score",
            "Backlogs": "backlogs",
        }

        for item in feature_explanations:
            item["label"] = label_map.get(
                item["feature"],
                item["feature"].replace("_", " ").lower(),
            )

        top = feature_explanations[:5]
        supporting = [
            item for item in top
            if item["impact"] == "Positive"
        ][:3]
        reducing = [
            item for item in top
            if item["impact"] == "Negative"
        ][:3]

        def pretty_value(item):
            value = item["value"]
            if item["feature"] == "Attendance_Percent":
                return f"{value:g}%"
            if item["feature"] == "CGPA":
                return f"{value:g}"
            return f"{value:g}"

        support_text = ", ".join(
            f"{item['label']} ({pretty_value(item)})"
            for item in supporting
        )
        reducing_text = ", ".join(
            f"{item['label']} ({pretty_value(item)})"
            for item in reducing
        )

        explanation_parts = [
            f"{student[1]} is currently assessed as "
            f"{str(prediction)} placement readiness."
        ]

        if support_text:
            explanation_parts.append(
                "This is mainly supported by "
                f"{support_text}."
            )

        if reducing_text:
            explanation_parts.append(
                "The main factors holding the score back are "
                f"{reducing_text}."
            )

        # Keep the report explanation short and readable.
        explanation = " ".join(explanation_parts)

        return {
            "student_id": int(student[0]),
            "student_name": student[1],
            "predicted_readiness": str(prediction),
            "explanation": explanation,
            "top_influencing_factors": top,
            "positive_factors": [
                item["label"]
                for item in feature_explanations
                if item["impact"] == "Positive"
            ],
            "negative_factors": [
                item["label"]
                for item in feature_explanations
                if item["impact"] == "Negative"
            ],
            "all_feature_explanations": feature_explanations,
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=f"SHAP explanation error: {str(error)}"

        )

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# EXTRACT SKILLS
# ==================================================

def extract_skills(
    job_description: str
):

    jd_lower = job_description.lower()

    detected_skills = []

    for skill in KNOWN_SKILLS:

        if skill.lower() in jd_lower:

            detected_skills.append(
                skill
            )

    return detected_skills


# ==================================================
# SKILL GAP ANALYSIS
# ==================================================

@app.post("/skill-gap/{student_id}", dependencies=[Depends(require_student_or_faculty)])
def skill_gap_analysis(
    student_id: int,
    request: JobDescriptionRequest,
):
    """Local-first, source-aware skill-gap analysis. No Gemini call."""

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT student_name
            FROM students
            WHERE student_id = %s
            """,
            (student_id,),
        )
        student = cursor.fetchone()

        if student is None:
            raise HTTPException(
                status_code=404,
                detail="Student not found.",
            )

        job_text = request.job_description.strip()
        if not job_text:
            raise HTTPException(
                status_code=400,
                detail="Job description is required.",
            )

        required_skills = extract_skills(job_text)

        if not required_skills:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No supported skills or technologies were detected in "
                    "the job description. Try including tools such as Python, "
                    "SQL, Power BI, AWS, Machine Learning, React, etc."
                ),
            )

        evidence = _collect_student_skill_evidence(
            student_id,
            cursor,
        )

        student_skill_set = set(evidence.keys())

        matched_skills = [
            skill
            for skill in required_skills
            if skill in student_skill_set
        ]

        missing_skills = [
            skill
            for skill in required_skills
            if skill not in student_skill_set
        ]

        skill_match_percent = round(
            (len(matched_skills) / len(required_skills)) * 100,
            2,
        )

        source_summary = {
            skill: sorted(evidence.get(skill, set()))
            for skill in matched_skills
        }

        evidence_strength = {}

        for skill in matched_skills:
            sources = evidence.get(skill, set())

            if len(sources) >= 2:
                evidence_strength[skill] = "Strong evidence"
            elif any(
                source in sources
                for source in (
                    "Profile skills",
                    "Project technologies",
                    "Internship technologies",
                )
            ):
                evidence_strength[skill] = "Direct evidence"
            else:
                evidence_strength[skill] = "Supporting evidence"

        if skill_match_percent >= 80:
            recommendation = (
                "Excellent match. The student's current profile covers most "
                "of the skills requested for this role. Focus on validating "
                "the missing skills and strengthening evidence through projects "
                "or practical experience."
            )
        elif skill_match_percent >= 50:
            recommendation = (
                "Moderate match. The student has a useful foundation for this "
                "role, but should prioritize the missing skills that appear most "
                "important in the job description."
            )
        else:
            recommendation = (
                "Low match at present. The student should build the missing "
                "skills through targeted learning, projects, and practical "
                "experience before applying for similar roles."
            )

        return {
            "student_id": student_id,
            "student_name": student[0],
            "required_skills": required_skills,
            "student_skills": sorted(student_skill_set),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "skill_match_percent": skill_match_percent,
            "skill_sources": source_summary,
            "evidence_strength": evidence_strength,
            "recommendation": recommendation,
            "analysis_method": (
                "Local normalized skill extraction from profile skills, "
                "project titles/descriptions/technologies, internship "
                "roles/descriptions/technologies, and the indexed skill profile."
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Skill-gap analysis error: {error}",
        )

    finally:
        close_db(cursor, connection)


# ==================================================
# DASHBOARD SUMMARY
# ==================================================

@app.get("/dashboard-summary", dependencies=[Depends(require_faculty)])
def dashboard_summary():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM students"
        )

        total_students = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT ROUND(
                AVG(NULLIF(cgpa::text, '')::numeric),
                2
            )
            FROM students
            WHERE NULLIF(cgpa::text, '') IS NOT NULL
            """
        )

        average_cgpa = cursor.fetchone()[0]

        # Attendance is not present in the live Excel dataset, so do not
        # fabricate an aggregate for the faculty dashboard.
        average_attendance = None

        cursor.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE COALESCE(NULLIF(projects_count::text, '')::numeric, 0) > 0),
                COUNT(*) FILTER (WHERE COALESCE(NULLIF(internships_count::text, '')::numeric, 0) > 0),
                COUNT(*) FILTER (WHERE COALESCE(NULLIF(certifications_count::text, '')::numeric, 0) > 0),
                COUNT(*) FILTER (WHERE COALESCE(NULLIF(backlogs::text, '')::numeric, 0) > 0),
                ROUND(AVG(NULLIF(technical_skill_score::text, '')::numeric), 2),
                ROUND(AVG(NULLIF(amcat_score::text, '')::numeric), 2)
            FROM students
            """
        )

        profile_counts = cursor.fetchone()

        cursor.execute(
            "SELECT COUNT(*) FROM predictions"
        )

        total_predictions = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT
                predicted_readiness,
                COUNT(*)
            FROM (
                SELECT DISTINCT ON (student_id)
                    student_id,
                    predicted_readiness
                FROM predictions
                ORDER BY
                    student_id,
                    prediction_date DESC
            ) latest_predictions
            GROUP BY predicted_readiness
            """
        )

        readiness_data = cursor.fetchall()

        readiness_distribution = {

            str(row[0]):
            int(row[1])

            for row in readiness_data

        }

        return {

            "total_students":
            total_students,

            "average_cgpa":
            float(average_cgpa)
            if average_cgpa
            else 0,

            "average_attendance":
            float(average_attendance)
            if average_attendance is not None
            else None,

            "average_technical_skill_score": float(profile_counts[4]) if profile_counts[4] is not None else None,
            "average_amcat_score": float(profile_counts[5]) if profile_counts[5] is not None else None,
            "students_with_internship": int(profile_counts[1]),
            "students_with_projects": int(profile_counts[0]),
            "students_with_certifications": int(profile_counts[2]),
            "students_with_backlogs": int(profile_counts[3]),
            "availability_note": "Attendance, standalone aptitude, communication, and coding assessment are not present in the live dataset.",

            "total_predictions":
            total_predictions,

            "readiness_distribution":
            readiness_distribution

        }

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=f"Dashboard error: {str(error)}"

        )

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# GET STUDENTS BY PLACEMENT READINESS
# ==================================================

@app.get("/readiness-details/{readiness}", dependencies=[Depends(require_faculty)])
def readiness_details(readiness: str):
    """
    Return the latest prediction for every student with the
    requested placement-readiness label.

    This is a direct PostgreSQL endpoint for dashboard drill-down,
    so clicking a readiness slice does not call Gemini.
    """

    requested = readiness.strip()

    if not requested:
        raise HTTPException(
            status_code=400,
            detail="Readiness value cannot be empty.",
        )

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                s.student_id,
                s.student_name,
                s.cgpa,
                s.attendance_percent,
                s.technical_skill_score,
                s.aptitude_score,
                s.communication_score,
                s.projects_count,
                s.internships_count,
                s.certifications_count,
                s.coding_assessment_score,
                s.backlogs,
                latest.predicted_readiness
            FROM students s
            INNER JOIN (
                SELECT DISTINCT ON (student_id)
                    student_id,
                    predicted_readiness
                FROM predictions
                ORDER BY
                    student_id,
                    prediction_date DESC
            ) latest
                ON latest.student_id = s.student_id
            WHERE LOWER(latest.predicted_readiness) = LOWER(%s)
            ORDER BY s.student_id
            """,
            (requested,),
        )

        rows = cursor.fetchall()

        students = []

        for row in rows:
            students.append(
                {
                    "student_id": int(row[0]),
                    "student_name": row[1],
                    "cgpa": _db_float(row[2]),
                    "attendance": _db_float(row[3]),
                    "technical_skills": _db_float(row[4]),
                    "aptitude": _db_float(row[5]),
                    "communication": _db_float(row[6]),
                    "projects": _db_int(row[7]),
                    "internships": _db_int(row[8]),
                    "certifications": _db_int(row[9]),
                    "coding_assessment": _db_float(row[10]),
                    "backlogs": _db_int(row[11]),
                    "predicted_readiness": str(row[12]),
                }
            )

        # Add a concise explanation for each student's current
        # readiness using the existing SHAP explainer. This keeps
        # the dashboard drill-down grounded in the trained model.
        for student in students:
            try:
                feature_row = pd.DataFrame(
                    [[
                        student["cgpa"],
                        student["attendance"],
                        student["technical_skills"],
                        student["aptitude"],
                        student["communication"],
                        student["projects"],
                        student["internships"],
                        student["certifications"],
                        student["coding_assessment"],
                        student["backlogs"],
                    ]],
                    columns=FEATURES,
                )

                shap_values = shap_explainer.shap_values(feature_row)
                shap_array = np.asarray(shap_values)

                if isinstance(shap_values, list):
                    shap_vector = np.asarray(shap_values[0])[0]
                elif shap_array.ndim == 3:
                    shap_vector = shap_array[0, :, 0]
                elif shap_array.ndim == 2:
                    shap_vector = shap_array[0]
                else:
                    shap_vector = shap_array

                feature_labels = {
                    "CGPA": "CGPA",
                    "Attendance_Percent": "attendance",
                    "Technical_Skill_Score": "technical skill score",
                    "Aptitude_Score": "aptitude score",
                    "Communication_Score": "communication score",
                    "Projects_Count": "projects",
                    "Internships_Count": "internships",
                    "Certifications_Count": "certifications",
                    "Coding_Assessment_Score": "coding assessment score",
                    "Backlogs": "backlogs",
                }

                factors = []
                for index, feature in enumerate(FEATURES):
                    shap_value = float(shap_vector[index])
                    actual_value = float(feature_row.iloc[0, index])

                    factors.append({
                        "feature": feature,
                        "label": feature_labels.get(
                            feature,
                            feature.replace("_", " ").lower(),
                        ),
                        "actual_value": actual_value,
                        "shap_value": round(shap_value, 4),
                        "impact": (
                            "supporting"
                            if shap_value >= 0
                            else "reducing"
                        ),
                    })

                factors.sort(
                    key=lambda item: abs(item["shap_value"]),
                    reverse=True,
                )

                supporting = [
                    item
                    for item in factors
                    if item["impact"] == "supporting"
                ][:3]

                reducing = [
                    item
                    for item in factors
                    if item["impact"] == "reducing"
                ][:3]

                readiness_label = student["predicted_readiness"]

                def format_value(label, value):
                    if label == "attendance":
                        return f"{value:g}%"
                    if label == "CGPA":
                        return f"{value:g}"
                    if label in {
                        "technical skill score",
                        "aptitude score",
                        "communication score",
                        "coding assessment score",
                    }:
                        return f"{value:g}/100"
                    return f"{value:g}"

                supporting_text = ", ".join(
                    f"{item['label']} ({format_value(item['label'], item['actual_value'])})"
                    for item in supporting
                )

                reducing_text = ", ".join(
                    f"{item['label']} ({format_value(item['label'], item['actual_value'])})"
                    for item in reducing
                )

                if supporting_text:
                    explanation_text = (
                        f"{student['student_name']} was predicted as "
                        f"**{readiness_label} readiness**. "
                        f"SHAP identifies {supporting_text} as the "
                        f"strongest factors supporting this prediction."
                    )
                else:
                    explanation_text = (
                        f"{student['student_name']} was predicted as "
                        f"**{readiness_label} readiness** based on the "
                        f"combined model features."
                    )

                if reducing_text:
                    explanation_text += (
                        f" The main factors pulling the prediction in the "
                        f"opposite direction were {reducing_text}."
                    )


                student["positive_factors"] = [
                    item["label"] for item in supporting
                ]
                student["negative_factors"] = [
                    item["label"] for item in reducing
                ]
                student["top_factors"] = [
                    {
                        "feature": item["label"],
                        "actual_value": item["actual_value"],
                        "shap_value": item["shap_value"],
                        "impact": item["impact"],
                    }
                    for item in factors[:3]
                ]
                student["shap_explanation"] = explanation_text

            except Exception:
                # Keep the readiness drill-down working even if SHAP
                # cannot be calculated for one student.
                student["positive_factors"] = []
                student["negative_factors"] = []
                student["top_factors"] = []

        return {
            "readiness": requested,
            "count": len(students),
            "students": students,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Readiness details error: {str(error)}",
        )

    finally:
        close_db(cursor, connection)


# ==================================================

# ==================================================
# GET AT-RISK STUDENT DETAILS
# ==================================================

@app.get("/at-risk-details", dependencies=[Depends(require_faculty)])
def at_risk_details():
    return _live_at_risk_details()

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                student_id,
                student_name,
                cgpa,
                attendance_percent,
                technical_skill_score,
                aptitude_score,
                communication_score,
                projects_count,
                internships_count,
                certifications_count,
                coding_assessment_score,
                backlogs
            FROM students
            WHERE
                cgpa < 6.5
                OR attendance_percent < 70
                OR backlogs > 1
            ORDER BY
                cgpa ASC,
                attendance_percent ASC,
                backlogs DESC
            """
        )

        rows = cursor.fetchall()

        students = [
            {
                "student_id": int(row[0]),
                "student_name": row[1],
                "cgpa": _db_float(row[2]),
                "attendance": _db_float(row[3]),
                "technical_skills": _db_float(row[4]),
                "aptitude": _db_float(row[5]),
                "communication": _db_float(row[6]),
                "projects": _db_int(row[7]),
                "internships": _db_int(row[8]),
                "certifications": _db_int(row[9]),
                "coding_assessment": _db_float(row[10]),
                "backlogs": _db_int(row[11]),
            }
            for row in rows
        ]

        return {
            "count": len(students),
            "students": students,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"At-risk details error: {str(error)}",
        )

    finally:
        close_db(cursor, connection)


# AI TOOL:
# GET STUDENT SKILLS
# ==================================================

def tool_get_student_skills(
    student_id: int
):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                s.student_id,
                s.student_name,
                ss.skill_name
            FROM students s
            LEFT JOIN student_skills ss
                ON s.student_id = ss.student_id
            WHERE s.student_id = %s
            ORDER BY ss.skill_name
            """,
            (student_id,)
        )

        rows = cursor.fetchall()

        if not rows:

            return {
                "found": False,
                "message": "Student not found."
            }

        skills = [

            row[2]

            for row in rows

            if row[2]

        ]

        return {

            "found": True,

            "student_id":
            int(rows[0][0]),

            "student_name":
            rows[0][1],

            "skills":
            skills

        }

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# AI TOOL:
# GET STUDENTS BY SKILL
# ==================================================

def tool_get_students_by_skill(
    skill: str
):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                s.student_id,
                s.student_name
            FROM students s
            INNER JOIN student_skills ss
                ON s.student_id = ss.student_id
            WHERE LOWER(ss.skill_name) = LOWER(%s)
            ORDER BY s.student_id
            """,
            (skill,)
        )

        rows = cursor.fetchall()

        students = [

            {
                "student_id":
                int(row[0]),

                "student_name":
                row[1]
            }

            for row in rows

        ]

        cursor.execute(
            "SELECT COUNT(*) FROM students"
        )

        total_students = cursor.fetchone()[0]

        return {

            "skill":
            skill,

            "matching_students":
            students,

            "count":
            len(students),

            "total_students":
            total_students,

            "data": [

                {

                    "name":
                    f"Know {skill}",

                    "value":
                    len(students)

                },

                {

                    "name":
                    f"Do Not Know {skill}",

                    "value":
                    max(
                        total_students -
                        len(students),
                        0
                    )

                }

            ]

        }

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# AI TOOL:
# GET STUDENT METRIC COMPARISON
# ==================================================

def tool_get_metric_comparison(
    metric: str
):

    metric_columns = {

        "cgpa":
        "cgpa",

        "attendance":
        "attendance_percent",

        "technical_skills":
        "technical_skill_score",

        "aptitude":
        "aptitude_score",

        "communication":
        "communication_score",

        "projects":
        "projects_count",

        "internships":
        "internships_count",

        "certifications":
        "certifications_count",

        "coding_assessment":
        "coding_assessment_score",

        "backlogs":
        "backlogs"

    }

    if metric not in metric_columns:

        return {

            "error":
            (
                "Unsupported metric. "
                "Supported metrics are: "
                + ", ".join(
                    metric_columns.keys()
                )
            )

        }

    column = metric_columns[metric]

    # Safe because column comes only
    # from the fixed dictionary above.
    query = f"""
        SELECT
            student_id,
            student_name,
            {column}
        FROM students
        ORDER BY {column} DESC
    """

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(query)

        rows = cursor.fetchall()

        data = []

        for row in rows:

            data.append({

                "student_id":
                int(row[0]),

                "name":
                row[1],

                "value":
                _db_float(row[2])

            })

        average = 0

        if data:

            average = round(

                sum(
                    item["value"]
                    for item in data
                )
                /
                len(data),

                2

            )

        return {

            "metric":
            metric,

            "average":
            average,

            "highest":
            data[0]
            if data
            else None,

            "lowest":
            data[-1]
            if data
            else None,

            "data":
            data

        }

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# AI TOOL:
# FILTER STUDENTS BY METRIC
# ==================================================

def tool_filter_students(
    metric: str,
    operator: str,
    value: float
):

    metric_columns = {

        "cgpa":
        "cgpa",

        "attendance":
        "attendance_percent",

        "technical_skills":
        "technical_skill_score",

        "aptitude":
        "aptitude_score",

        "communication":
        "communication_score",

        "projects":
        "projects_count",

        "internships":
        "internships_count",

        "certifications":
        "certifications_count",

        "coding_assessment":
        "coding_assessment_score",

        "backlogs":
        "backlogs"

    }

    operators = {

        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "eq": "="

    }

    if metric not in metric_columns:

        return {

            "error":
            "Unsupported metric."

        }

    if operator not in operators:

        return {

            "error":
            "Unsupported operator."

        }

    column = metric_columns[metric]

    sql_operator = operators[operator]

    query = f"""
        SELECT
            student_id,
            student_name,
            {column}
        FROM students
        WHERE {column} {sql_operator} %s
        ORDER BY {column}
    """

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            query,
            (value,)
        )

        rows = cursor.fetchall()

        students = [

            {

                "student_id":
                int(row[0]),

                "name":
                row[1],

                "value":
                _db_float(row[2])

            }

            for row in rows

        ]

        return {

            "metric":
            metric,

            "operator":
            operator,

            "threshold":
            value,

            "count":
            len(students),

            "data":
            students

        }

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# AI TOOL:
# GET READINESS DISTRIBUTION
# ==================================================

def tool_get_readiness_distribution():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                predicted_readiness,
                COUNT(*)
            FROM (
                SELECT DISTINCT ON (student_id)
                    student_id,
                    predicted_readiness
                FROM predictions
                ORDER BY
                    student_id,
                    prediction_date DESC
            ) latest_predictions
            GROUP BY predicted_readiness
            ORDER BY predicted_readiness
            """
        )

        rows = cursor.fetchall()

        data = [

            {

                "name":
                str(row[0]),

                "value":
                int(row[1])

            }

            for row in rows

        ]

        return {

            "distribution":
            data,

            "total":
            sum(
                item["value"]
                for item in data
            )

        }

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# AI TOOL:
# GET AT-RISK STUDENTS
# ==================================================

def tool_get_at_risk_students():
    return _live_at_risk_details()

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                student_id,
                student_name,
                cgpa,
                attendance_percent,
                backlogs
            FROM students
            WHERE
                cgpa < 6.5
                OR attendance_percent < 70
                OR backlogs > 1
            ORDER BY
                cgpa ASC,
                attendance_percent ASC
            """
        )

        rows = cursor.fetchall()

        data = [

            {

                "student_id":
                int(row[0]),

                "student_name":
                row[1],

                "cgpa":
                _db_float(row[2]),

                "attendance":
                _db_float(row[3]),

                "backlogs":
                int(row[4])

            }

            for row in rows

        ]

        return {

            "count":
            len(data),

            "students":
            data

        }

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# AI TOOL:
# GET STUDENT PROFILE
# ==================================================

def tool_get_student_profile(
    student_id: int
):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                student_id,
                student_name,
                cgpa,
                attendance_percent,
                technical_skill_score,
                aptitude_score,
                communication_score,
                projects_count,
                internships_count,
                certifications_count,
                coding_assessment_score,
                backlogs
            FROM students
            WHERE student_id = %s
            """,
            (student_id,)
        )

        row = cursor.fetchone()

        if row is None:

            return {

                "found": False,

                "message":
                "Student not found."

            }

        return {

            "found": True,

            "student_id":
            int(row[0]),

            "student_name":
            row[1],

            "cgpa":
            _db_float(row[2]),

            "attendance":
            _db_float(row[3]),

            "technical_skills":
            _db_float(row[4]),

            "aptitude":
            _db_float(row[5]),

            "communication":
            _db_float(row[6]),

            "projects":
            _db_int(row[7]),

            "internships":
            _db_int(row[8]),

            "certifications":
            _db_int(row[9]),

            "coding_assessment":
            _db_float(row[10]),

            "backlogs":
            _db_int(row[11])

        }

    finally:

        close_db(
            cursor,
            connection
        )


# ==================================================
# AI TOOL:
# GET SHAP EXPLANATION
# ==================================================

def tool_get_shap_explanation(
    student_id: int
):

    result = explain_student(
        student_id
    )

    return result


# ==================================================

# ==================================================
# GET STUDENTS BY K-MEANS CLUSTER
# ==================================================

@app.get("/cluster-details/{cluster_name}", dependencies=[Depends(require_faculty)])
def cluster_details(cluster_name: str):
    return _live_cluster_details(cluster_name)

    requested = cluster_name.strip()

    if not requested:
        raise HTTPException(
            status_code=400,
            detail="Cluster name cannot be empty.",
        )

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                student_id,
                student_name,
                cgpa,
                attendance_percent,
                technical_skill_score,
                aptitude_score,
                communication_score,
                projects_count,
                internships_count,
                certifications_count,
                coding_assessment_score,
                backlogs
            FROM students
            ORDER BY student_id
            """
        )

        rows = cursor.fetchall()

        if not rows:
            return {
                "cluster": requested,
                "count": 0,
                "students": [],
            }

        frame = pd.DataFrame(
            [
                [
                    _db_float(row[2]),
                    _db_float(row[3]),
                    _db_float(row[4]),
                    _db_float(row[5]),
                    _db_float(row[6]),
                    _db_int(row[7]),
                    _db_int(row[8]),
                    _db_int(row[9]),
                    _db_float(row[10]),
                    _db_int(row[11]),
                ]
                for row in rows
            ],
            columns=FEATURES,
        )

        scaled = kmeans_scaler.transform(frame)
        assignments = kmeans_model.predict(scaled)

        students = []

        for index, row in enumerate(rows):

            cluster_number = int(assignments[index])

            label = cluster_labels.get(
                cluster_number,
                f"Cluster {cluster_number}",
            )

            if str(label).lower() != requested.lower():
                continue

            students.append(
                {
                    "student_id": int(row[0]),
                    "student_name": row[1],
                    "cgpa": _db_float(row[2]),
                    "attendance": _db_float(row[3]),
                    "technical_skills": _db_float(row[4]),
                    "aptitude": _db_float(row[5]),
                    "communication": _db_float(row[6]),
                    "projects": _db_int(row[7]),
                    "internships": _db_int(row[8]),
                    "certifications": _db_int(row[9]),
                    "coding_assessment": _db_float(row[10]),
                    "backlogs": _db_int(row[11]),
                    "cluster_number": cluster_number,
                    "cluster_name": str(label),
                }
            )

        return {
            "cluster": requested,
            "count": len(students),
            "students": students,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Cluster details error: {str(error)}",
        )

    finally:
        close_db(cursor, connection)


# AI TOOL:
# GET CLUSTER DISTRIBUTION
# ==================================================

def tool_get_cluster_distribution():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                student_id,
                student_name,
                cgpa,
                attendance_percent,
                technical_skill_score,
                aptitude_score,
                communication_score,
                projects_count,
                internships_count,
                certifications_count,
                coding_assessment_score,
                backlogs
            FROM students
            ORDER BY student_id
            """
        )

        rows = cursor.fetchall()

        if not rows:

            return {

                "distribution": [],

                "students": []

            }

        cluster_data = pd.DataFrame(

            [

                [

                    _db_float(row[2]),
                    _db_float(row[3]),
                    _db_float(row[4]),
                    _db_float(row[5]),
                    _db_float(row[6]),
                    _db_int(row[7]),
                    _db_int(row[8]),
                    _db_int(row[9]),
                    _db_float(row[10]),
                    _db_int(row[11])

                ]

                for row in rows

            ],

            columns=FEATURES

        )

        scaled = kmeans_scaler.transform(
            cluster_data
        )

        clusters = kmeans_model.predict(
            scaled
        )

        students = []

        distribution = {}

        for index, row in enumerate(rows):

            cluster_number = int(
                clusters[index]
            )

            cluster_name = cluster_labels.get(

                cluster_number,

                f"Cluster {cluster_number}"

            )

            students.append({

                "student_id":
                int(row[0]),

                "student_name":
                row[1],

                "cluster":
                cluster_name

            })

            distribution[cluster_name] = (

                distribution.get(
                    cluster_name,
                    0
                ) + 1

            )

        distribution_data = [

            {

                "name":
                name,

                "value":
                count

            }

            for name, count
            in distribution.items()

        ]

        return {

            "distribution":
            distribution_data,

            "students":
            students

        }

    finally:

        close_db(
            cursor,
            connection
        )



# ==================================================
# AI TOOL:
# GENERIC STUDENT ANALYTICS QUERY
# ==================================================

def tool_query_student_data(
    filters=None,
    select_metrics=None,
    aggregation="rows",
    order_by="student_id",
    order_direction="asc",
    limit=20,
):
    """
    Flexible, safe analytics tool for natural-language student queries.

    Examples handled by Gemini:
      - CGPA above 8 and attendance above 75
      - top 5 students by technical skills
      - average CGPA
      - students with at least 2 projects
      - compare communication scores
    """

    # Safe mapping from LLM-friendly names to real DB columns.
    metric_columns = {
        "cgpa": "cgpa",
        "attendance": "attendance_percent",
        "technical_skills": "technical_skill_score",
        "aptitude": "aptitude_score",
        "communication": "communication_score",
        "projects": "projects_count",
        "internships": "internships_count",
        "certifications": "certifications_count",
        "coding_assessment": "coding_assessment_score",
        "backlogs": "backlogs",
    }

    operator_map = {
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "eq": "=",
    }

    aggregation = str(aggregation or "rows").lower()
    order_direction = str(order_direction or "asc").lower()

    if aggregation not in {"rows", "avg", "count", "max", "min"}:
        aggregation = "rows"

    if order_direction not in {"asc", "desc"}:
        order_direction = "asc"

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 20

    limit = max(1, min(limit, 100))

    filters = filters or []
    select_metrics = select_metrics or []

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        where_clauses = []
        params = []

        # Build safe parameterized filters.
        for item in filters:
            metric = str(item.get("metric", "")).lower()
            operator = str(item.get("operator", "eq")).lower()
            value = item.get("value")

            if metric not in metric_columns:
                continue

            if operator not in operator_map:
                continue

            column = metric_columns[metric]

            where_clauses.append(
                f"{column} {operator_map[operator]} %s"
            )

            params.append(value)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # --------------------------------------------------
        # Aggregated analytics
        # --------------------------------------------------
        if aggregation in {"avg", "count", "max", "min"}:
            metric = (
                select_metrics[0]
                if select_metrics
                else "cgpa"
            )
            metric = str(metric).lower()

            if metric not in metric_columns:
                metric = "cgpa"

            column = metric_columns[metric]

            aggregate_sql = {
                "avg": f"AVG({column})",
                "count": "COUNT(*)",
                "max": f"MAX({column})",
                "min": f"MIN({column})",
            }[aggregation]

            query = f"""
                SELECT {aggregate_sql}
                FROM students
                {where_sql}
            """

            cursor.execute(query, tuple(params))
            value = cursor.fetchone()[0]

            numeric_value = (
                float(value)
                if value is not None
                else 0
            )

            return {
                "query_type": aggregation,
                "metric": metric,
                "value": round(numeric_value, 2),
                "filters": filters,
            }

        # --------------------------------------------------
        # Row-based analytics
        # --------------------------------------------------
        selected = [
            str(metric).lower()
            for metric in select_metrics
            if str(metric).lower() in metric_columns
        ]

        # Always include student identity.
        if not selected:
            selected = [
                "cgpa",
                "attendance",
                "technical_skills",
            ]

        select_sql = [
            "student_id",
            "student_name",
        ]

        for metric in selected:
            select_sql.append(
                f"{metric_columns[metric]} AS {metric}"
            )

        safe_order = str(order_by or "student_id").lower()

        if safe_order not in metric_columns and safe_order not in {
            "student_id",
            "student_name",
        }:
            safe_order = "student_id"

        order_sql = (
            "student_name"
            if safe_order == "student_name"
            else (
                metric_columns[safe_order]
                if safe_order in metric_columns
                else "student_id"
            )
        )

        query = f"""
            SELECT
                {", ".join(select_sql)}
            FROM students
            {where_sql}
            ORDER BY {order_sql} {order_direction.upper()}
            LIMIT %s
        """

        params_with_limit = list(params)
        params_with_limit.append(limit)

        cursor.execute(
            query,
            tuple(params_with_limit)
        )

        rows = cursor.fetchall()
        column_names = [
            desc[0]
            for desc in cursor.description
        ]

        data = []
        for row in rows:
            record = dict(zip(column_names, row))

            normalized = {}
            for key, value in record.items():
                if isinstance(value, (np.integer,)):
                    normalized[key] = int(value)
                elif isinstance(value, (np.floating,)):
                    normalized[key] = float(value)
                else:
                    normalized[key] = value

            data.append(normalized)

        return {
            "query_type": "rows",
            "filters": filters,
            "metrics": selected,
            "order_by": safe_order,
            "order_direction": order_direction,
            "limit": limit,
            "count": len(data),
            "data": data,
        }

    finally:
        close_db(cursor, connection)


# ==================================================
# FORMAT GENERIC ANALYTICS RESULT
# ==================================================

def build_generic_data_response(result):
    """Create a frontend-ready response without a second Gemini call."""

    if not isinstance(result, dict):
        return {
            "type": "data",
            "answer": str(result),
            "chart_type": "none",
            "title": "Student Data",
            "data": [],
        }

    query_type = result.get("query_type")

    # ---------- Aggregation ----------
    if query_type in {"avg", "count", "max", "min"}:
        metric = result.get("metric", "metric")
        value = result.get("value", 0)

        labels = {
            "avg": "Average",
            "count": "Count",
            "max": "Maximum",
            "min": "Minimum",
        }

        answer = (
            f"**{labels.get(query_type, query_type)} "
            f"{metric.replace('_', ' ')}:** {value}"
        )

        return {
            "type": "data",
            "answer": answer,
            "chart_type": "none",
            "title": labels.get(query_type, "Result"),
            "data": [],
        }

    # ---------- Row results ----------
    data = result.get("data", [])

    if not data:
        return {
            "type": "data",
            "answer": "No students matched the requested criteria.",
            "chart_type": "none",
            "title": "Student Data",
            "data": [],
        }

    metrics = result.get("metrics", [])

    # Create a useful answer from the first few rows.
    answer_lines = [
        f"Found **{result.get('count', len(data))} student(s)**."
    ]

    preview = data[:10]

    for row in preview:
        name = row.get("student_name", "Student")
        values = []

        for metric in metrics:
            if metric in row:
                display_name = metric.replace("_", " ").title()
                values.append(
                    f"{display_name}: {row[metric]}"
                )

        if values:
            answer_lines.append(
                f"- **{name}** — " + ", ".join(values)
            )
        else:
            answer_lines.append(
                f"- **{name}**"
            )

    if len(data) > 10:
        answer_lines.append(
            f"\nShowing the first 10 of {len(data)} results."
        )

    # Choose the first metric as chart value when suitable.
    chart_data = []

    if metrics:
        chart_metric = metrics[0]
        for row in data:
            value = row.get(chart_metric)
            if value is not None:
                chart_data.append({
                    "name": row.get(
                        "student_name",
                        "Student"
                    ),
                    "value": value,
                })

    chart_type = "bar" if chart_data else "table"

    title = (
        f"{metrics[0].replace('_', ' ').title()} Results"
        if metrics
        else "Student Results"
    )

    return {
        "type": "data",
        "answer": "\n".join(answer_lines),
        "chart_type": chart_type,
        "title": title,
        "data": chart_data if chart_data else data,
    }



# ==================================================
# GEMINI TOOL DECLARATIONS
# ==================================================


    types.FunctionDeclaration(
        name="query_student_data",
        description=(
            "Flexible student analytics query. Use this for natural-language "
            "questions involving multiple conditions, averages, counts, "
            "top/bottom students, sorting, or selecting one or more student "
            "performance metrics. Use AND logic for multiple filters. "
            "Examples: students with CGPA above 8 and attendance above 75; "
            "top 5 students by technical skills; average CGPA; students "
            "with at least 2 projects."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "enum": [
                                    "cgpa",
                                    "attendance",
                                    "technical_skills",
                                    "aptitude",
                                    "communication",
                                    "projects",
                                    "internships",
                                    "certifications",
                                    "coding_assessment",
                                    "backlogs",
                                ],
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "eq",
                                ],
                            },
                            "value": {
                                "type": "number",
                            },
                        },
                        "required": [
                            "metric",
                            "operator",
                            "value",
                        ],
                    },
                },
                "select_metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "cgpa",
                            "attendance",
                            "technical_skills",
                            "aptitude",
                            "communication",
                            "projects",
                            "internships",
                            "certifications",
                            "coding_assessment",
                            "backlogs",
                        ],
                    },
                },
                "aggregation": {
                    "type": "string",
                    "enum": [
                        "rows",
                        "avg",
                        "count",
                        "max",
                        "min",
                    ],
                },
                "order_by": {
                    "type": "string",
                    "enum": [
                        "student_id",
                        "student_name",
                        "cgpa",
                        "attendance",
                        "technical_skills",
                        "aptitude",
                        "communication",
                        "projects",
                        "internships",
                        "certifications",
                        "coding_assessment",
                        "backlogs",
                    ],
                },
                "order_direction": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                },
            },
        },
    ),

TOOL_DECLARATIONS = [

    types.FunctionDeclaration(

        name="get_student_profile",

        description=(
            "Get the complete academic and placement-related "
            "profile of a specific student by student ID."
        ),

        parameters_json_schema={
            "type": "object",
            "properties": {
                "student_id": {
                    "type": "integer",
                    "description":
                    "Student ID."
                }
            },
            "required": [
                "student_id"
            ]
        }

    ),

    types.FunctionDeclaration(

        name="get_student_skills",

        description=(
            "Get all recorded skills for a specific student."
        ),

        parameters_json_schema={
            "type": "object",
            "properties": {
                "student_id": {
                    "type": "integer",
                    "description":
                    "Student ID."
                }
            },
            "required": [
                "student_id"
            ]
        }

    ),

    types.FunctionDeclaration(

        name="get_students_by_skill",

        description=(
            "Find students who have a specific recorded skill. "
            "Use this for questions such as how many students "
            "know Python, Java, React, SQL, HTML, etc."
        ),

        parameters_json_schema={
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description":
                    "Skill name, for example Python, Java, SQL or React."
                }
            },
            "required": [
                "skill"
            ]
        }

    ),

    types.FunctionDeclaration(

        name="get_metric_comparison",

        description=(
            "Compare a student metric across all students. "
            "Useful for CGPA, attendance, technical skills, aptitude, "
            "communication, projects, internships, certifications, "
            "coding assessment and backlogs."
        ),

        parameters_json_schema={
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": [
                        "cgpa",
                        "attendance",
                        "technical_skills",
                        "aptitude",
                        "communication",
                        "projects",
                        "internships",
                        "certifications",
                        "coding_assessment",
                        "backlogs"
                    ],
                    "description":
                    "Metric to compare."
                }
            },
            "required": [
                "metric"
            ]
        }

    ),

    types.FunctionDeclaration(

        name="filter_students",

        description=(
            "Filter students using a measurable student metric and "
            "a comparison operator. Use for questions such as "
            "students with CGPA above 8, attendance below 75, "
            "or students with at least 3 projects."
        ),

        parameters_json_schema={
            "type": "object",
            "properties": {

                "metric": {
                    "type": "string",
                    "enum": [
                        "cgpa",
                        "attendance",
                        "technical_skills",
                        "aptitude",
                        "communication",
                        "projects",
                        "internships",
                        "certifications",
                        "coding_assessment",
                        "backlogs"
                    ]
                },

                "operator": {
                    "type": "string",
                    "enum": [
                        "gt",
                        "gte",
                        "lt",
                        "lte",
                        "eq"
                    ]
                },

                "value": {
                    "type": "number"
                }

            },

            "required": [
                "metric",
                "operator",
                "value"
            ]

        }

    ),

    types.FunctionDeclaration(

        name="get_readiness_distribution",

        description=(
            "Get the latest placement-readiness distribution "
            "across students."
        ),

        parameters_json_schema={
            "type": "object",
            "properties": {}
        }

    ),

    types.FunctionDeclaration(

        name="get_at_risk_students",

        description=(
            "Get students currently considered at risk based on "
            "low CGPA, low attendance or multiple backlogs."
        ),

        parameters_json_schema={
            "type": "object",
            "properties": {}
        }

    ),

    types.FunctionDeclaration(

        name="get_shap_explanation",

        description=(
            "Explain why a student's Random Forest placement "
            "readiness prediction was produced using SHAP."
        ),

        parameters_json_schema={
            "type": "object",
            "properties": {
                "student_id": {
                    "type": "integer"
                }
            },
            "required": [
                "student_id"
            ]
        }

    ),

    types.FunctionDeclaration(

        name="get_cluster_distribution",

        description=(
            "Get K-Means student cluster distribution and "
            "cluster membership."
        ),

        parameters_json_schema={
            "type": "object",
            "properties": {}
        }

    )

]


# ==================================================
# GEMINI TOOL CONFIGURATION
# ==================================================

GEMINI_TOOLS = [

    types.Tool(

        function_declarations=
        TOOL_DECLARATIONS

    )

]


# ==================================================
# EXECUTE GEMINI TOOL CALL
# ==================================================

def execute_tool(
    name,
    arguments
):

    if name == "get_student_profile":

        return tool_get_student_profile(
            int(
                arguments["student_id"]
            )
        )

    if name == "get_student_skills":

        return tool_get_student_skills(
            int(
                arguments["student_id"]
            )
        )

    if name == "get_students_by_skill":

        return tool_get_students_by_skill(
            str(
                arguments["skill"]
            )
        )

    if name == "get_metric_comparison":

        return tool_get_metric_comparison(
            str(
                arguments["metric"]
            )
        )

    if name == "filter_students":

        return tool_filter_students(

            str(
                arguments["metric"]
            ),

            str(
                arguments["operator"]
            ),

            float(
                arguments["value"]
            )

        )

    if name == "get_readiness_distribution":

        return tool_get_readiness_distribution()

    if name == "get_at_risk_students":

        return tool_get_at_risk_students()

    if name == "get_shap_explanation":

        return tool_get_shap_explanation(
            int(
                arguments["student_id"]
            )
        )

    if name == "get_cluster_distribution":

        return tool_get_cluster_distribution()

    if name == "query_student_data":

        return tool_query_student_data(
            filters=arguments.get("filters"),
            select_metrics=arguments.get("select_metrics"),
            aggregation=arguments.get("aggregation", "rows"),
            order_by=arguments.get("order_by", "student_id"),
            order_direction=arguments.get("order_direction", "asc"),
            limit=arguments.get("limit", 20),
        )

    return {

        "error":
        f"Unknown tool: {name}"

    }


# ==================================================
# DUAL GEMINI ASSISTANT
# ==================================================

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["ai_advice"],
        },
        "answer": {"type": "string"},
        "chart_type": {
            "type": "string",
            "enum": ["none"],
        },
        "title": {"type": "string"},
        "data": {
            "type": "array",
            "items": {"type": "object"},
        },
    },
    "required": [
        "type",
        "answer",
        "chart_type",
        "title",
        "data",
    ],
}


AI_ASSISTANT_INSTRUCTION = """
You are the general AI assistant for an AI Student Placement System.

Answer open-ended questions about:
career guidance, learning roadmaps, Python, Java, React, HTML/CSS,
cloud, data science, AI/ML, projects, interviews and placement
preparation.

Do not use the student database.
Do not invent student-specific statistics.

Give practical, structured advice.
Use clean Markdown with headings and bullets when useful.
Do not use unnecessary long explanations or separators.

Return JSON matching the configured response schema.
"""


DATA_ASSISTANT_INSTRUCTION = """
You are the natural-language data assistant for a Student Placement
Readiness and Analytics System.

You must use the available tools for real student/database questions.

Examples:
- Show students above 8 CGPA.
- How many students know Python?
- Compare attendance.
- Who has the highest technical score?
- Which students are at risk?
- Explain student 3's placement-readiness factors.
- What skills does student 2 have?

Do not invent database data.
Do not generate arbitrary SQL.
Select the safest and most relevant tool.

Prefer query_student_data for questions involving:
- multiple conditions
- top/bottom N students
- sorting/ranking
- averages/counts/min/max
- multiple student metrics
- natural-language comparisons

Use the more specific tools when the question is clearly about a
single student, a skill lookup, SHAP explanation, readiness distribution,
or cluster distribution.

If the user asks a general career/learning question, this endpoint
is not the right assistant; the frontend should use /ai-chat.
"""


def parse_final_json(text):
    """
    Parse Gemini's structured JSON response into a dict.

    Tries, in order:
      1. Direct json.loads on the cleaned text.
      2. json.loads on the substring between the first '{' and the
         last '}' (handles stray text/fences around valid JSON).
      3. Regex extraction of the "answer" field's value out of a
         truncated/malformed JSON blob, so a cut-off response still
         renders as readable Markdown instead of raw JSON.
      4. As an absolute last resort, the raw text - but only if it
         does not look like a JSON object (avoids ever dumping a
         broken '{"type": ...}' blob into the answer field).
    """

    if not text:
        return {
            "type": "ai_advice",
            "answer": "I could not generate a response.",
            "chart_type": "none",
            "title": "AI Assistant",
            "data": [],
        }

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Attempt 1: direct parse.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: parse only the {...} span, in case of stray
    # leading/trailing text around an otherwise valid JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Attempt 3: the JSON was likely truncated mid-string (hit
    # max_output_tokens). Pull out whatever "answer" text exists so
    # far via regex instead of surfacing the broken JSON to the user.
    answer_match = re.search(
        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)',
        cleaned,
        re.DOTALL,
    )

    if answer_match:
        recovered = answer_match.group(1)
        # Unescape common JSON string escapes.
        recovered = (
            recovered.replace('\\n', '\n')
            .replace('\\"', '"')
            .replace('\\t', '\t')
            .replace('\\\\', '\\')
        )
        return {
            "type": "ai_advice",
            "answer": recovered.strip(),
            "chart_type": "none",
            "title": "AI Assistant",
            "data": [],
        }

    # Attempt 4: fall back to the raw text, but only when it does not
    # look like an (unparseable) JSON object - never show raw JSON.
    if cleaned.startswith("{") and '"answer"' in cleaned:
        safe_text = (
            "I generated a response but it was cut off before "
            "completing. Please try asking again, or ask for a "
            "shorter answer."
        )
    else:
        safe_text = cleaned

    return {
        "type": "ai_advice",
        "answer": safe_text,
        "chart_type": "none",
        "title": "AI Assistant",
        "data": [],
    }


def normalize_ai_response(result):

    if not isinstance(result, dict):
        result = {}

    chart_type = result.get("chart_type", "none")

    if chart_type not in {
        "bar",
        "pie",
        "line",
        "radar",
        "table",
        "none",
    }:
        chart_type = "none"

    data = result.get("data", [])

    if not isinstance(data, list):
        data = []

    return {
        "type": result.get("type", "ai_advice"),
        "answer": result.get(
            "answer",
            "I could not generate a response.",
        ),
        "chart_type": chart_type,
        "title": result.get("title", ""),
        "data": data,
    }


def build_ai_config():

    return types.GenerateContentConfig(
        system_instruction=AI_ASSISTANT_INSTRUCTION,
        max_output_tokens=2000,
        response_mime_type="application/json",
        response_json_schema=FINAL_RESPONSE_SCHEMA,
    )


def build_data_config():

    return types.GenerateContentConfig(
        system_instruction=DATA_ASSISTANT_INSTRUCTION,
        tools=GEMINI_TOOLS,
        max_output_tokens=500,
        automatic_function_calling=(
            types.AutomaticFunctionCallingConfig(
                disable=True
            )
        ),
    )


def build_data_response(
    tool_name,
    arguments,
    result,
):
    """
    Format the database result locally.
    No second Gemini call is needed.
    """

    if not isinstance(result, dict):
        return {
            "type": "data",
            "answer": str(result),
            "chart_type": "none",
            "title": "Student Data",
            "data": [],
        }

    # ---------- skill lookup ----------
    if tool_name == "get_students_by_skill":

        skill = arguments.get("skill", "skill")
        count = result.get("count", 0)
        students = result.get("matching_students", [])

        answer = (
            f"**{count} student(s)** have **{skill}** recorded."
        )

        if students:
            answer += "\n\n" + "\n".join(
                f"- {s['student_name']}"
                for s in students
            )

        return {
            "type": "data",
            "answer": answer,
            "chart_type": "pie",
            "title": f"{skill} Skill Distribution",
            "data": result.get("data", []),
        }

    # ---------- filter ----------
    if tool_name == "filter_students":

        metric = result.get("metric", "metric")
        operator = result.get("operator", "gte")
        threshold = result.get("threshold")
        students = result.get("data", [])

        words = {
            "gt": "greater than",
            "gte": "greater than or equal to",
            "lt": "less than",
            "lte": "less than or equal to",
            "eq": "equal to",
        }

        condition = words.get(operator, operator)

        answer = (
            f"Found **{len(students)} student(s)** with "
            f"**{metric.replace('_', ' ')} {condition} {threshold}**."
        )

        if students:
            answer += "\n\n" + "\n".join(
                f"- **{s['name']}** — {s['value']}"
                for s in students
            )

        return {
            "type": "data",
            "answer": answer,
            "chart_type": "bar",
            "title": f"Students by {metric.replace('_', ' ').title()}",
            "data": students,
        }

    # ---------- metric comparison ----------
    if tool_name == "get_metric_comparison":

        metric = result.get("metric", "metric")
        data = result.get("data", [])

        answer = (
            f"### {metric.replace('_', ' ').title()}\n\n"
            f"- **Average:** {result.get('average', 0)}"
        )

        highest = result.get("highest")
        if highest:
            answer += (
                f"\n- **Highest:** {highest['name']} "
                f"({highest['value']})"
            )

        if data:
            answer += "\n\n" + "\n".join(
                f"- {s['name']}: {s['value']}"
                for s in data
            )

        return {
            "type": "data",
            "answer": answer,
            "chart_type": "bar",
            "title": f"{metric.replace('_', ' ').title()} Comparison",
            "data": data,
        }

    # ---------- readiness ----------
    if tool_name == "get_readiness_distribution":

        data = result.get("distribution", [])

        answer = "### Placement Readiness Distribution\n\n"

        if data:
            answer += "\n".join(
                f"- **{item['name']}**: {item['value']}"
                for item in data
            )
        else:
            answer = "No placement-readiness predictions are available."

        return {
            "type": "data",
            "answer": answer,
            "chart_type": "pie",
            "title": "Placement Readiness Distribution",
            "data": data,
        }

    # ---------- at risk ----------
    if tool_name == "get_at_risk_students":

        students = result.get("students", [])

        answer = (
            f"### At-Risk Students\n\n"
            f"Found **{len(students)} student(s)**."
        )

        if students:
            answer += "\n\n" + "\n".join(
                f"- **{s['student_name']}** — "
                f"CGPA {s['cgpa']}, "
                f"Attendance {s['attendance']}%, "
                f"Backlogs {s['backlogs']}"
                for s in students
            )

        return {
            "type": "data",
            "answer": answer,
            "chart_type": "bar" if students else "none",
            "title": "At-Risk Students",
            "data": [
                {
                    "name": s["student_name"],
                    "value": s["cgpa"],
                }
                for s in students
            ],
        }

    # ---------- student skills ----------
    if tool_name == "get_student_skills":

        name = result.get("student_name", "Student")
        skills = result.get("skills", [])

        answer = f"### {name}'s Skills\n\n"

        if skills:
            answer += "\n".join(
                f"- {skill}"
                for skill in skills
            )
        else:
            answer += "No skills are currently recorded."

        return {
            "type": "data",
            "answer": answer,
            "chart_type": "table" if skills else "none",
            "title": f"{name} Skills",
            "data": [
                {
                    "name": skill,
                    "value": 1,
                }
                for skill in skills
            ],
        }

    # ---------- profile ----------
    if tool_name == "get_student_profile":

        if not result.get("found"):
            return {
                "type": "data",
                "answer": result.get(
                    "message",
                    "Student not found.",
                ),
                "chart_type": "none",
                "title": "Student Profile",
                "data": [],
            }

        name = result["student_name"]

        answer = (
            f"### {name}\n\n"
            f"- **CGPA:** {result['cgpa']}\n"
            f"- **Attendance:** {result['attendance']}%\n"
            f"- **Technical Skills:** {result['technical_skills']}\n"
            f"- **Aptitude:** {result['aptitude']}\n"
            f"- **Communication:** {result['communication']}\n"
            f"- **Projects:** {result['projects']}\n"
            f"- **Internships:** {result['internships']}\n"
            f"- **Certifications:** {result['certifications']}\n"
            f"- **Coding Assessment:** {result['coding_assessment']}\n"
            f"- **Backlogs:** {result['backlogs']}"
        )

        return {
            "type": "data",
            "answer": answer,
            "chart_type": "table",
            "title": f"{name} Profile",
            "data": [
                {"name": "CGPA", "value": result["cgpa"]},
                {"name": "Attendance", "value": result["attendance"]},
                {
                    "name": "Technical Skills",
                    "value": result["technical_skills"],
                },
                {"name": "Aptitude", "value": result["aptitude"]},
                {
                    "name": "Communication",
                    "value": result["communication"],
                },
            ],
        }

    # ---------- SHAP ----------
    if tool_name == "get_shap_explanation":

        name = result.get(
            "student_name",
            "Student",
        )

        explanation = result.get(
            "explanation",
            f"{name}'s placement readiness was calculated "
            "based on the student\'s placement-readiness model and the strongest factors influencing the prediction.",
        )

        factors = result.get(
            "top_influencing_factors",
            [],
        )

        return {
            "type": "data",
            "answer": explanation,
            "explanation": explanation,
            "chart_type": "bar" if factors else "none",
            "title": f"Why {name} received this placement prediction",
            "data": [
                {
                    "name": f.get(
                        "label",
                        f.get("feature", "Factor"),
                    ),
                    "value": abs(
                        float(
                            f.get("shap_value", 0)
                        )
                    ),
                }
                for f in factors
            ],
        }

    # ---------- clusters ----------

    if tool_name == "get_cluster_distribution":

        data = result.get("distribution", [])

        answer = (
            "### Student Cluster Distribution\n\n"
            + (
                "\n".join(
                    f"- **{x['name']}**: {x['value']}"
                    for x in data
                )
                if data
                else "No cluster data is available."
            )
        )

        return {
            "type": "data",
            "answer": answer,
            "chart_type": "pie",
            "title": "Student Cluster Distribution",
            "data": data,
        }

    return {
        "type": "data",
        "answer": (
            "The database query completed successfully."
        ),
        "chart_type": "none",
        "title": "Student Data",
        "data": result.get("data", []),
    }



# ==================================================
# LOCAL NLU FOR DATA ASSISTANT
# ==================================================

LOCAL_METRIC_ALIASES = {
    "cgpa": ["cgpa", "gpa"],
    "attendance": ["attendance", "attendance percentage", "attendance percent"],
    "technical_skills": [
        "technical skills",
        "technical skill",
        "technical score",
        "technical"
    ],
    "aptitude": ["aptitude", "aptitude score"],
    "communication": ["communication", "communication score"],
    "projects": ["projects", "project", "project count", "projects count"],
    "internships": ["internships", "internship", "internship count", "internships count"],
    "certifications": ["certifications", "certification", "certification count"],
    "coding_assessment": [
        "coding assessment",
        "coding test",
        "coding score",
        "coding"
    ],
    "backlogs": ["backlogs", "backlog"],
}

LOCAL_OPERATOR_PATTERNS = [
    ("gte", r"(?:at least|greater than or equal to|more than or equal to|not less than)\s*(\d+(?:\.\d+)?)"),
    ("lte", r"(?:at most|less than or equal to|not more than)\s*(\d+(?:\.\d+)?)"),
    ("gt", r"(?:greater than|more than|above|over|higher than)\s*(\d+(?:\.\d+)?)"),
    ("lt", r"(?:less than|below|under|lower than)\s*(\d+(?:\.\d+)?)"),
    ("eq", r"(?:equal to|equals|equal)\s*(\d+(?:\.\d+)?)"),
]


def normalize_local_query(text: str) -> str:
    """Normalize common natural-language variations for fast local parsing."""
    text = text.lower().strip()
    text = text.replace(">=", " greater than or equal to ")
    text = text.replace("<=", " less than or equal to ")
    text = text.replace(">", " greater than ")
    text = text.replace("<", " less than ")
    text = re.sub(r"\s+", " ", text)
    return text


def _detect_local_metrics(text: str):
    """Return unique known metrics mentioned in the query."""
    found = []
    for metric, aliases in LOCAL_METRIC_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                found.append(metric)
                break
    return found


def _detect_local_skill(text: str):
    """Detect a skill only when the query is clearly skill-oriented."""
    skill_context = any(
        phrase in text
        for phrase in (
            "know",
            "knows",
            "have",
            "has",
            "skill",
            "skills",
            "students with",
            "students who",
            "learned",
            "learn",
        )
    )

    if not skill_context:
        return None

    for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(skill.lower())}\b", text):
            return skill

    return None


def _extract_local_condition(text: str, metric: str):
    """Extract metric/operator/value from natural language."""
    aliases = sorted(
        LOCAL_METRIC_ALIASES[metric],
        key=len,
        reverse=True,
    )
    metric_regex = "(?:" + "|".join(
        re.escape(alias) for alias in aliases
    ) + ")"

    # operator -> number -> metric
    for operator, pattern in LOCAL_OPERATOR_PATTERNS:
        match = re.search(
            rf"{pattern}\s*(?:percent|percentage)?\s*{metric_regex}",
            text,
        )
        if match:
            return {
                "metric": metric,
                "operator": operator,
                "value": float(match.group(1)),
            }

    # metric -> operator -> number
    for operator, pattern in LOCAL_OPERATOR_PATTERNS:
        match = re.search(
            rf"{metric_regex}\s*{pattern}",
            text,
        )
        if match:
            return {
                "metric": metric,
                "operator": operator,
                "value": float(match.group(1)),
            }

    return None


def _extract_local_limit(text: str):
    match = re.search(
        r"\b(?:top|bottom|first|last)\s+(\d+)\b",
        text,
    )
    if match:
        return max(1, min(int(match.group(1)), 100))
    return None


def _extract_local_order(text: str, metrics):
    """Return (metric, direction) for ranking queries."""
    if not metrics:
        return None, None

    direction = "desc"
    if any(
        phrase in text
        for phrase in (
            "lowest",
            "smallest",
            "bottom",
            "least",
            "lowest score",
        )
    ):
        direction = "asc"

    order_metric = None
    match = re.search(
        r"(?:by|according to|based on)\s+(.+)$",
        text,
    )
    tail = match.group(1) if match else text

    for metric in metrics:
        if any(
            re.search(rf"\b{re.escape(alias)}\b", tail)
            for alias in LOCAL_METRIC_ALIASES[metric]
        ):
            order_metric = metric
            break

    if order_metric is None:
        order_metric = metrics[0]

    return order_metric, direction


def parse_local_nlu(message: str):
    """
    Parse common student-data questions locally.

    Returns a structured request or None when the query is not
    confidently understood. No Gemini call is made here.
    """
    text = normalize_local_query(message)

    if not text:
        return None

    # -----------------------------------------------
    # Placement readiness distribution
    # -----------------------------------------------
    if (
        "readiness distribution" in text
        or "placement readiness distribution" in text
        or "readiness breakdown" in text
        or "placement readiness breakdown" in text
    ):
        return {
            "operation": "readiness",
            "confidence": 0.99,
        }

    # -----------------------------------------------
    # At-risk students
    # -----------------------------------------------
    if (
        "at risk" in text
        or "at-risk" in text
        or "at risk students" in text
        or "weak students" in text
    ):
        return {
            "operation": "at_risk",
            "confidence": 0.96,
        }

    metrics = _detect_local_metrics(text)

    # -----------------------------------------------
    # Skill lookup / count
    # -----------------------------------------------
    skill = _detect_local_skill(text)
    if skill is not None and any(
        phrase in text
        for phrase in (
            "how many",
            "who know",
            "who knows",
            "students know",
            "students who know",
            "students with",
            "students have",
            "skill",
            "skills",
        )
    ):
        return {
            "operation": "skill",
            "skill": skill,
            "confidence": 0.97,
        }

    # -----------------------------------------------
    # Conditions such as:
    # CGPA above 8 and attendance above 75
    # -----------------------------------------------
    conditions = []
    for metric in metrics:
        condition = _extract_local_condition(text, metric)
        if condition:
            conditions.append(condition)

    if conditions:
        select_metrics = metrics.copy()
        if not select_metrics:
            select_metrics = [conditions[0]["metric"]]

        wants_count = any(
            phrase in text
            for phrase in (
                "how many",
                "number of students",
                "count of students",
                "student count",
            )
        )

        return {
            "operation": "query",
            "filters": conditions,
            "select_metrics": select_metrics,
            "aggregation": "count" if wants_count else "rows",
            "order_by": conditions[0]["metric"],
            "order_direction": "desc",
            "limit": 1 if wants_count else 100,
            "confidence": min(0.98, 0.90 + 0.02 * len(conditions)),
        }

    # -----------------------------------------------
    # Aggregations
    # -----------------------------------------------
    aggregation = None
    if any(
        word in text
        for word in (
            "average",
            "avg",
            "mean",
        )
    ):
        aggregation = "avg"
    elif any(
        word in text
        for word in (
            "how many students",
            "number of students",
            "count of students",
            "student count",
            "how many",
        )
    ) and metrics:
        aggregation = "count"

    if aggregation and metrics:
        return {
            "operation": "query",
            "filters": [],
            "select_metrics": [metrics[0]],
            "aggregation": aggregation,
            "order_by": metrics[0],
            "order_direction": "desc",
            "limit": 1,
            "confidence": 0.95,
        }

    # -----------------------------------------------
    # Ranking queries
    # -----------------------------------------------
    limit = _extract_local_limit(text)
    ranking_requested = any(
        phrase in text
        for phrase in (
            "top",
            "bottom",
            "highest",
            "lowest",
            "best",
            "worst",
        )
    )

    if ranking_requested and metrics:
        order_metric, direction = _extract_local_order(
            text,
            metrics,
        )

        if any(
            phrase in text
            for phrase in (
                "highest",
                "best",
            )
        ):
            direction = "desc"
            if limit is None:
                limit = 1

        if any(
            phrase in text
            for phrase in (
                "lowest",
                "worst",
            )
        ):
            direction = "asc"
            if limit is None:
                limit = 1

        return {
            "operation": "query",
            "filters": [],
            "select_metrics": [order_metric],
            "aggregation": "rows",
            "order_by": order_metric,
            "order_direction": direction,
            "limit": limit or 5,
            "confidence": 0.96,
        }

    # -----------------------------------------------
    # Comparison / list queries
    # -----------------------------------------------
    if metrics and any(
        phrase in text
        for phrase in (
            "compare",
            "comparison",
            "show",
            "display",
            "list",
            "give me",
            "what are",
            "show me",
        )
    ):
        return {
            "operation": "query",
            "filters": [],
            "select_metrics": metrics,
            "aggregation": "rows",
            "order_by": "student_id",
            "order_direction": "asc",
            "limit": 100,
            "confidence": 0.92,
        }

    # -----------------------------------------------
    # Single metric bare queries, e.g. "average attendance"
    # already handled above; here we accept direct list requests.
    # -----------------------------------------------
    if metrics:
        return {
            "operation": "query",
            "filters": [],
            "select_metrics": metrics,
            "aggregation": "rows",
            "order_by": "student_id",
            "order_direction": "asc",
            "limit": 100,
            "confidence": 0.88,
        }

    return None


def execute_local_data_query(parsed):
    """Execute a locally understood data request with zero Gemini calls."""
    operation = parsed.get("operation")

    if operation == "readiness":
        result = tool_get_readiness_distribution()
        return build_data_response(
            "get_readiness_distribution",
            {},
            result,
        )

    if operation == "at_risk":
        result = tool_get_at_risk_students()
        return build_data_response(
            "get_at_risk_students",
            {},
            result,
        )

    if operation == "skill":
        result = tool_get_students_by_skill(
            parsed["skill"]
        )
        return build_data_response(
            "get_students_by_skill",
            {"skill": parsed["skill"]},
            result,
        )

    if operation == "query":
        result = tool_query_student_data(
            filters=parsed.get("filters", []),
            select_metrics=parsed.get("select_metrics", []),
            aggregation=parsed.get("aggregation", "rows"),
            order_by=parsed.get("order_by", "student_id"),
            order_direction=parsed.get("order_direction", "asc"),
            limit=parsed.get("limit", 20),
        )
        return build_generic_data_response(result)

    return None

# ==================================================
# GENERAL AI ENDPOINT
# ==================================================

@app.post("/ai-chat")
def ai_chat(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """Unified Placement AI Advisor for general and personalized guidance."""

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    try:

        # One assistant handles both generic AI advice and personalized
        # placement/career guidance. For students we attach a compact copy
        # of verified profile data, so the model can personalize the answer
        # without introducing a second advisor agent or Gemini call.
        prompt = message

        if user.get("role") == "student" and user.get("student_id"):
            try:
                profile = get_student(int(user["student_id"]))
                context = {
                    "student_name": profile.get("student_name", ""),
                    "objective": profile.get("objective", ""),
                    "skills": profile.get("skills", []),
                    "soft_skills": profile.get("soft_skills", []),
                    "projects": [
                        {
                            "title": x.get("title", ""),
                            "description": x.get("description", ""),
                            "technologies": x.get("technologies", ""),
                        }
                        for x in (profile.get("projects") or [])[:10]
                    ],
                    "internships": [
                        {
                            "company_name": x.get("company_name", ""),
                            "role": x.get("role", ""),
                            "description": x.get("description", ""),
                            "technologies": x.get("technologies", ""),
                        }
                        for x in (profile.get("internships") or [])[:10]
                    ],
                    "certifications": [
                        {"name": x.get("name", ""), "issuer": x.get("issuer", "")}
                        for x in (profile.get("certifications") or [])[:10]
                    ],
                    "achievements": [
                        {"title": x.get("title", ""), "description": x.get("description", "")}
                        for x in (profile.get("achievements") or [])[:10]
                    ],
                }

                prompt = (
                    "You are the Placement AI Advisor for this student. "
                    "Use the verified profile below for career, placement, skill, "
                    "project, internship, certification, achievement, interview, "
                    "and role-specific learning questions. Never invent student facts. "
                    "For general questions, answer normally.\n\n"
                    "VERIFIED STUDENT PROFILE:\n"
                    + json.dumps(context, ensure_ascii=False)
                    + "\n\nSTUDENT QUESTION:\n"
                    + message
                )
            except Exception as context_error:
                print("[Placement AI Advisor] Profile context unavailable:", context_error)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt)
                ],
            )
        ]

        response = generate_gemini_response(
            assistant_type="ai",
            contents=contents,
            config=build_ai_config(),
        )

        return normalize_ai_response(
            parse_final_json(response.text)
        )

    except Exception as error:

        print(
            "[AI CHAT ERROR]",
            repr(error)
        )

        error_text = str(error).lower()

        if (
            "429" in error_text
            or "resource_exhausted" in error_text
            or "quota" in error_text
        ):

            raise HTTPException(
                status_code=429,
                detail=(
                    "AI assistant quota is exhausted "
                    "for all configured Gemini projects."
                ),
            )

        raise HTTPException(
            status_code=500,
            detail=f"AI assistant error: {error}",
        )


# ==================================================
# NATURAL-LANGUAGE DATA ENDPOINT
# ==================================================

@app.post("/data-chat", dependencies=[Depends(require_faculty)])
def data_chat(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """
    Local-NLU-first data assistant.

    Common analytics requests are handled with zero Gemini calls.
    Gemini is used only when the local parser cannot confidently
    understand the request.
    """

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # --------------------------------------------------
    # 1. LOCAL NLU FIRST
    # --------------------------------------------------
    try:
        parsed = parse_local_nlu(message)
    except Exception as local_error:
        print(
            "[Data NLU ERROR]",
            repr(local_error),
        )
        parsed = None

    if parsed is not None:
        confidence = float(
            parsed.get("confidence", 0.0)
        )

        print(
            f"[Data NLU] confidence={confidence:.2f} "
            f"query={message!r}"
        )

        if confidence >= 0.85:
            try:
                local_response = execute_local_data_query(
                    parsed
                )

                if local_response is not None:
                    print(
                        "[Data Assistant] Local NLU handled query."
                    )
                    return local_response

            except Exception as local_db_error:
                print(
                    "[Data NLU DB ERROR]",
                    repr(local_db_error),
                )
                # Fall through to Gemini only if the local path
                # could not safely execute.

    # --------------------------------------------------
    # 2. GEMINI FALLBACK
    # --------------------------------------------------
    try:
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=message)
                ],
            )
        ]

        response = generate_gemini_response(
            assistant_type="data",
            contents=contents,
            config=build_data_config(),
        )

        function_calls = response.function_calls or []

        if not function_calls:
            return {
                "type": "data",
                "answer": (
                    "I could not map that request to the "
                    "student database. Try asking about "
                    "CGPA, attendance, skills, projects, "
                    "internships or placement readiness."
                ),
                "chart_type": "none",
                "title": "Student Data Assistant",
                "data": [],
            }

        # Execute only the first selected tool.
        function_call = function_calls[0]
        tool_name = function_call.name
        tool_arguments = dict(
            function_call.args or {}
        )

        tool_result = execute_tool(
            tool_name,
            tool_arguments,
        )

        if tool_name == "query_student_data":
            return build_generic_data_response(tool_result)

        return build_data_response(
            tool_name,
            tool_arguments,
            tool_result,
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            "[DATA CHAT ERROR]",
            repr(error),
        )

        error_text = str(error).lower()

        if (
            "429" in error_text
            or "resource_exhausted" in error_text
            or "quota" in error_text
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Data assistant quota is currently unavailable "
                    "for the configured Gemini fallback."
                ),
            )

        raise HTTPException(
            status_code=500,
            detail=f"Data assistant error: {error}",
        )


# ==================================================
# BACKWARD-COMPATIBLE /chat
# ==================================================

@app.post("/chat", dependencies=[Depends(require_faculty)])
def chat_with_ai(request: ChatRequest):
    """
    Existing frontend compatibility.

    The current React application uses /chat, so it is kept as
    an alias for the database assistant. New dual UI code should
    call /ai-chat for the general AI popup and /data-chat for
    database analytics.
    """

    return data_chat(request)




# ==================================================
# END
# ==================================================

# Start backend with:
# python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
# ==================================================
# LIVE DATASET COMPATIBILITY LAYER
# ==================================================

LIVE_DATASET_CANDIDATES = [
    Path(os.getenv("LIVE_DATASET_PATH")) if os.getenv("LIVE_DATASET_PATH") else None,
    Path(__file__).resolve().parent / "AIDS live Dataset.xlsx",
    Path(__file__).resolve().parent / "AI&DS DATA SET (2)-Current_2027.csv",
    Path(__file__).resolve().parent.parent / "AIDS live Dataset.xlsx",
    Path(r"C:\Users\USER\Desktop\Final yr project\AIDS live Dataset.xlsx"),
]


def _find_live_dataset_path():
    for path in LIVE_DATASET_CANDIDATES:
        if path and path.exists():
            return path
    return None


def _db_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _db_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_live_float(value):
    try:
        return 0.0 if pd.isna(value) else float(value)
    except Exception:
        return 0.0


def _safe_live_int(value):
    try:
        return 0 if pd.isna(value) else int(float(value))
    except Exception:
        return 0


def _ensure_live_student_columns(connection, cursor):
    specs = {
        "roll_number":"TEXT","gender":"TEXT","year":"TEXT","branch":"TEXT",
        "tenth_percentage":"DOUBLE PRECISION","twelfth_percentage":"DOUBLE PRECISION",
        "sem_1_percentage":"DOUBLE PRECISION","sem_1_sgpa":"DOUBLE PRECISION",
        "sem_2_percentage":"DOUBLE PRECISION","sem_2_sgpa":"DOUBLE PRECISION",
        "sem_3_percentage":"DOUBLE PRECISION","sem_3_sgpa":"DOUBLE PRECISION",
        "sem_4_percentage":"DOUBLE PRECISION","sem_4_sgpa":"DOUBLE PRECISION",
        "sem_5_percentage":"DOUBLE PRECISION","sem_5_sgpa":"DOUBLE PRECISION",
        "sem_6_percentage":"DOUBLE PRECISION","sem_6_sgpa":"DOUBLE PRECISION",
        "sem_7_percentage":"DOUBLE PRECISION","sem_7_sgpa":"DOUBLE PRECISION",
        "sem_8_percentage":"DOUBLE PRECISION","sem_8_sgpa":"DOUBLE PRECISION",
        "amcat_logical_score":"DOUBLE PRECISION","amcat_programming_score":"DOUBLE PRECISION",
        "amcat_writex_score":"DOUBLE PRECISION","amcat_quantitative_score":"DOUBLE PRECISION",
        "amcat_english_score":"DOUBLE PRECISION","amcat_data_structures_score":"DOUBLE PRECISION",
        "amcat_python_score":"DOUBLE PRECISION","amcat_cpp_score":"DOUBLE PRECISION",
        "amcat_core_java_score":"DOUBLE PRECISION","amcat_automata_score":"DOUBLE PRECISION",
        "amcat_automata_fix_score":"DOUBLE PRECISION","amcat_svar_spoken_english_score":"DOUBLE PRECISION",
        "amcat_score":"DOUBLE PRECISION","project_description":"TEXT","internship_done":"TEXT",
        "internship_type":"TEXT","internship_description":"TEXT","achievements":"TEXT",
        "skills":"TEXT","area_of_interest":"TEXT","programming_language_interest_1":"TEXT",
        "programming_language_interest_2":"TEXT",
    }
    for name, dtype in specs.items():
        cursor.execute(f'ALTER TABLE students ADD COLUMN IF NOT EXISTS "{name}" {dtype}')


def _safe_live_text(value):
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip() if value is not None else ""


def _sync_live_dataset_to_db(source_path=None, overwrite_existing=False):
    """Load live Excel records without overwriting user-edited database fields unless requested."""
    path = Path(source_path) if source_path else _find_live_dataset_path()
    if not path or not path.exists():
        return 0
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    if not {"Student_Name", "Roll_Number"}.issubset(df.columns):
        return 0
    connection = cursor = None
    count = 0
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        _ensure_live_student_columns(connection, cursor)
        for _, r in df.iterrows():
            name = str(r.get("Student_Name", "") or "").strip()
            roll = str(r.get("Roll_Number", "") or "").strip()
            if not name or not roll or roll.lower() == "nan":
                continue
            vals = {
                "gender": _safe_live_text(r.get("Gender", "")), "year": _safe_live_text(r.get("Year", "")), "branch": _safe_live_text(r.get("Branch", "")),
                "tenth_percentage": _safe_live_float(r.get("10th_Percentage")), "twelfth_percentage": _safe_live_float(r.get("12th_Percentage")),
                "sem_1_percentage": _safe_live_float(r.get("Sem_1_Percentage")), "sem_1_sgpa": _safe_live_float(r.get("Sem_1_SGPA")),
                "sem_2_percentage": _safe_live_float(r.get("Sem_2_Percentage")), "sem_2_sgpa": _safe_live_float(r.get("Sem_2_SGPA")),
                "sem_3_percentage": _safe_live_float(r.get("Sem_3_Percentage")), "sem_3_sgpa": _safe_live_float(r.get("Sem_3_SGPA")),
                "sem_4_percentage": _safe_live_float(r.get("Sem_4_Percentage")), "sem_4_sgpa": _safe_live_float(r.get("Sem_4_SGPA")),
                "sem_5_percentage": _safe_live_float(r.get("Sem_5_Percentage")), "sem_5_sgpa": _safe_live_float(r.get("Sem_5_SGPA")),
                "sem_6_percentage": _safe_live_float(r.get("Sem_6_Percentage")), "sem_6_sgpa": _safe_live_float(r.get("Sem_6_SGPA")),
                "sem_7_percentage": _safe_live_float(r.get("Sem_7_Percentage")), "sem_7_sgpa": _safe_live_float(r.get("Sem_7_SGPA")),
                "sem_8_percentage": _safe_live_float(r.get("Sem_8_Percentage")), "sem_8_sgpa": _safe_live_float(r.get("Sem_8_SGPA")),
                "amcat_logical_score": _safe_live_float(r.get("AMCAT_Logical_Score")), "amcat_programming_score": _safe_live_float(r.get("AMCAT_Programming_Score")),
                "amcat_writex_score": _safe_live_float(r.get("AMCAT_WriteX_Score")), "amcat_quantitative_score": _safe_live_float(r.get("AMCAT_Quantitative_Score")),
                "amcat_english_score": _safe_live_float(r.get("AMCAT_English_Score")), "amcat_data_structures_score": _safe_live_float(r.get("AMCAT_Data_Structures_Score")),
                "amcat_python_score": _safe_live_float(r.get("AMCAT_Python_Score")), "amcat_cpp_score": _safe_live_float(r.get("AMCAT_Cpp_Score")),
                "amcat_core_java_score": _safe_live_float(r.get("AMCAT_Core_Java_Score")), "amcat_automata_score": _safe_live_float(r.get("AMCAT_Automata_Score")),
                "amcat_automata_fix_score": _safe_live_float(r.get("AMCAT_Automata_Fix_Score")), "amcat_svar_spoken_english_score": _safe_live_float(r.get("AMCAT_SVAR_Spoken_English_Score")),
                "amcat_score": _safe_live_float(r.get("AMCAT_Score")), "project_description": _safe_live_text(r.get("Project_Description", "")),
                "internship_done": _safe_live_text(r.get("Internship_Done", "")), "internship_type": _safe_live_text(r.get("Internship_Type", "")),
                "internship_description": _safe_live_text(r.get("Internship_Description", "")), "achievements": _safe_live_text(r.get("Achievements", "")),
                "skills": _safe_live_text(r.get("Skills", "")), "area_of_interest": _safe_live_text(r.get("Area_of_Interest", "")),
                "programming_language_interest_1": _safe_live_text(r.get("Programming_Language_Interest_1", "")), "programming_language_interest_2": _safe_live_text(r.get("Programming_Language_Interest_2", "")),
            }
            base = (_safe_live_float(r.get("CGPA")), _safe_live_float(r.get("Technical_Skill_Score")), _safe_live_int(r.get("Projects_Count")), _safe_live_int(r.get("Internship_Count")), _safe_live_int(r.get("Certifications_Count")), _safe_live_int(r.get("Backlogs")))
            cursor.execute("SELECT student_id FROM students WHERE roll_number = %s LIMIT 1", (roll,))
            existing = cursor.fetchone()
            if existing and not overwrite_existing:
                continue
            if existing:
                sid = int(existing[0])
                cursor.execute("""UPDATE students SET student_name=%s,cgpa=%s,technical_skill_score=%s,projects_count=%s,internships_count=%s,certifications_count=%s,backlogs=%s,roll_number=%s,gender=%s,year=%s,branch=%s,tenth_percentage=%s,twelfth_percentage=%s,sem_1_percentage=%s,sem_1_sgpa=%s,sem_2_percentage=%s,sem_2_sgpa=%s,sem_3_percentage=%s,sem_3_sgpa=%s,sem_4_percentage=%s,sem_4_sgpa=%s,sem_5_percentage=%s,sem_5_sgpa=%s,sem_6_percentage=%s,sem_6_sgpa=%s,sem_7_percentage=%s,sem_7_sgpa=%s,sem_8_percentage=%s,sem_8_sgpa=%s,amcat_logical_score=%s,amcat_programming_score=%s,amcat_writex_score=%s,amcat_quantitative_score=%s,amcat_english_score=%s,amcat_data_structures_score=%s,amcat_python_score=%s,amcat_cpp_score=%s,amcat_core_java_score=%s,amcat_automata_score=%s,amcat_automata_fix_score=%s,amcat_svar_spoken_english_score=%s,amcat_score=%s,project_description=%s,internship_done=%s,internship_type=%s,internship_description=%s,achievements=%s,skills=%s,area_of_interest=%s,programming_language_interest_1=%s,programming_language_interest_2=%s WHERE student_id=%s""", (name,*base,roll,vals["gender"],vals["year"],vals["branch"],vals["tenth_percentage"],vals["twelfth_percentage"],vals["sem_1_percentage"],vals["sem_1_sgpa"],vals["sem_2_percentage"],vals["sem_2_sgpa"],vals["sem_3_percentage"],vals["sem_3_sgpa"],vals["sem_4_percentage"],vals["sem_4_sgpa"],vals["sem_5_percentage"],vals["sem_5_sgpa"],vals["sem_6_percentage"],vals["sem_6_sgpa"],vals["sem_7_percentage"],vals["sem_7_sgpa"],vals["sem_8_percentage"],vals["sem_8_sgpa"],vals["amcat_logical_score"],vals["amcat_programming_score"],vals["amcat_writex_score"],vals["amcat_quantitative_score"],vals["amcat_english_score"],vals["amcat_data_structures_score"],vals["amcat_python_score"],vals["amcat_cpp_score"],vals["amcat_core_java_score"],vals["amcat_automata_score"],vals["amcat_automata_fix_score"],vals["amcat_svar_spoken_english_score"],vals["amcat_score"],vals["project_description"],vals["internship_done"],vals["internship_type"],vals["internship_description"],vals["achievements"],vals["skills"],vals["area_of_interest"],vals["programming_language_interest_1"],vals["programming_language_interest_2"],sid))
            else:
                cursor.execute("""INSERT INTO students (student_name,cgpa,attendance_percent,technical_skill_score,aptitude_score,communication_score,projects_count,internships_count,certifications_count,coding_assessment_score,backlogs,roll_number,gender,year,branch,tenth_percentage,twelfth_percentage,sem_1_percentage,sem_1_sgpa,sem_2_percentage,sem_2_sgpa,sem_3_percentage,sem_3_sgpa,sem_4_percentage,sem_4_sgpa,sem_5_percentage,sem_5_sgpa,sem_6_percentage,sem_6_sgpa,sem_7_percentage,sem_7_sgpa,sem_8_percentage,sem_8_sgpa,amcat_logical_score,amcat_programming_score,amcat_writex_score,amcat_quantitative_score,amcat_english_score,amcat_data_structures_score,amcat_python_score,amcat_cpp_score,amcat_core_java_score,amcat_automata_score,amcat_automata_fix_score,amcat_svar_spoken_english_score,amcat_score,project_description,internship_done,internship_type,internship_description,achievements,skills,area_of_interest,programming_language_interest_1,programming_language_interest_2) VALUES (%s,%s,0,%s,0,0,%s,%s,%s,0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING student_id""", (name,*base,roll,vals["gender"],vals["year"],vals["branch"],vals["tenth_percentage"],vals["twelfth_percentage"],vals["sem_1_percentage"],vals["sem_1_sgpa"],vals["sem_2_percentage"],vals["sem_2_sgpa"],vals["sem_3_percentage"],vals["sem_3_sgpa"],vals["sem_4_percentage"],vals["sem_4_sgpa"],vals["sem_5_percentage"],vals["sem_5_sgpa"],vals["sem_6_percentage"],vals["sem_6_sgpa"],vals["sem_7_percentage"],vals["sem_7_sgpa"],vals["sem_8_percentage"],vals["sem_8_sgpa"],vals["amcat_logical_score"],vals["amcat_programming_score"],vals["amcat_writex_score"],vals["amcat_quantitative_score"],vals["amcat_english_score"],vals["amcat_data_structures_score"],vals["amcat_python_score"],vals["amcat_cpp_score"],vals["amcat_core_java_score"],vals["amcat_automata_score"],vals["amcat_automata_fix_score"],vals["amcat_svar_spoken_english_score"],vals["amcat_score"],vals["project_description"],vals["internship_done"],vals["internship_type"],vals["internship_description"],vals["achievements"],vals["skills"],vals["area_of_interest"],vals["programming_language_interest_1"],vals["programming_language_interest_2"]))
            count += 1
        connection.commit()
        return count
    except Exception as error:
        if connection: connection.rollback()
        print("[LIVE DATA SYNC ERROR]", repr(error))
        return 0
    finally:
        close_db(cursor, connection)


def _find_live_student(student_name, roll_number):
    _sync_live_dataset_to_db()
    connection = None
    cursor = None
    try:
        connection = get_db_connection(); cursor = connection.cursor()
        cursor.execute("SELECT student_id,student_name,roll_number FROM students WHERE LOWER(TRIM(student_name))=LOWER(TRIM(%s)) AND LOWER(TRIM(roll_number))=LOWER(TRIM(%s)) LIMIT 1", (student_name,roll_number))
        return cursor.fetchone()
    finally:
        close_db(cursor, connection)



# ==================================================
# LIVE FACULTY RISK/CLUSTER HELPERS
# ==================================================
def _live_at_risk_details():
    _sync_live_dataset_to_db()
    conn = cur = None
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT student_id,student_name,roll_number,cgpa,technical_skill_score,amcat_score,projects_count,internships_count,certifications_count,backlogs FROM students ORDER BY student_id")
        students=[]
        for row in cur.fetchall():
            cgpa=_db_float(row[3]); tech=_db_float(row[4]); backlogs=_db_int(row[9])
            reasons=[]
            if cgpa is not None and cgpa < 6.5: reasons.append("CGPA < 6.5")
            if tech is not None and tech < 50: reasons.append("Technical Skill Score < 50")
            if backlogs > 1: reasons.append("Backlogs > 1")
            if not reasons: continue
            students.append({"student_id":int(row[0]),"student_name":row[1],"roll_number":row[2],"cgpa":cgpa,"technical_skills":tech,"amcat_score":_db_float(row[5]),"projects":_db_int(row[6]),"internships":_db_int(row[7]),"certifications":_db_int(row[8]),"backlogs":backlogs,"attendance":None,"aptitude":None,"communication":None,"coding_assessment":None,"risk_rule":" OR ".join(reasons)})
        students.sort(key=lambda x: (x["cgpa"] if x["cgpa"] is not None else 999, -x["backlogs"]))
        return {"count":len(students),"students":students,"availability_note":"Risk uses available live fields only: CGPA, Technical Skill Score and Backlogs. Attendance is not present in the live dataset."}
    finally:
        close_db(cur,conn)

def _live_cluster_details(cluster_name):
    requested=str(cluster_name).strip().lower()
    if not requested: raise HTTPException(status_code=400,detail="Cluster name cannot be empty.")
    _sync_live_dataset_to_db(); conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); cur.execute("SELECT student_id FROM students ORDER BY student_id")
        students=[]
        for r in cur.fetchall():
            sid=int(r[0])
            try:
                c=_live_cluster_student(sid)
                if str(c.get("cluster_name","")).strip().lower()!=requested: continue
                p=_live_get_student_feature_dict(sid)
                students.append({"student_id":sid,"student_name":p.get("Student_Name"),"roll_number":p.get("Roll_Number"),"cgpa":_safe_live_float(p.get("CGPA")),"technical_skills":_safe_live_float(p.get("Technical_Skill_Score")),"amcat_score":_safe_live_float(p.get("AMCAT_Score")),"projects":_safe_live_int(p.get("Projects_Count")),"internships":_safe_live_int(p.get("Internship_Count")),"certifications":_safe_live_int(p.get("Certifications_Count")),"backlogs":_safe_live_int(p.get("Backlogs"))})
            except Exception:
                continue
        return {"cluster":cluster_name,"count":len(students),"students":students}
    finally:
        close_db(cur,conn)

def _live_explain_student(student_id):
    result=_live_shap_student(student_id)
    features=result.get("features",[])
    positive=[x for x in features if float(x.get("impact",0) or 0)>=0]
    negative=[x for x in features if float(x.get("impact",0) or 0)<0]
    parts=[]
    if positive: parts.append("Positive factors: "+", ".join(str(x.get("feature","")).replace("_"," ") for x in positive[:3]))
    if negative: parts.append("Factors reducing the model output: "+", ".join(str(x.get("feature","")).replace("_"," ") for x in negative[:3]))
    result["explanation"]=". ".join(parts) if parts else "SHAP factors are unavailable for this student."
    result["all_feature_explanations"]=features; result["top_influencing_factors"]=features[:5]
    result["positive_factors"]=[str(x.get("feature")) for x in positive]; result["negative_factors"]=[str(x.get("feature")) for x in negative]
    result["predicted_readiness"]=result.get("predicted_readiness","Unknown")
    return result

# ==================================================
# LIVE DATASET / SEMESTER-AWARE COMPATIBILITY LAYER
# ==================================================

LIVE_MODEL_DIR = Path(__file__).resolve().parent / "trained_models"
if not LIVE_MODEL_DIR.exists():
    LIVE_MODEL_DIR = Path(r"C:\Users\USER\Desktop\Final yr project\Backend\trained_models")

LIVE_SEMESTER_FEATURES = {
    4: ["10th_Percentage","12th_Percentage","Backlogs","Sem_1_Percentage","Sem_1_SGPA","Sem_2_Percentage","Sem_2_SGPA","Sem_3_Percentage","Sem_3_SGPA","Sem_4_Percentage","Sem_4_SGPA","CGPA","Semester_Average","Recent_Semester_Average","Semester_Trend","Technical_Skill_Score","AMCAT_Score"],
    5: ["10th_Percentage","12th_Percentage","Backlogs","Sem_1_Percentage","Sem_1_SGPA","Sem_2_Percentage","Sem_2_SGPA","Sem_3_Percentage","Sem_3_SGPA","Sem_4_Percentage","Sem_4_SGPA","Sem_5_Percentage","Sem_5_SGPA","CGPA","Semester_Average","Recent_Semester_Average","Semester_Trend","Technical_Skill_Score","AMCAT_Score"],
    6: ["10th_Percentage","12th_Percentage","Backlogs","Sem_1_Percentage","Sem_1_SGPA","Sem_2_Percentage","Sem_2_SGPA","Sem_3_Percentage","Sem_3_SGPA","Sem_4_Percentage","Sem_4_SGPA","Sem_5_Percentage","Sem_5_SGPA","Sem_6_Percentage","Sem_6_SGPA","CGPA","Semester_Average","Recent_Semester_Average","Semester_Trend","Technical_Skill_Score","AMCAT_Score"],
    7: ["10th_Percentage","12th_Percentage","Backlogs","Sem_1_Percentage","Sem_1_SGPA","Sem_2_Percentage","Sem_2_SGPA","Sem_3_Percentage","Sem_3_SGPA","Sem_4_Percentage","Sem_4_SGPA","Sem_5_Percentage","Sem_5_SGPA","Sem_6_Percentage","Sem_6_SGPA","Sem_7_Percentage","Sem_7_SGPA","CGPA","Semester_Average","Recent_Semester_Average","Semester_Trend","Technical_Skill_Score","AMCAT_Score"],
    8: ["10th_Percentage","12th_Percentage","Backlogs","Sem_1_Percentage","Sem_1_SGPA","Sem_2_Percentage","Sem_2_SGPA","Sem_3_Percentage","Sem_3_SGPA","Sem_4_Percentage","Sem_4_SGPA","Sem_5_Percentage","Sem_5_SGPA","Sem_6_Percentage","Sem_6_SGPA","Sem_7_Percentage","Sem_7_SGPA","Sem_8_Percentage","Sem_8_SGPA","CGPA","Semester_Average","Recent_Semester_Average","Semester_Trend","Technical_Skill_Score","AMCAT_Score"],
}

LIVE_DB_MAP = {
    "Student_Name":"student_name", "Roll_Number":"roll_number", "Gender":"gender", "Year":"year", "Branch":"branch",
    "10th_Percentage":"tenth_percentage", "12th_Percentage":"twelfth_percentage",
    "Sem_1_Percentage":"sem_1_percentage", "Sem_1_SGPA":"sem_1_sgpa", "Sem_2_Percentage":"sem_2_percentage", "Sem_2_SGPA":"sem_2_sgpa",
    "Sem_3_Percentage":"sem_3_percentage", "Sem_3_SGPA":"sem_3_sgpa", "Sem_4_Percentage":"sem_4_percentage", "Sem_4_SGPA":"sem_4_sgpa",
    "Sem_5_Percentage":"sem_5_percentage", "Sem_5_SGPA":"sem_5_sgpa", "Sem_6_Percentage":"sem_6_percentage", "Sem_6_SGPA":"sem_6_sgpa",
    "Sem_7_Percentage":"sem_7_percentage", "Sem_7_SGPA":"sem_7_sgpa", "Sem_8_Percentage":"sem_8_percentage", "Sem_8_SGPA":"sem_8_sgpa",
    "AMCAT_Logical_Score":"amcat_logical_score", "AMCAT_Programming_Score":"amcat_programming_score", "AMCAT_WriteX_Score":"amcat_writex_score",
    "AMCAT_Quantitative_Score":"amcat_quantitative_score", "AMCAT_English_Score":"amcat_english_score", "AMCAT_Data_Structures_Score":"amcat_data_structures_score",
    "AMCAT_Python_Score":"amcat_python_score", "AMCAT_Cpp_Score":"amcat_cpp_score", "AMCAT_Core_Java_Score":"amcat_core_java_score",
    "AMCAT_Automata_Score":"amcat_automata_score", "AMCAT_Automata_Fix_Score":"amcat_automata_fix_score", "AMCAT_SVAR_Spoken_English_Score":"amcat_svar_spoken_english_score",
    "Technical_Skill_Score":"technical_skill_score", "AMCAT_Score":"amcat_score", "Projects_Count":"projects_count", "Project_Description":"project_description",
    "Internship_Done":"internship_done", "Internship_Type":"internship_type", "Internship_Count":"internships_count", "Internship_Description":"internship_description",
    "Certifications_Count":"certifications_count", "Achievements":"achievements", "Skills":"skills", "Area_of_Interest":"area_of_interest",
    "Programming_Language_Interest_1":"programming_language_interest_1", "Programming_Language_Interest_2":"programming_language_interest_2", "Backlogs":"backlogs", "CGPA":"cgpa",
}
LIVE_TEXT_FIELDS={"Student_Name","Roll_Number","Gender","Year","Branch","Project_Description","Internship_Done","Internship_Type","Internship_Description","Achievements","Skills","Area_of_Interest","Programming_Language_Interest_1","Programming_Language_Interest_2"}

def _live_current_semester(profile):
    for sem in range(8,3,-1):
        if profile.get(f"Sem_{sem}_SGPA") is not None or profile.get(f"Sem_{sem}_Percentage") is not None:
            return sem
    return 3


def _live_row_to_profile(columns,row):
    p={"student_id":int(row[0])}
    for i,col in enumerate(columns,1):
        value=row[i]
        if col not in LIVE_TEXT_FIELDS and value is not None:
            try: value=float(value)
            except (TypeError,ValueError): value=None
        p[col]=value
    # Derived values use only semesters actually present in the live workbook.
    sem_values=[p.get(f"Sem_{i}_SGPA") for i in range(1,7)]
    sem_values=[float(v) for v in sem_values if v is not None]
    p["Semester_Average"]=float(np.mean(sem_values)) if sem_values else None
    p["Recent_Semester_Average"]=float(np.mean(sem_values[-2:])) if sem_values else None
    early=sem_values[:3]; late=sem_values[3:]
    p["Semester_Trend"]=float(np.mean(late)-np.mean(early)) if early and late else None
    tech=[p.get(x) for x in ["AMCAT_Programming_Score","AMCAT_Data_Structures_Score","AMCAT_Python_Score","AMCAT_Cpp_Score","AMCAT_Core_Java_Score","AMCAT_Automata_Score","AMCAT_Automata_Fix_Score"]]
    tech=[float(x) for x in tech if x is not None]
    amcat=[p.get(x) for x in ["AMCAT_Logical_Score","AMCAT_Programming_Score","AMCAT_WriteX_Score","AMCAT_Quantitative_Score","AMCAT_English_Score","AMCAT_Data_Structures_Score","AMCAT_Python_Score","AMCAT_Cpp_Score","AMCAT_Core_Java_Score","AMCAT_Automata_Score","AMCAT_Automata_Fix_Score","AMCAT_SVAR_Spoken_English_Score"]]
    amcat=[float(x) for x in amcat if x is not None]
    p["Technical_Skill_Score"]=float(np.mean(tech)) if tech else p.get("Technical_Skill_Score")
    p["AMCAT_Score"]=float(np.mean(amcat)) if amcat else p.get("AMCAT_Score")
    return p


def _live_get_student_feature_dict(student_id:int):
    conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn, cur); conn.commit(); cols=list(LIVE_DB_MAP.keys()); quoted=",".join(f'"{LIVE_DB_MAP[c]}"' for c in cols)
        cur.execute(f'SELECT student_id,{quoted} FROM students WHERE student_id=%s',(student_id,)); row=cur.fetchone()
        if row is None: raise HTTPException(status_code=404,detail="Student not found")
        return _live_row_to_profile(cols,row)
    finally: close_db(cur,conn)


def _live_list_students():
    conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); cols=list(LIVE_DB_MAP.keys()); quoted=",".join(f'"{LIVE_DB_MAP[c]}"' for c in cols)
        cur.execute(f'SELECT student_id,{quoted} FROM students WHERE "{LIVE_DB_MAP["Roll_Number"]}" IS NOT NULL ORDER BY student_id'); rows=cur.fetchall(); out=[]
        for row in rows:
            p=_live_row_to_profile(cols,row); p.update({"student_name":p.get("Student_Name"),"roll_number":p.get("Roll_Number"),"cgpa":p.get("CGPA"),"technical_skill_score":p.get("Technical_Skill_Score"),"amcat_score":p.get("AMCAT_Score"),"projects_count":p.get("Projects_Count") or 0,"internships_count":p.get("Internship_Count") or 0,"certifications_count":p.get("Certifications_Count") or 0,"backlogs":p.get("Backlogs") or 0}); out.append(p)
        return {"students":out,"count":len(out)}
    finally: close_db(cur,conn)


def _live_get_student_profile(student_id:int):
    p=_live_get_student_feature_dict(student_id); conn=cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); _ensure_student_detail_tables(conn,cur); conn.commit(); cur.execute('SELECT contact_email,phone,objective,soft_skills,profile_image_path FROM students WHERE student_id=%s',(student_id,)); meta=cur.fetchone() or ("","","","",None)
        p.update({"student_name":p.get("Student_Name"),"roll_number":p.get("Roll_Number"),"contact_email":meta[0] or "","phone":meta[1] or "","objective":meta[2] or "","soft_skills":[x.strip() for x in str(meta[3] or "").split(",") if x.strip()],"profile_image_path":meta[4],"cgpa":p.get("CGPA"),"technical_skill_score":p.get("Technical_Skill_Score"),"amcat_score":p.get("AMCAT_Score"),"semester_average":p.get("Semester_Average"),"semester_trend":p.get("Semester_Trend"),"projects_count":p.get("Projects_Count") or 0,"internships_count":p.get("Internship_Count") or 0,"certifications_count":p.get("Certifications_Count") or 0,"backlogs":p.get("Backlogs") or 0,"attendance_percent":None,"aptitude_score":None,"communication_score":None,"coding_assessment_score":None,"project_description":p.get("Project_Description") or "","internship_description":p.get("Internship_Description") or "","dataset_achievements":p.get("Achievements") or ""})
        cur.execute('SELECT project_id,title,description,technologies,start_date,end_date FROM student_projects WHERE student_id=%s ORDER BY created_at DESC,project_id DESC',(student_id,)); p["projects"]=[{"project_id":int(r[0]),"title":r[1],"description":r[2],"technologies":r[3],"start_date":r[4],"end_date":r[5]} for r in cur.fetchall()]
        cur.execute('SELECT internship_id,company_name,role,description,technologies,start_date,end_date FROM student_internships WHERE student_id=%s ORDER BY created_at DESC,internship_id DESC',(student_id,)); p["internships"]=[{"internship_id":int(r[0]),"company_name":r[1],"role":r[2],"description":r[3],"technologies":r[4],"start_date":r[5],"end_date":r[6]} for r in cur.fetchall()]
        cur.execute('SELECT skill_name FROM student_manual_skills WHERE student_id=%s ORDER BY skill_name',(student_id,)); p["manual_skills"]=[r[0] for r in cur.fetchall() if r[0]]
        base_skills=_split_technology_text(str(p.get("Skills") or ""))
        merged=[]; seen=set()
        for item in base_skills + p["manual_skills"]:
            key=str(item).strip().lower()
            if key and key not in seen:
                seen.add(key); merged.append(str(item).strip())
        p["skills"]=merged
        cur.execute('SELECT certification_id,name,issuer,issue_date,credential_url FROM student_certifications WHERE student_id=%s ORDER BY created_at DESC,certification_id DESC',(student_id,)); p["certifications"]=[{"certification_id":int(r[0]),"name":r[1],"issuer":r[2],"issue_date":r[3],"credential_url":r[4]} for r in cur.fetchall()]
        cur.execute('SELECT achievement_id,title,description,achievement_date FROM student_achievements WHERE student_id=%s ORDER BY created_at DESC,achievement_id DESC',(student_id,)); p["achievements"]=[{"achievement_id":int(r[0]),"title":r[1],"description":r[2],"achievement_date":r[3]} for r in cur.fetchall()]
        return p
    finally: close_db(cur,conn)


_LIVE_MODEL_CACHE = {}
_LIVE_KMEANS_CACHE = None
_FACULTY_OVERVIEW_CACHE = {"data": None, "created": 0.0, "version": 0}
_FACULTY_OVERVIEW_TTL = 30.0

def _invalidate_live_dashboard_cache():
    global _LIVE_KMEANS_CACHE
    _FACULTY_OVERVIEW_CACHE["data"] = None
    _FACULTY_OVERVIEW_CACHE["created"] = 0.0
    _FACULTY_OVERVIEW_CACHE["version"] += 1
    _LIVE_KMEANS_CACHE = None


def _live_load_model(semester):
    cached = _LIVE_MODEL_CACHE.get(semester)
    if cached is not None:
        return cached
    path=LIVE_MODEL_DIR/f"placement_model_sem{semester}.pkl"; cfg=LIVE_MODEL_DIR/f"feature_config_sem{semester}.json"
    if not (path.exists() and cfg.exists()):
        result=(None,{"semester":semester,"features":LIVE_SEMESTER_FEATURES.get(semester,[])})
    else:
        result=(joblib.load(path),json.loads(cfg.read_text(encoding="utf-8")))
    _LIVE_MODEL_CACHE[semester]=result
    return result


def _readiness_from_probability_live(probability):
    if probability is None: return "Unknown"
    if probability>=0.75: return "High"
    if probability>=0.50: return "Medium"
    return "Low"


def _live_predict_student(student_id:int):
    student=_live_get_student_feature_dict(student_id); sem=_live_current_semester(student); estimator,cfg=_live_load_model(sem)
    if estimator is None: raise HTTPException(status_code=503,detail=f"Semester {sem} placement model is unavailable.")
    features=cfg.get("features") or LIVE_SEMESTER_FEATURES[sem]; frame=pd.DataFrame([[student.get(f) for f in features]],columns=features).apply(pd.to_numeric,errors="coerce")
    prediction=estimator.predict(frame)[0]; probabilities=estimator.predict_proba(frame)[0] if hasattr(estimator,"predict_proba") else None; classes=list(getattr(estimator,"classes_",[])); placed=None
    if probabilities is not None:
        mapping={str(c).strip().lower():float(p) for c,p in zip(classes,probabilities)}; placed=mapping.get("placed")
        if placed is None: placed=mapping.get("1")
        if placed is None and len(probabilities)==2: placed=float(probabilities[-1])
    status="Placed" if str(prediction).strip().lower() in {"placed","1","true"} else "Not Placed"; confidence=float(max(probabilities))*100 if probabilities is not None else None
    return {"student_id":student_id,"student_name":student.get("Student_Name"),"prediction":int(prediction) if isinstance(prediction,(int,np.integer)) else prediction,"placement_status":status,"predicted_readiness":status,"placed_probability":round(placed*100,2) if placed is not None else None,"confidence":round(confidence,2) if confidence is not None else None,"technical_skill_score":student.get("Technical_Skill_Score"),"amcat_score":student.get("AMCAT_Score"),"semester_average":student.get("Semester_Average"),"semester_trend":student.get("Semester_Trend"),"current_semester":sem,"model":sem,"readiness_level":_readiness_from_probability_live(placed)}


def _live_cluster_student(student_id:int):
    student=_live_get_student_feature_dict(student_id); km_path=LIVE_MODEL_DIR/"kmeans_model.pkl"; sc_path=LIVE_MODEL_DIR/"kmeans_scaler.pkl"; labels_path=LIVE_MODEL_DIR/"cluster_labels.pkl"; cfg_path=LIVE_MODEL_DIR/"kmeans_feature_config.json"
    if not (km_path.exists() and sc_path.exists() and labels_path.exists() and cfg_path.exists()): raise HTTPException(status_code=503,detail="K-Means artifacts are unavailable.")
    km=joblib.load(km_path); scaler=joblib.load(sc_path); labels=joblib.load(labels_path); features=json.loads(cfg_path.read_text(encoding="utf-8")).get("features",LIVE_SEMESTER_FEATURES[6]); frame=pd.DataFrame([[student.get(f) for f in features]],columns=features).apply(pd.to_numeric,errors="coerce")
    frame=frame.fillna(pd.Series(getattr(scaler,"mean_",np.zeros(len(features))),index=features)); cluster=int(km.predict(scaler.transform(frame))[0]); label=str(labels.get(cluster,f"Cluster {cluster}"))
    return {"student_id":student_id,"cluster":cluster,"cluster_number":cluster,"cluster_name":label}


def _live_shap_student(student_id:int):
    student=_live_get_student_feature_dict(student_id)
    sem=_live_current_semester(student)
    estimator,cfg=_live_load_model(sem)
    if estimator is None:
        raise HTTPException(status_code=503,detail=f"Semester {sem} placement model is unavailable.")
    features=cfg.get("features") or LIVE_SEMESTER_FEATURES[sem]
    frame=pd.DataFrame([[student.get(f) for f in features]],columns=features).apply(pd.to_numeric,errors="coerce")
    try:
        model_for_shap=estimator.named_steps["model"] if hasattr(estimator,"named_steps") and "model" in estimator.named_steps else estimator
        transformed=estimator.named_steps["imputer"].transform(frame) if hasattr(estimator,"named_steps") and "imputer" in estimator.named_steps else frame
        explainer=shap.TreeExplainer(model_for_shap)
        raw=explainer.shap_values(transformed)
        arr=None
        values=np.asarray(raw)
        if isinstance(raw,list):
            if len(raw)>=2:
                arr=np.asarray(raw[1])[0]
            elif len(raw)==1:
                arr=np.asarray(raw[0])[0]
        elif values.ndim==3:
            arr=values[0,:,1] if values.shape[2]>1 else values[0,:,0]
        elif values.ndim==2:
            arr=values[0]
        elif values.ndim==1:
            arr=values
        if arr is None or np.asarray(arr).size != len(features):
            explanation=explainer(transformed)
            vals=np.asarray(explanation.values)
            if vals.ndim==3:
                arr=vals[0,:,1] if vals.shape[2]>1 else vals[0,:,0]
            elif vals.ndim==2:
                arr=vals[0]
            else:
                arr=vals.reshape(-1)
        arr=np.asarray(arr,dtype=float).reshape(-1)
        if arr.size != len(features):
            raise ValueError(f"SHAP returned {arr.size} values for {len(features)} features")
        items=[]
        for feature,impact in zip(features,arr):
            value=student.get(feature)
            numeric_value=_safe_live_float(value)
            items.append({"feature":feature,"value":numeric_value if numeric_value is not None else value,"impact":round(float(impact),6)})
        items.sort(key=lambda x:abs(x["impact"]),reverse=True)
        prediction=_live_predict_student(student_id)
        return {"student_id":student_id,"current_semester":sem,"features":items,"prediction":prediction,"message":"SHAP explanation generated from the live Sem 6 model."}
    except Exception as error:
        return {"student_id":student_id,"current_semester":sem,"features":[],"message":f"SHAP unavailable: {error}"}



@app.get("/faculty/dashboard-overview", dependencies=[Depends(require_faculty)])
def faculty_dashboard_overview():
    """Fast, cached live dashboard snapshot. All values are verified records or model predictions."""
    import time
    now=time.monotonic()
    cached=_FACULTY_OVERVIEW_CACHE.get("data")
    if cached is not None and now-_FACULTY_OVERVIEW_CACHE.get("created",0.0) < _FACULTY_OVERVIEW_TTL:
        return cached
    profiles = _live_list_students().get("students", [])
    sem = max((_live_current_semester(p) for p in profiles), default=6)
    estimator, cfg = _live_load_model(sem)
    if estimator is None:
        raise HTTPException(status_code=503, detail=f"Semester {sem} placement model is unavailable.")
    features = cfg.get("features") or LIVE_SEMESTER_FEATURES[sem]
    rows=[]
    # Batch prediction is substantially faster than loading/predicting one student at a time.
    frame=pd.DataFrame([[student.get(f) for f in features] for student in profiles],columns=features).apply(pd.to_numeric,errors="coerce")
    try:
        preds=estimator.predict(frame)
        probs_all=estimator.predict_proba(frame) if hasattr(estimator,"predict_proba") else None
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Dashboard model error: {error}")
    classes=list(getattr(estimator,"classes_",[]))
    placed_index=None
    if probs_all is not None:
        mapping={str(c).strip().lower():i for i,c in enumerate(classes)}
        placed_index=mapping.get("placed",mapping.get("1"))
        if placed_index is None and probs_all.shape[1]==2: placed_index=1
    global _LIVE_KMEANS_CACHE
    km=scaler=labels=None; cluster_features=[]
    if _LIVE_KMEANS_CACHE is None:
        km_path=LIVE_MODEL_DIR/"kmeans_model.pkl"; sc_path=LIVE_MODEL_DIR/"kmeans_scaler.pkl"; labels_path=LIVE_MODEL_DIR/"cluster_labels.pkl"; cfg_path=LIVE_MODEL_DIR/"kmeans_feature_config.json"
        if km_path.exists() and sc_path.exists() and labels_path.exists() and cfg_path.exists():
            _LIVE_KMEANS_CACHE=(joblib.load(km_path),joblib.load(sc_path),joblib.load(labels_path),json.loads(cfg_path.read_text(encoding="utf-8")).get("features",LIVE_SEMESTER_FEATURES[6]))
        else:
            _LIVE_KMEANS_CACHE=(None,None,None,[])
    km,scaler,labels,cluster_features=_LIVE_KMEANS_CACHE
    cluster_values=None
    if km is not None and cluster_features:
        cframe=pd.DataFrame([[student.get(f) for f in cluster_features] for student in profiles],columns=cluster_features).apply(pd.to_numeric,errors="coerce")
        cframe=cframe.fillna(pd.Series(getattr(scaler,"mean_",np.zeros(len(cluster_features))),index=cluster_features))
        cluster_values=km.predict(scaler.transform(cframe))
    rows=[]
    for i,student in enumerate(profiles):
        pred=preds[i]
        placed=float(probs_all[i,placed_index]) if probs_all is not None and placed_index is not None else None
        status="Placed" if str(pred).strip().lower() in {"placed","1","true"} else "Not Placed"
        cluster_number=int(cluster_values[i]) if cluster_values is not None else None
        cluster_name=str(labels.get(cluster_number,f"Cluster {cluster_number}")) if cluster_number is not None else None
        rows.append({"student_id":student.get("student_id"),"student_name":student.get("Student_Name"),"roll_number":student.get("Roll_Number"),"cgpa":_safe_live_float(student.get("CGPA")),"technical_skill_score":_safe_live_float(student.get("Technical_Skill_Score")),"amcat_score":_safe_live_float(student.get("AMCAT_Score")),"placement_status":status,"placed_probability":round(placed*100,2) if placed is not None else None,"readiness_level":_readiness_from_probability_live(placed),"cluster_number":cluster_number,"cluster_name":cluster_name})
    placed=sum(1 for r in rows if r["placement_status"]=="Placed")
    not_placed=len(rows)-placed
    probs=[r["placed_probability"] for r in rows if r["placed_probability"] is not None]
    readiness={}; clusters={}
    for r in rows:
        readiness[r["readiness_level"]]=readiness.get(r["readiness_level"],0)+1
        if r["cluster_name"]: clusters[r["cluster_name"]]=clusters.get(r["cluster_name"],0)+1
    bands={"<6":0,"6-7":0,"7-8":0,"8-9":0,"9-10":0}
    for p in profiles:
        c=_safe_live_float(p.get("CGPA"))
        if c is None: continue
        if c<6: bands["<6"]+=1
        elif c<7: bands["6-7"]+=1
        elif c<8: bands["7-8"]+=1
        elif c<9: bands["8-9"]+=1
        else: bands["9-10"]+=1
    result={"current_semester":sem,"student_count":len(rows),"predicted_placed_count":placed,"predicted_not_placed_count":not_placed,"predicted_placement_rate":round(placed/len(rows)*100,2) if rows else 0,"average_placement_probability":round(sum(probs)/len(probs),2) if probs else None,"readiness_distribution":readiness,"cluster_distribution":clusters,"cgpa_distribution":bands,"students":rows,"placement_activity":[],"availability_note":"Placement camp schedules, historical placement outcomes, company-wise placements and alumni records are not present in the live dataset/API; the dashboard therefore shows model predictions and verified student metrics only."}
    _FACULTY_OVERVIEW_CACHE.update({"data":result,"created":now})
    return result

# ==================================================
# FRONTEND COMPATIBILITY ROUTES
# ==================================================

@app.get("/students/{student_id}/prediction", dependencies=[Depends(require_student_or_faculty)])
def frontend_prediction(student_id: int):
    return _live_predict_student(student_id)


@app.get("/students/{student_id}/cluster", dependencies=[Depends(require_student_or_faculty)])
def frontend_cluster(student_id: int):
    return _live_cluster_student(student_id)


@app.get("/students/{student_id}/shap", dependencies=[Depends(require_student_or_faculty)])
def frontend_shap(student_id: int):
    return _live_shap_student(student_id)


@app.get("/students/{student_id}/placement-intelligence", dependencies=[Depends(require_student_or_faculty)])
def frontend_placement_intelligence(student_id: int):
    return {"student_id":student_id,"prediction":_live_predict_student(student_id),"shap":_live_shap_student(student_id)}


@app.post("/students/import-excel", dependencies=[Depends(require_faculty)])
async def frontend_import_excel(file: UploadFile = File(...)):
    suffix=Path(file.filename or "dataset.xlsx").suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        tmp.write(await file.read()); temp_path=Path(tmp.name)
    try:
        count=_sync_live_dataset_to_db(temp_path, overwrite_existing=True)
        _invalidate_live_dashboard_cache()
        return {"message":"Excel imported successfully.","imported_students":count}
    finally:
        try: temp_path.unlink()
        except OSError: pass


# ==================================================
# END OF PRESERVED ORIGINAL + COMPATIBILITY ADDITIONS
# ==================================================

# ==================================================
# TARGETED UPDATE AUDIT NOTE
# ==================================================
# This file is based on the exact original 8,995-line backend.
# The only functional change in this audit revision is student
# registration: it must link to an existing live-dataset student
# instead of creating a blank student with zero placeholder values.
# All original API routes, ML logic, portfolio logic, Gemini agents,
# reports, imports, analytics, clustering and SHAP code remain present.
# The live dataset continues to provide the student records.
# Sem 1 through Sem 6 are the available live academic semesters.
# Derived semester values are calculated in Python where required.
# Missing live metrics are not fabricated by this revision.
# Placement_Status, Package_LPA and Company are not model inputs.
# Existing model artifacts are packaged separately with this backend.
# CORS support for localhost:8080 is retained in the original backend.
# No frontend code is embedded in this backend file.
# No frontend mock data is used by the connected frontend package.
# The frontend calls the existing FastAPI compatibility routes.
# Project, internship, certification and achievement CRUD routes remain.
# Resume generation remains backed by the existing Gemini endpoint.
# Resume export renders the supplied generated resume payload.
# SHAP values are read from the existing live explanation endpoint.
# Faculty overview uses the existing live dashboard endpoint.
# End of targeted-update audit note.
