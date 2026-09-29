-- Optional helper: the backend also creates/upgrades these tables automatically.
-- This file does NOT create the existing students table because its schema/data
-- belongs to the original PlacementIQ project.

CREATE TABLE IF NOT EXISTS auth_users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('faculty','student')),
    student_id INTEGER NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
