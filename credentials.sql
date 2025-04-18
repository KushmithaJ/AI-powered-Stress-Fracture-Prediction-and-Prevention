-- Create a table to store user credentials
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert a sample user
INSERT INTO users (username, password_hash, email)
VALUES ('example_user', 'hashed_password', 'user@example.com');

-- Retrieve all users
SELECT * FROM users;
