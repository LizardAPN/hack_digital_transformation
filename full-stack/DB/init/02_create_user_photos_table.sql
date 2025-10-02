CREATE TABLE IF NOT EXISTS user_photos (
    id SERIAL PRIMARY KEY,
    owner_id INT NOT NULL,
    photo_url VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);