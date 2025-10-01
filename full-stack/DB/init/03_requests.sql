CREATE TABLE IF NOT EXISTS processing_results (
    id SERIAL PRIMARY KEY,
    image_path TEXT NOT NULL,
    task_id VARCHAR(255),
    request_id VARCHAR(255),
    coordinates JSONB,
    address TEXT,
    ocr_result TEXT,
    buildings JSONB,
    processed_at TIMESTAMPTZ DEFAULT now(),
    error TEXT,
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
);
