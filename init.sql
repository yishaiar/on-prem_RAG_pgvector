CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
  -- id SERIAL PRIMARY KEY,--old standard
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  embedding vector,
  content TEXT,
  created_at timestamptz DEFAULT now()
);
