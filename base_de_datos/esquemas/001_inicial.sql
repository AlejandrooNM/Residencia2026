-- Esquema inicial de la base de datos del sistema de apoyo diagnóstico IAM.
-- Compatible con SQLite (desarrollo) y adaptable a PostgreSQL.

CREATE TABLE IF NOT EXISTS registros_analisis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre_archivo VARCHAR(255) NOT NULL,
    etiqueta VARCHAR(50) NOT NULL,
    probabilidad_iam REAL NOT NULL CHECK (probabilidad_iam >= 0 AND probabilidad_iam <= 1),
    confianza REAL NOT NULL CHECK (confianza >= 0 AND confianza <= 1),
    mensaje TEXT NOT NULL,
    creado_en DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS indice_registros_creado_en
    ON registros_analisis (creado_en DESC);

CREATE INDEX IF NOT EXISTS indice_registros_etiqueta
    ON registros_analisis (etiqueta);
