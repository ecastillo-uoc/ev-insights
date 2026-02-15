
-- Connect as superuser
\c postgres

-- Destroy the databases if they already exist
DROP DATABASE IF EXISTS mlflow;
DROP DATABASE IF EXISTS evinsights;

-- Drop users if they exist
DROP USER IF EXISTS mlflow;
DROP USER IF EXISTS evinsights;


-- Create db
CREATE DATABASE mlflow;
CREATE DATABASE evinsights;

-- Create user
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE USER evinsights WITH PASSWORD 'evinsights';

-- Connect as mlflow user to the mlflow database to create the schema and assign permissions
\c mlflow postgres

-- Create mlflow schema
CREATE SCHEMA mlflow AUTHORIZATION mlflow;

-- Assigning permissions to the mlflow user.
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
GRANT ALL PRIVILEGES ON SCHEMA mlflow TO mlflow;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA mlflow TO mlflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA mlflow TO mlflow;


-- Connect as evinsights user to the evinsights database to create the schema and assign permissions
\c evinsights postgres

-- Create evinsights schema
CREATE SCHEMA evinsights AUTHORIZATION evinsights;

-- Assigning permissions to the evinsights user
GRANT ALL PRIVILEGES ON DATABASE evinsights TO evinsights;
GRANT ALL PRIVILEGES ON SCHEMA evinsights TO evinsights;


-- It seems that although the evinsights user has permissions on the schema, the tables (specifically Dataset) were likely created by the postgres superuser (or another user), and evinsights wasn't granted explicit permissions on those specific tables. In PostgreSQL, granting permissions on a schema does not automatically grant permissions on existing tables within it, nor on future tables created by other users unless ALTER DEFAULT PRIVILEGES is used.

-- This ensures that any tables created by the postgres user in the evinsights schema will automatically be accessible to the evinsights user.

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA evinsights TO evinsights;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA evinsights TO evinsights;
