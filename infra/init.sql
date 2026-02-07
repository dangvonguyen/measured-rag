-- Enable vector support
CREATE EXTENSION IF NOT EXISTS "vector";

-- Enable UUID generation for primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Log extension status for container startup verification
DO $$
BEGIN
    RAISE NOTICE 'Extension vector: %', (SELECT installed_version FROM pg_available_extensions WHERE name = 'vector');
    RAISE NOTICE 'Extension uuid-ossp: %', (SELECT installed_version FROM pg_available_extensions WHERE name = 'uuid-ossp');
END $$;
