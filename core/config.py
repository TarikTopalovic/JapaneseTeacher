class DBSettings:
    path: str = "storage/db/immersion.db"
    wal_mode: bool = True

class Settings:
    database = DBSettings()

settings = Settings()
