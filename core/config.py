class DBSettings:
    path: str = "immersion.db"
    wal_mode: bool = True

class Settings:
    database = DBSettings()

settings = Settings()
