import sqlite3

def ConnectToDB(pathtodb):
    try:
        sqliteConnection = sqlite3.connect(pathtodb)
        cursor = sqliteConnection.cursor()
        print("Database created and Successfully Connected to SQLite")      
        return sqliteConnection

    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)