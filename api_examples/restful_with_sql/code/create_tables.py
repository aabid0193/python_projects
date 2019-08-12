import sqlite3

connection = sqlite3.connect('data.db')
cursor = connection.cursor()

create_table = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username text, password text)"
## must use INTEGER as opposed to int only if you want auto incrementing IDs
cursor.execute(create_table)

create_table = "CREATE TABLE IF NOT EXISTS items (name text, price real)"  #real is like float
cursor.execute(create_table)

connection.commit()

connection.close()
