import sqlite3

conn = sqlite3.connect('database.db')
conn.row_factory = sqlite3.Row

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', [t[0] for t in tables])

try:
    count = conn.execute('SELECT COUNT(*) FROM analysis_history').fetchone()[0]
    print(f'analysis_history rows: {count}')
except Exception as e:
    print(f'analysis_history error: {e}')

try:
    unk = conn.execute("SELECT student_id, name FROM students WHERE student_id='UNKNOWN'").fetchone()
    if unk:
        print(f'UNKNOWN student: id={unk[0]}, name={unk[1]}')
    else:
        print('UNKNOWN student: NOT FOUND')
except Exception as e:
    print(f'UNKNOWN student error: {e}')

conn.close()
