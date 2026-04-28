from app.database import get_student
s = get_student('STU003')
if s:
    print(f"Name: {s.get('name')}")
    emb = s.get('embedding')
    if emb:
        print(f"Embedding length: {len(emb)}")
    else:
        print("No embedding found.")
else:
    print("Student not found.")
