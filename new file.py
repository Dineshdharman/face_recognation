import face_recognition
import numpy as np
import mysql.connector

# Load an image
image = face_recognition.load_image_file(r"C:\Users\HP\PycharmProjects\pythonProject7\photos\WhatsApp Image 2024-07-21 at 14.56.35_adbb17c7.jpg")
face_encoding = face_recognition.face_encodings(image)[0]

# Ensure face encoding is in np.float64 format
face_encoding = np.array(face_encoding, dtype=np.float64)

# Convert face encoding to binary format
face_encoding_blob = face_encoding.tobytes()

def create_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',  # Replace with your MySQL username
        password='Dine@2003',  # Replace with your MySQL password
        database='face_recognition')

def insert_customer(conn, unique_id, name, email, face_encoding_blob):
    cursor = conn.cursor()
    query = '''INSERT INTO customers (unique_id, name, email, face_encoding) VALUES (%s, %s, %s, %s)'''
    cursor.execute(query, (unique_id, name, email, face_encoding_blob))
    conn.commit()
conn = create_connection()

# Example face encoding
face_encoding = np.random.rand(128).astype(np.float64)  # Placeholder for actual face encoding
face_encoding_blob = face_encoding.tobytes()

insert_customer(conn, 5, 'Dharneesh', 'dharneeshbe@gmail.com', face_encoding_blob)
