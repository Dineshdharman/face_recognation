import mysql.connector
import cv2
import dlib
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid

# Database functions
def create_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='Dine@2003',
        database='face_recognition'
    )

def insert_customer(conn, unique_id, name, email, face_encoding):
    cursor = conn.cursor()
    query = '''INSERT INTO customers (unique_id, name, email, face_encoding) VALUES (%s, %s, %s, %s)'''
    cursor.execute(query, (unique_id, name, email, face_encoding))
    conn.commit()

def fetch_customers(conn):
    cursor = conn.cursor()
    query = '''SELECT * FROM customers'''
    cursor.execute(query)
    return cursor.fetchall()

conn = create_connection()

# SMTP configuration
def send_email(customer_id, recipient_email, customer_name):
    sender_email = 'dineshdev.be@gmail.com'
    password = 'pevv gpfq whiu bvqd'

    subject = 'Thank You for Visiting Our Shop'
    body = f"Thank you for visiting our shop, {customer_name} (Customer ID: {customer_id})!"

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}. Error: {e}")

def is_existing_customer(face_encoding, customers):
    for customer in customers:
        db_encoding = np.frombuffer(customer[4], dtype=np.float64)
        if face_recognition.compare_faces([db_encoding], face_encoding, tolerance=0.6)[0]:
            return True, customer
    return False, None

def save_new_customer(face_encoding):
    unique_id = str(uuid.uuid4())  # Generate a unique ID
    name = input("Enter the customer's name: ")
    email = input("Enter the customer's email: ")
    encoding_blob = face_encoding.tobytes()
    insert_customer(conn, unique_id, name, email, encoding_blob)
    return unique_id, name, email

def train_knn_model(customers):
    X = [np.frombuffer(customer[4], dtype=np.float64) for customer in customers]
    y = [customer[0] for customer in customers]

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X, y)
    return knn

# Fetch existing customers and train KNN model
customers = fetch_customers(conn)
knn_model = train_knn_model(customers)

# Initialize dlib's face detector and facial landmark pDredictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video stream
cap = cv2.VideoCapture(0)

# Cache to store IDs of customers who received emails today
email_cache = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (face_recognition uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)

    for face_location in face_locations:
        # Get the facial encoding
        face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]

        # Predict using KNN model
        customer_id = knn_model.predict([face_encoding])[0]

        # Check if the customer is existing or new
        is_existing, customer = is_existing_customer(face_encoding, customers)

        if is_existing:
            customer_id = customer[0]
            customer_name = customer[2]
            print(f"Existing customer detected. Customer ID: {customer_id}, Name: {customer_name}")
            if customer[3] not in email_cache:  # Check if email has been sent today
                send_email(customer_id, customer[3], customer_name)
                email_cache.add(customer[3])  # Cache email
        else:
            print("New customer detected. Saving to database.")
            unique_id, name, email = save_new_customer(face_encoding)
            send_email(unique_id, email, name)
            email_cache.add(email)  # Cache email
            customers = fetch_customers(conn)  # Update the customer list
            knn_model = train_knn_model(customers)  # Retrain the model

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()