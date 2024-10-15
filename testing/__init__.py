import mysql.connector
import cv2
import dlib
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Database functions
def create_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',  # Replace with your MySQL username
        password='Dine@2003',  # Replace with your MySQL password
        database='face_recognition'
    )

def create_table(conn):
    cursor = conn.cursor()
    query = '''CREATE TABLE IF NOT EXISTS customers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    face_encoding BLOB NOT NULL
                );'''
    cursor.execute(query)
    conn.commit()

def insert_customer(conn, face_encoding):
    cursor = conn.cursor()
    query = '''INSERT INTO customers (face_encoding) VALUES (%s)'''
    cursor.execute(query, (face_encoding,))
    conn.commit()

def fetch_customers(conn):
    cursor = conn.cursor()
    query = '''SELECT * FROM customers'''
    cursor.execute(query)
    return cursor.fetchall()

# Initialize database connection and create table if it doesn't exist
conn = create_connection()
create_table(conn)

# SMTP configuration
def send_email(customer_id, recipient_email):
    sender_email = 'dineshdev.be@gmail.com'  # Replace with your Gmail address
    app_password = 'pevv gpfq whiu bvqd'  # Replace with your Gmail app password

    subject = 'Thank You for Visiting Our Shop'
    body = f"Thank you for visiting our shop, Customer {customer_id}!"

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    try:
        print(f"Attempting to send email to {recipient_email}")
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}. Error: {e}")

def is_existing_customer(face_encoding, customers):
    for customer in customers:
        db_encoding = np.frombuffer(customer[1], dtype=np.float64)
        if face_recognition.compare_faces([db_encoding], face_encoding, tolerance=0.6)[0]:
            return True
    return False

def save_new_customer(face_encoding):
    encoding_blob = face_encoding.tobytes()
    insert_customer(conn, encoding_blob)

def train_knn_model(customers):
    X = [np.frombuffer(customer[1], dtype=np.float64) for customer in customers]
    y = [customer[0] for customer in customers]

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X, y)
    return knn

# Fetch existing customers and train KNN model
customers = fetch_customers(conn)
knn_model = train_knn_model(customers)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video stream
cap = cv2.VideoCapture(0)

# Cache for the first 50 customers
cached_customers = set()

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
        customer_id = knn_model.predict([face_encoding])

        if is_existing_customer(face_encoding, customers):
            print(f"Existing customer detected. Customer ID: {customer_id}")
        else:
            print("New customer detected. Saving to database.")
            save_new_customer(face_encoding)

            # Check if customer is in the cache
            if customer_id[0] not in cached_customers:
                recipient_email = 'dhineshdharman007@gmail.com'  # Update this as necessary
                print(f"Sending email to {recipient_email}")
                send_email(customer_id[0], recipient_email)

                # Add to cache and limit to 50 customers
                if len(cached_customers) >= 50:
                    cached_customers.pop()  # Remove an old customer from the cache
                cached_customers.add(customer_id[0])

            customers = fetch_customers(conn)  # Update the customer list
            knn_model = train_knn_model(customers)  # Retrain the model

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
