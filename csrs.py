import mysql.connector
import cv2
import dlib
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from twilio.rest import Client


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


# Add sample data for initial training
def create_sample_data(conn):
    # Generate some random sample encodings
    sample_encodings = [np.random.rand(128).astype(np.float64).tobytes() for _ in range(10)]

    for encoding in sample_encodings:
        insert_customer(conn, encoding)


create_sample_data(conn)

# Twilio configuration
account_sid = 'AC2275165686cfb0b480a304b430e9bf23'
auth_token = '7448beabd312c7147ceb2b2894f9e069'
twilio_client = Client(account_sid, auth_token)


def send_message(customer_id):
    try:
        message = twilio_client.messages.create(
            body=f"Thank you for visiting our shop, Customer {customer_id}!",
            from_='+8610189065',
            to='+916382250835'
        )
        if message.sid:
            print(f"Message sent successfully to Customer {customer_id}. Message SID: {message.sid}")
            return message.sid
    except Exception as e:
        print(f"Failed to send message to Customer {customer_id}. Error: {e}")
        return None


def get_message_status(message_sid):
    try:
        message = twilio_client.messages(message_sid).fetch()
        return message.status
    except Exception as e:
        print(f"Failed to fetch message status. Error: {e}")
        return None


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
            message_sid = send_message(customer_id)
            if message_sid:
                status = get_message_status(message_sid)
                print(f"Message status: {status}")
            customers = fetch_customers(conn)  # Update the customer list
            knn_model = train_knn_model(customers)  # Retrain the model

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
