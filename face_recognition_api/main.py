import sys
import os
sys.path.append(os.path.abspath('..'))
from fastapi import FastAPI, status, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from fastapi.params import Depends
from .import models
from .import schemas
from .database import engine, SessionLocal
from face_data import FaceData
import json
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
import cv2

app = FastAPI()
# fd = FaceData()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

models.Base.metadata.create_all(engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/notify")
async def notify(request: Request):
    data = await request.json()
    await manager.broadcast(data)
    return {"message": "Notification sent"}

@app.get('/')
def index():
    return 'Hello world!'

#momentan extrage datele faciale din pozele din face_pictures si le scrie in baza de date
@app.post('/person', status_code=status.HTTP_201_CREATED)
def add_person(
    firstName: str = Form(...), 
    lastName: str = Form(...), 
    function: str = Form(...),
    email: str = Form(...),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    # Read the uploaded file
    file_bytes = np.frombuffer(photo.file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Extract face encodings
    face_encodings = face_recognition.face_encodings(img)
    if not face_encodings:
        raise HTTPException(status_code=400, detail="No face found in the uploaded photo.")

    encoding = face_encodings[0].tolist()
    encoding_json = json.dumps(encoding)

    new_person = models.Person(
        firstName=firstName,
        lastName=lastName,
        function=function,
        email=email,
        encoding=encoding_json
    )

    db.add(new_person)
    db.commit()
    db.refresh(new_person)
    return {'person added'}

    #encodings = fd.known_face_encondings

    # if not encodings:
    #     raise HTTPException(status_code=404, detail="No face found in the image")
    
  
    # encoding_bill = encodings[0].tolist()
    # encoding_json_bill = json.dumps(encoding_bill)

    # new_person_bill = models.Person(firstName="Bill", lastName="Gates", function = "Patronache", email = "bill_smecheru@gmail.com", encoding=encoding_json_bill)

    # db.add(new_person_bill)
    # db.commit()
    # db.refresh(new_person_bill)
    # ###########################################################################
   


@app.get('/people', response_model=List[schemas.DisplayPerson])
def get_people(db: Session = Depends(get_db)):
    people = db.query(models.Person).all()

    # Parse the `encoding` JSON string into actual lists
    results = []
    for person in people:
        #encoding_list = json.loads(person.encoding) 
        results.append(schemas.DisplayPerson(id=person.id, firstName=person.firstName, lastName=person.lastName, function = person.function, email = person.email))

    return results

    
@app.get('/people/encodings', response_model=List[schemas.PersonWithEncoding])
def get_people_with_encodings(db: Session = Depends(get_db)):
    people = db.query(models.Person).all()
    results = []
    for person in people:
        results.append(
            schemas.PersonWithEncoding(
                id=person.id,
                firstName=person.firstName,
                lastName=person.lastName,
                function=person.function,
                email=person.email,
                encoding=json.loads(person.encoding)
            )
        )
    return results

