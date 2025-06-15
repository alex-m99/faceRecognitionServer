import sys
import os
sys.path.append(os.path.abspath('..'))
from fastapi import FastAPI, status, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request, Body
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
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
import asyncio

app = FastAPI()
# fd = FaceData()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.mount(
    "/photos",
    StaticFiles(directory=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'face_pictures'))),
    name="photos"
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

logs_manager = ConnectionManager()
updates_manager = ConnectionManager()


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await logs_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        logs_manager.disconnect(websocket)

@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    await updates_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        updates_manager.disconnect(websocket)

@app.post("/notify")
async def notify(request: Request):
    data = await request.json()
    await manager.broadcast(data)
    return {"message": "Notification sent"}

@app.get('/')
def index():
    return 'Hello world!'

@app.post('/person', status_code=status.HTTP_201_CREATED)
def add_person(
    firstName: str = Form(...), 
    lastName: str = Form(...), 
    function: str = Form(...),
    email: str = Form(...),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    
    #save the uploaded photo
    file_ext = os.path.splitext(photo.filename)[1]
    file_name = f"{uuid4().hex}{file_ext}"  # Generate a unique filename
    file_path = os.path.join(os.path.dirname(__file__), '..', 'face_pictures', file_name)
    file_path = os.path.abspath(file_path)

    #write the photo to the disk
    with open(file_path, "wb") as f:
        f.write(photo.file.read())
    

    # Read the uploaded file
    file_bytes = np.frombuffer(open(file_path, "rb").read(), np.uint8)
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
        photo_url=f"/photos/{file_name}",
        encoding=encoding_json
    )

    db.add(new_person)
    db.commit()
    db.refresh(new_person)
    return {'person added'}


@app.put('/person/{id}', status_code=status.HTTP_200_OK)
def update_person(
    id: int,
    firstName: str = Form(...),
    lastName: str = Form(...),
    function: str = Form(...),
    email: str = Form(...),
    photo: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    person = db.query(models.Person).filter(models.Person.id == id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    person.firstName = firstName
    person.lastName = lastName
    person.function = function
    person.email = email

    if photo is not None:
        # Delete old photo file if it exists
        if person.photo_url:
            old_photo_path = os.path.join(os.path.dirname(__file__), '..', 'face_pictures', os.path.basename(person.photo_url))
            old_photo_path = os.path.abspath(old_photo_path)
            if os.path.exists(old_photo_path):
                os.remove(old_photo_path)

        # Save new photo
        file_ext = os.path.splitext(photo.filename)[1]
        file_name = f"{uuid4().hex}{file_ext}"
        file_path = os.path.join(os.path.dirname(__file__), '..', 'face_pictures', file_name)
        file_path = os.path.abspath(file_path)
        with open(file_path, "wb") as f:
            f.write(photo.file.read())

        # Read and encode new photo
        file_bytes = np.frombuffer(open(file_path, "rb").read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        face_encodings = face_recognition.face_encodings(img)
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face found in the uploaded photo.")
        encoding = face_encodings[0].tolist()
        encoding_json = json.dumps(encoding)

        person.photo_url = f"/photos/{file_name}"
        person.encoding = encoding_json

    db.commit()
    db.refresh(person)
    return {"message": "Person updated"}


@app.patch('/person/{id}/access', status_code=status.HTTP_200_OK)
def set_person_access(
    id: int,
    access: bool = Body(...),
    db: Session = Depends(get_db)
):
    person = db.query(models.Person).filter(models.Person.id == id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    person.access = access
    db.commit()
    db.refresh(person)
    return {"id": person.id, "access": person.access}

@app.delete('/person/{id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_person(
    id: int,
    db: Session = Depends(get_db)
):
    person = db.query(models.Person).filter(models.Person.id == id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Delete photo file if it exists
    if person.photo_url:
        photo_path = os.path.join(os.path.dirname(__file__), '..', 'face_pictures', os.path.basename(person.photo_url))
        photo_path = os.path.abspath(photo_path)
        if os.path.exists(photo_path):
            os.remove(photo_path)

    db.delete(person)
    db.commit()
    return {"message": "Person deleted"}

@app.get('/people', response_model=List[schemas.DisplayPerson])
def get_people(db: Session = Depends(get_db)):
    people = db.query(models.Person).all()

    # Parse the `encoding` JSON string into actual lists
    results = []
    for person in people:
        #encoding_list = json.loads(person.encoding) 
        results.append(schemas.DisplayPerson(id=person.id, access=person.access, firstName=person.firstName, lastName=person.lastName, function = person.function, email = person.email, photo_url=person.photo_url))

    return results

    
@app.get('/people/encodings', response_model=List[schemas.PersonWithEncoding])
def get_people_with_encodings(db: Session = Depends(get_db)):
    people = db.query(models.Person).all()
    results = []
    for person in people:
        results.append(
            schemas.PersonWithEncoding(
                id=person.id,
                access=person.access,
                firstName=person.firstName,
                lastName=person.lastName,
                function=person.function,
                email=person.email,
                encoding=json.loads(person.encoding)
            )
        )
    return results

