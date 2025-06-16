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
from datetime import datetime
from passlib.context import CryptContext


app = FastAPI()
# fd = FaceData()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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


def hash_password(password: str) -> str:
    return pwd_context.hash(password)

class ConnectionManagerFrontend:
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

class ConnectionManagerFaceApp:
    def __init__(self):
        self.system_connections: dict[int, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, system_id: int):
        # await websocket.accept()
        if system_id not in self.system_connections:
            self.system_connections[system_id] = []
        self.system_connections[system_id].append(websocket)

    def disconnect(self, websocket: WebSocket):
        for ws_list in self.system_connections.values():
            if websocket in ws_list:
                ws_list.remove(websocket)

    async def broadcast(self, system_id: int, message: dict):
        for ws in self.system_connections.get(system_id, []):
            await ws.send_json(message)

logs_manager = ConnectionManagerFrontend()
updates_manager = ConnectionManagerFaceApp()


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
    await websocket.accept()
    # Wait for the client to send {"system_id": ...}
    data = await websocket.receive_json()
    system_id = data.get("system_id")
    print(system_id)
    if not isinstance(system_id, int):
        await websocket.close()
        return
    await updates_manager.connect(websocket, system_id)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        updates_manager.disconnect(websocket)

@app.post("/notify")
async def notify(request: Request):
    data = await request.json()
    await logs_manager.broadcast(data)
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


# @app.patch('/person/{id}/access', status_code=status.HTTP_200_OK)
# def set_person_access(
#     id: int,
#     access: bool = Body(...),
#     db: Session = Depends(get_db)
# ):
#     person = db.query(models.Person).filter(models.Person.id == id).first()
#     if not person:
#         raise HTTPException(status_code=404, detail="Person not found")
#     person.access = access
#     db.commit()
#     db.refresh(person)
#     return {"id": person.id, "access": person.access}

@app.delete('/person/{id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_person(
    id: int,
    db: Session = Depends(get_db)
):
    person = db.query(models.Person).filter(models.Person.id == id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Delete all PersonSystem entries for this person
    db.query(models.PersonSystem).filter(models.PersonSystem.person_id == id).delete()

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
        results.append(schemas.DisplayPerson(id=person.id, firstName=person.firstName, lastName=person.lastName, function = person.function, email = person.email, photo_url=person.photo_url))

    return results

    
@app.get('/{system_id}/encodings', response_model=List[schemas.PersonWithEncoding])
def get_people_with_encodings(system_id: int, db: Session = Depends(get_db)):
    people = (
        db.query(models.Person)
        .join(models.PersonSystem, models.Person.id == models.PersonSystem.person_id)
        .filter(models.PersonSystem.system_id == system_id, models.PersonSystem.access == True)
        .all()
    )
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

@app.post("/recognition-login")
def recognition_login(
    name: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db)
):
    system = db.query(models.System).filter(models.System.name == name).first()
    if not system or not pwd_context.verify(password, system.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    system.started = True
    return {"success": True, "system_id": system.id}


####################################################################


@app.get("/systems")
def get_systems(db: Session = Depends(get_db)):
    return db.query(models.System).all()

@app.get("/systems/{system_id}/people")
def get_people_for_system(system_id: int, db: Session = Depends(get_db)):
    people = (
        db.query(models.Person)
        .join(models.PersonSystem, models.Person.id == models.PersonSystem.person_id)
        .filter(models.PersonSystem.system_id == system_id)
        .all()
    )
    return people

@app.post("/systems")
def add_system(name: str = Form(...), password: str = Form(...), address: str = Form(...), db: Session = Depends(get_db)):
    hashed_password = hash_password(password)
    new_system = models.System(name=name, password = hashed_password, address=address, started = False, starting_date=datetime.utcnow())
    db.add(new_system)
    db.commit()
    db.refresh(new_system)
    return {"id": new_system.id, "name": new_system.name, "address": new_system.address}


@app.post("/systems/{system_id}/people", status_code=status.HTTP_201_CREATED)
def add_person_to_system(
    system_id: int,
    person_id: int = Body(...),
    access: bool = Body(True),
    db: Session = Depends(get_db)
):
    person_system = models.PersonSystem(
        person_id=person_id,
        system_id=system_id,
        access=access
    )
    db.add(person_system)
    db.commit()
  
    asyncio.run(updates_manager.broadcast(system_id, {"event": "update_encodings"}))
    return {"person_id": person_id, "system_id": system_id, "access": access}

@app.patch("/systems/{system_id}/start")
def start_system(system_id: int, db: Session = Depends(get_db)):
    system = db.query(models.System).get(system_id)
    if not system:
        raise HTTPException(status_code=404)
    system.started = True
    db.commit()
    return {"success": True}

@app.patch("/systems/{system_id}/stop")
def stop_system(system_id: int, db: Session = Depends(get_db)):
    system = db.query(models.System).get(system_id)
    if not system:
        raise HTTPException(status_code=404)
    system.started = False
    db.commit()
    return {"success": True}

@app.patch("/systems/{system_id}/people/{person_id}/access")
def set_person_access_in_system(
    system_id: int,
    person_id: int,
    access: bool = Body(...),
    db: Session = Depends(get_db)
):
    person_system = db.query(models.PersonSystem).filter(
        models.PersonSystem.system_id == system_id,
        models.PersonSystem.person_id == person_id
    ).first()
    if not person_system:
        raise HTTPException(status_code=404, detail="PersonSystem entry not found")
    person_system.access = access
    db.commit()
    db.refresh(person_system)
    asyncio.run(updates_manager.broadcast(system_id, {"event": "update_encodings"}))
    return {"system_id": system_id, "person_id": person_id, "access": person_system.access}

@app.put("/systems/{system_id}")
def edit_system(
    system_id: int,
    name: str = Form(...),
    password: str = Form(None),
    address: str = Form(...),
    db: Session = Depends(get_db)
):
    system = db.query(models.System).get(system_id)
    if not system:
        raise HTTPException(status_code=404)
    system.name = name
    system.address = address
    if password:
        system.password = hash_password(password)
    db.commit()
    return {"success": True}

@app.delete("/systems/{system_id}")
def delete_system(system_id: int, db: Session = Depends(get_db)):
    system = db.query(models.System).get(system_id)
    if not system:
        raise HTTPException(status_code=404)
    
    # Delete all PersonSystem entries for this system
    db.query(models.PersonSystem).filter(models.PersonSystem.system_id == system_id).delete()
    
    db.delete(system)
    db.commit()
    return {"success": True}


@app.delete("/systems/{system_id}/people/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person_from_system(
    system_id: int,
    id: int,
    db: Session = Depends(get_db)
):
    person_system = db.query(models.PersonSystem).filter(
        models.PersonSystem.system_id == system_id,
        models.PersonSystem.person_id == id
    ).first()
    if not person_system:
        raise HTTPException(status_code=404, detail="PersonSystem entry not found")
    db.delete(person_system)
    db.commit()
    asyncio.run(updates_manager.broadcast(system_id, {"event": "update_encodings"}))
    return {"message": "Person removed from system"}