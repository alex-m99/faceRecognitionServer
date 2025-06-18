import sys
import os
sys.path.append(os.path.abspath('..'))
from fastapi import FastAPI, status, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request, Body, Header
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
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta

#CHANGE IT LATER!!!
SECRET_KEY = "your-secret-key" 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


app = FastAPI()
# fd = FaceData()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


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

def generate_system_token():
    return uuid4().hex

def hash_token(token: str) -> str:
    return pwd_context.hash(token)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_current_admin(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    admin = db.query(models.Admin).filter(models.Admin.username == username).first()
    if admin is None:
        raise credentials_exception
    return admin


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


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    admin = db.query(models.Admin).filter(models.Admin.username == form_data.username).first()
    if not admin or not verify_password(form_data.password, admin.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": admin.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/notify")
async def notify(request: Request):
    data = await request.json()
    await logs_manager.broadcast(data)
    return {"message": "Notification sent"}

@app.get('/')
def index():
    return 'Hello world!'

@app.post('/person',  status_code=status.HTTP_201_CREATED)
def add_person(
    firstName: str = Form(...), 
    lastName: str = Form(...), 
    function: str = Form(...),
    email: str = Form(...),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin)
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
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin)
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
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin)
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
def get_people(db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
    people = db.query(models.Person).all()

    # Parse the `encoding` JSON string into actual lists
    results = []
    for person in people:
        #encoding_list = json.loads(person.encoding) 
        results.append(schemas.DisplayPerson(id=person.id, firstName=person.firstName, lastName=person.lastName, function = person.function, email = person.email, photo_url=person.photo_url))

    return results

    
@app.get('/{system_id}/encodings', response_model=List[schemas.PersonWithEncoding],)
def get_people_with_encodings(
    system_id: int, 
    db: Session = Depends(get_db), 
):
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
async def recognition_login(
    name: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db)
):
    system = db.query(models.System).filter(models.System.name == name).first()
    if not system or not pwd_context.verify(password, system.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    system.started = True
    db.commit() 
    await logs_manager.broadcast({ "event": "system_started" })
    return {"success": True, "system_id": system.id}


####################################################################


@app.get("/systems")
def get_systems(db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
    return db.query(models.System).all()

@app.get("/systems/{system_id}/people")
def get_people_for_system(system_id: int, db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
    people = (
        db.query(models.Person)
        .join(models.PersonSystem, models.Person.id == models.PersonSystem.person_id)
        .filter(models.PersonSystem.system_id == system_id)
        .all()
    )
    return people

@app.post("/systems")
def add_system(
    name: str = Form(...),
    password: str = Form(...),
    lock_password: str = Form(...),
    address: str = Form(...),
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin)
):
    hashed_password = hash_password(password)
    hashed_lock_password = hash_password(lock_password)
    plain_token = uuid4().hex
    hashed_token = hash_token(plain_token)
    new_system = models.System(
        name=name,
        password=hashed_password,
        lock_password=hashed_lock_password,
        address=address,
        started=False,
        starting_date=datetime.utcnow(),
        system_token=hashed_token
    )
    db.add(new_system)
    db.commit()
    db.refresh(new_system)
    # Return the plain token only once!
    return {
        "id": new_system.id,
        "name": new_system.name,
        "address": new_system.address,
        "system_token": plain_token
    }


@app.post("/systems/{system_id}/people", status_code=status.HTTP_201_CREATED)
def add_person_to_system(
    system_id: int,
    person_id: int = Body(...),
    access: bool = Body(True),
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin)
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
def start_system(
    system_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin)
):
    system = db.query(models.System).get(system_id)
    if not system:
        raise HTTPException(status_code=404)
    system.started = True
    plain_token = uuid4().hex
    hashed_token = hash_token(plain_token)
    system.system_token = hashed_token
    db.commit()
    asyncio.run(updates_manager.broadcast(system_id, {"event": "system_started"}))
    # Return the new plain token only once!
    return {"success": True, "system_token": plain_token}

@app.patch("/systems/{system_id}/stop")
def stop_system(system_id: int, db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
    system = db.query(models.System).get(system_id)
    if not system:
        raise HTTPException(status_code=404)
    system.started = False
    db.commit()
    asyncio.run(updates_manager.broadcast(system_id, {"event": "system_stopped"}))
    return {"success": True}

@app.patch("/systems/{system_id}/logout")
def logout_system(system_id: int, db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
    system = db.query(models.System).get(system_id)
    if not system:
        raise HTTPException(status_code=404)
    system.started = False
    db.commit()
    asyncio.run(updates_manager.broadcast(system_id, {"event": "system_logout"}))
    return {"success": True}

@app.get("/systems/{system_id}/system-token")
def get_system_token(
    system_id: int,
    lock_token: str = Header(..., alias="X-Lock-Token"),
    db: Session = Depends(get_db)
):
    system = db.query(models.System).get(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")
    if not system.lock_token or lock_token != system.lock_token:
        raise HTTPException(status_code=403, detail="Invalid lock token")
    return {"system_token": system.system_token}

@app.patch("/systems/{system_id}/people/{person_id}/access")
def set_person_access_in_system(
    system_id: int,
    person_id: int,
    access: bool = Body(...),
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin)
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
    lock_password: str = Form(None),
    address: str = Form(...),
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin)
):
    system = db.query(models.System).get(system_id)
    if not system:
        raise HTTPException(status_code=404)
    system.name = name
    system.address = address
    if password:
        system.password = hash_password(password)
    if lock_password:
        system.lock_password = hash_password(lock_password)
    db.commit()
    return {"success": True}

@app.delete("/systems/{system_id}")
def delete_system(system_id: int, db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
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
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin)
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



@app.post("/create-admin")
def create_admin(username: str = Body(...), password: str = Body(...), db: Session = Depends(get_db)):
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    hashed = pwd_context.hash(password)
    admin = models.Admin(username=username, password_hash=hashed)
    db.add(admin)
    db.commit()
    return {"username": username}