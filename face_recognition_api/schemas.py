from pydantic import BaseModel 
from typing import List

class DisplayPerson(BaseModel):
    id: int  
    firstName: str
    lastName: str
    function: str
    email: str
    photo_url: str = None  # Optional field for storing photo URL
    #encoding: List[float]

    # orm_mode = True allows Pydantic to read data 
    # from SQLAlchemy model objects directly.
    class Config:
        orm_mode = True


class PersonWithEncoding(DisplayPerson):
    encoding: List[float]