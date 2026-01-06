from pydantic import BaseModel, Field
from typing import List, Optional

class Skill(BaseModel):
    name: str = Field(..., description="Nazwa umiejętności, np. Python, FastAPI")
    proficiency: str = Field(..., description="Poziom umiejętności: Beginner, Intermediate, Advanced, Expert")

class Certification(BaseModel):
    name: str = Field(..., description="Nazwa certyfikatu")

class ProgrammerData(BaseModel):
    id: int = Field(..., description="Unikalny identyfikator programisty")
    name: str = Field(..., description="Imię i nazwisko")
    email: str = Field(..., description="Adres e-mail")
    location: str = Field(..., description="Lokalizacja")
    skills: List[Skill] = Field(..., description="Lista umiejętności z poziomem")
    projects: List[str] = Field(..., description="Lista projektów, w których uczestniczył")
    certifications: List[str] = Field(default_factory=list, description="Lista certyfikatów")
