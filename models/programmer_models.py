from pydantic import BaseModel, Field
from typing import List, Optional

class Skill(BaseModel):
    name: str = Field(..., description="Nazwa umiejętności, np. Python, FastAPI")
    proficiency: str = Field(..., description="Poziom umiejętności: Beginner, Intermediate, Advanced, Expert")

class Certification(BaseModel):
    name: str = Field(..., description="Nazwa certyfikatu")

class ProgrammerData(BaseModel):
    id: str = Field(..., description="Unikalny identyfikator programisty (name from graph)")
    name: str = Field(..., description="Imię i nazwisko")
    email: Optional[str] = Field(None, description="Adres e-mail")
    location: Optional[str] = Field(None, description="Lokalizacja")
    skills: List[Skill] = Field(..., description="Lista umiejętności z poziomem")
    projects: List[str] = Field(default_factory=list, description="Lista projektów, w których uczestniczył")
    certifications: List[str] = Field(default_factory=list, description="Lista certyfikatów")
    availability: Optional[float] = Field(100.0, description="Dostępność programisty (procent)")
    years_experience: Optional[int] = Field(0, description="Lata doświadczenia")
