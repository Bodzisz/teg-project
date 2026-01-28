from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import date

class Requirement(BaseModel):
    skill_name: str = Field(..., description="Nazwa wymaganej technologii")
    min_proficiency: str = Field(..., description="Minimalny poziom umiejętności: Beginner, Intermediate, Advanced, Expert")
    is_mandatory: bool = Field(False, description="Czy wymaganie jest obowiązkowe")

class Assignment(BaseModel):
    programmer_name: str
    programmer_id: int
    assignment_start_date: Optional[str] = None
    assignment_end_date: Optional[str] = None
    allocation_percentage: int = Field(100, description="Procent alokacji w projekcie")

class ProjectData(BaseModel):
    id: str = Field(..., description="Unikalny identyfikator projektu")
    name: str = Field(..., description="Nazwa projektu")
    client: str = Field(..., description="Nazwa klienta")
    description: str = Field(..., description="Opis projektu")
    start_date: date = Field(..., description="Data rozpoczęcia")
    end_date: Optional[date] = Field(None, description="Data zakończenia (opcjonalna)")
    estimated_duration_months: int = Field(..., description="Szacowany czas trwania w miesiącach")
    budget: Optional[Union[str, int]] = Field(None, description="Budżet projektu")
    status: str = Field(..., description="Status projektu: completed, active, planned, on_hold")
    team_size: int = Field(..., description="Rozmiar zespołu")
    requirements: List[Requirement] = Field(..., description="Lista wymagań projektowych")
    assigned_programmers: List[Assignment] = Field(default_factory=list, description="Lista przypisanych programistów")
