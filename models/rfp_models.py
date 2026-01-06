from pydantic import BaseModel, Field, validator
from typing import List
from datetime import date

class RFPRequirement(BaseModel):
    skill_name: str = Field(..., description="Nazwa wymaganej technologii")
    min_proficiency: str = Field(..., description="Minimalny poziom umiejętności")
    is_mandatory: bool = Field(False, description="Czy wymaganie jest obowiązkowe")
    preferred_certifications: List[str] = Field(default_factory=list, description="Preferowane certyfikaty")

    @validator("preferred_certifications", pre=True)
    def ensure_list(cls, v):
        """Ensure preferred_certifications is always a list."""
        if isinstance(v, str):
            return [v]
        if v is None:
            return []
        return v

class RFPData(BaseModel):
    id: str = Field(..., description="Unikalny identyfikator RFP")
    title: str = Field(..., description="Tytuł zapytania ofertowego")
    client: str = Field(..., description="Nazwa klienta")
    description: str = Field(..., description="Opis zapytania")
    project_type: str = Field(..., description="Typ projektu")
    duration_months: int = Field(..., description="Szacowany czas trwania w miesiącach")
    team_size: int = Field(..., description="Rozmiar zespołu")
    budget_range: str = Field(..., description="Zakres budżetu")
    start_date: date = Field(..., description="Data rozpoczęcia")
    requirements: List[RFPRequirement] = Field(..., description="Lista wymagań")
    location: str = Field(..., description="Lokalizacja projektu")
    remote_allowed: bool = Field(..., description="Czy praca zdalna jest dozwolona")
