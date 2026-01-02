from pydantic import BaseModel
from typing import List, Optional

class SkillRequirement(BaseModel):
    name: str
    experience_level: str

class RFPData(BaseModel):
    title: str
    description: str
    skills: List[SkillRequirement]
    budget: Optional[str]
    deadline: Optional[str]
    team_size: Optional[int]
