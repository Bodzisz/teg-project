graph TD
    %% Węzły główne
    RFP["RFP ✅\n{id, title, description, requirements, budget, deadline}"]
    Project["Project ✅\n{id, title, description, start_date, end_date, budget}"]
    Person["Person ✅\n{id, name, location, email, phone, years_experience}"]
    Skill["Skill ✅\n{id, category, subcategory}"]
    Company["Company ✅\n{id, name, industry, size, location}"]
    Certification["Certification ✅\n{id, name, provider, date_earned, expiry_date}"]
    University["University ✅\n{id, name, location, ranking}"]

    %% Relacje RFP
    RFP -->|NEEDS {required_count, experience_level}| Skill
    RFP -->|RELATED_TO {status: 'won', source: 'rfp_id'}| Project

    %% Relacje Project
    Project -->|REQUIRES {minimum_level, preferred_level}| Skill

    %% Relacje Person
    Person -->|HAS_SKILL {proficiency: 1-5, years_experience}| Skill
    Person -->|WORKED_AT {role, start_date, end_date}| Company
    Person -->|WORKED_ON {role, contribution, start_date, end_date}| Project
    Person -->|EARNED {date, score}| Certification
    Person -->|STUDIED_AT {degree, graduation_year, gpa}| University
    Person -->|ASSIGNED_TO {allocation_percentage, start_date, end_date}| Project
    Person -->|MATCHED_TO {score, matched_skills, mandatory_met}| RFP

    %% Klasy indeksów
    classDef indexed fill:#d1e7dd,stroke:#0f5132,stroke-width:2px;
    class RFP,Project,Person,Skill,Company,Certification,University indexed;
