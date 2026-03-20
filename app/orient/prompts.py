"""Default prompts used by the agent."""

import xml.etree.ElementTree as ET

from pydantic import BaseModel

from app.utils import logger


class ProblemStatementInstructions(BaseModel):
    """DTO for the Problem Statement Map Instructions."""

    executive_summary: str
    background: str
    users_and_stakeholders: str
    problem_and_need: str
    evidence: str
    constraints: str
    success_criteria_and_metrics: str
    assumptions: str
    principles_and_values: str
    risks_and_gaps: str
    open_questions: str
    next_steps: str


logger.info("# Loading Prompts...")

instructions = ProblemStatementInstructions.model_validate(
    {
        child.tag: ET.tostring(child)
        for child in ET.parse("prompts/ps_instructions.xml").getroot()
    }
)
with open("prompts/qa_review.md") as f:
    REVIEWER_PROMPT = f.read()

with open("prompts/ps_synthesis.md") as f:
    SYNTHESIS_PROMPT = f.read()

logger.info("# Prompts loaded.")


SYSTEM_PROMPT = """<role>
You are a helpful AI assistant.
</role>
<objective>
Take the <user_summary /> provided by the user and create a <problem_statement /> document according to the instructions:
<instructions>
ONLY use the tools provided. 
<document_format>
Format the <problem_statement /> as a well-formed markdown string that could be written to a '.md' file.
</document_format>
</instructions>
</objective>"""
