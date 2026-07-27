from typing import List

from pydantic import BaseModel, Field


class SupportedClaim(BaseModel):
    claim: str = Field(min_length=1)
    evidence_chunk_ids: List[str] = Field(min_length=1)


class ReasoningOutput(BaseModel):
    claims: List[SupportedClaim] = Field(default_factory=list)


class CriticDecision(BaseModel):
    claim: str
    supported: bool
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    feedback: str


class CriticOutput(BaseModel):
    decisions: List[CriticDecision] = Field(default_factory=list)