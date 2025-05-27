from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class WhisperSegment(BaseModel):
    """Represents a segment in the transcription"""
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: List[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.1
    speaker: Optional[str] = None  # For speaker diarization

class TranscriptionResponse(BaseModel):
    """Represents the response format for transcription"""
    text: str
    segments: Optional[List[WhisperSegment]] = None
    language: Optional[str] = None
    task: str = "transcribe"
    duration: Optional[float] = None
    model: Optional[str] = None
    
    class Config:
        schema_extra = {"example": {"text": "Hello world", "segments": []}}
    
    def dict(self, **kwargs):
        """Custom dict method to handle response format"""
        # If we don't need segments, remove them
        result = super().dict(**kwargs)
        if not self.segments:
            result.pop("segments", None)
        return result

class ModelInfo(BaseModel):
    """Information about a model available in the API"""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: str
    parent: Optional[str] = None

class ModelList(BaseModel):
    """List of available models"""
    object: str = "list"
    data: List[ModelInfo]