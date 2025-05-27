# Speaker diarization module for Parakeet
# This module integrates pyannote.audio for speaker identification

from typing import Dict, List, Optional, Tuple, Union
import os
import logging
import tempfile
import numpy as np
import torch
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class SpeakerSegment(BaseModel):
    """A segment of speech from a specific speaker"""
    start: float
    end: float
    speaker: str

class DiarizationResult(BaseModel):
    """Result of speaker diarization"""
    segments: List[SpeakerSegment]
    num_speakers: int

class Diarizer:
    """Speaker diarization using pyannote.audio"""

    def __init__(self, access_token: Optional[str] = None):
        self.pipeline = None
        self.access_token = access_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize()

    def _initialize(self):
        """Initialize the diarization pipeline"""
        try:
            from pyannote.audio import Pipeline

            if not self.access_token:
                logger.warning("No access token provided. Using HUGGINGFACE_ACCESS_TOKEN environment variable.")
                self.access_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

            if not self.access_token:
                logger.error("No access token available. Diarization will not work.")
                return

            # Initialize the pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.access_token
            )

            # Move to GPU if available
            self.pipeline.to(torch.device(self.device))
            logger.info(f"Diarization pipeline initialized on {self.device}")

        except ImportError:
            logger.error("Failed to import pyannote.audio. Please install it with 'pip install pyannote.audio'")
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {str(e)}")

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file

        Args:
            audio_path: Path to the audio file
            num_speakers: Optional number of speakers (if known)

        Returns:
            DiarizationResult with speaker segments
        """
        if self.pipeline is None:
            logger.error("Diarization pipeline not initialized")
            return DiarizationResult(segments=[], num_speakers=0)

        try:
            # Run the diarization pipeline
            diarization = self.pipeline(
                audio_path,
                num_speakers=num_speakers
            )

            # Convert to our format
            segments = []
            speakers = set()

            # Process the diarization result
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Convert speaker label to consistent format
                # This handles different formats from pyannote.audio versions
                if isinstance(speaker, str) and not speaker.startswith("SPEAKER_"):
                    speaker_id = f"SPEAKER_{speaker}"
                else:
                    speaker_id = speaker

                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=f"speaker_{speaker_id}"
                ))
                speakers.add(speaker_id)

            # Sort segments by start time
            segments.sort(key=lambda x: x.start)

            return DiarizationResult(
                segments=segments,
                num_speakers=len(speakers)
            )

        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            return DiarizationResult(segments=[], num_speakers=0)

    def merge_with_transcription(self,
                                diarization: DiarizationResult,
                                transcription_segments: list) -> list:
        """
        Merge diarization results with transcription segments

        Args:
            diarization: Speaker diarization result
            transcription_segments: List of transcription segments with start/end times

        Returns:
            Merged list of segments with speaker information
        """
        # If no diarization results, return original transcription
        if not diarization.segments:
            return transcription_segments

        # For each transcription segment, find the dominant speaker
        for segment in transcription_segments:
            # Get segment time bounds
            start = segment.start
            end = segment.end

            # Find overlapping diarization segments
            overlapping = []
            for spk_segment in diarization.segments:
                # Calculate overlap
                overlap_start = max(start, spk_segment.start)
                overlap_end = min(end, spk_segment.end)

                if overlap_end > overlap_start:
                    # There is an overlap
                    duration = overlap_end - overlap_start
                    overlapping.append((spk_segment.speaker, duration))

            # Assign the speaker with most overlap
            if overlapping:
                # Sort by duration (descending)
                overlapping.sort(key=lambda x: x[1], reverse=True)
                # Assign the dominant speaker
                setattr(segment, "speaker", overlapping[0][0])
            else:
                # No overlap found, assign unknown
                setattr(segment, "speaker", "unknown")

        return transcription_segments
