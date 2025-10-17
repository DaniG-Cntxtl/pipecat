import asyncio
import logging
import os
import time

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, AudioInterface
from pipecat.frames.frames import AudioRawFrame, Frame, StartFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger("pipecat")

class ElevenLabsSTSService(FrameProcessor):
    """A speech-to-speech service that uses ElevenLabs to generate responses.

    This service uses the ElevenLabs Conversational AI API to stream audio
    to an agent and receive audio responses back.

    Args:
        api_key (str): The ElevenLabs API key.
        agent_id (str): The ID of the agent to converse with.
        requires_auth (bool, optional): Whether the agent requires
            authentication. Defaults to False.
    """
    def __init__(self, api_key: str, agent_id: str, **kwargs):
        super().__init__(**kwargs)
        self.downstream_audio_format = "pcm_16000"
        self._client = ElevenLabs(api_key=api_key)
        self._agent_id = agent_id
        self.requires_auth = kwargs.get("requires_auth", False)
        self._conversation = None
        self._audio_interface = self.CustomAudioInterface(self)
        self._conversation_task = None

    async def run_conversation(self):
        self._conversation = Conversation(
            self._client,
            self._agent_id,
            audio_interface=self._audio_interface,
            callback_agent_response=self._on_agent_response,
            callback_agent_response_correction=self._on_agent_response_correction,
            callback_user_transcript=self._on_user_transcript,
            requires_auth=self.requires_auth,
        )
        self._conversation.start_session()
        self._conversation.wait_for_session_end()

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._conversation_task = self.create_task(self.run_conversation(), "elevenlabs_conversation")
            await self.push_frame(frame)
        elif isinstance(frame, AudioRawFrame):
            if self._audio_interface.input_callback:
                self._audio_interface.input_callback(frame.audio)
        else:
            await self.push_frame(frame)

    def _on_agent_response(self, response: str):
        logger.info(f"Agent response: {response}")

    def _on_agent_response_correction(self, original: str, corrected: str):
        logger.info(f"Agent response corrected from '{original}' to '{corrected}'")

    def _on_user_transcript(self, transcript: str):
        logger.info(f"User transcript: {transcript}")
        self.create_task(self.push_frame(TranscriptionFrame(transcript, "user", int(time.time() * 1000))), "push_transcription")

    class CustomAudioInterface(AudioInterface):
        def __init__(self, service: "ElevenLabsSTSService"):
            self._service = service
            self.input_callback = None

        def start(self, input_callback):
            self.input_callback = input_callback

        def stop(self):
            pass

        def output(self, audio: bytes):
            frame = AudioRawFrame(audio, self._service.downstream_audio_format, 1)
            self._service.create_task(self._service.push_frame(frame), "push_audio")

        def interrupt(self):
            pass