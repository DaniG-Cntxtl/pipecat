import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from pipecat.frames.frames import AudioRawFrame, TranscriptionFrame
from pipecat.services.elevenlabs.sts import ElevenLabsSTSService

class TestElevenLabsSTSService(unittest.IsolatedAsyncioTestCase):

    @patch('elevenlabs.client.ElevenLabs')
    def setUp(self, mock_elevenlabs):
        self.service = ElevenLabsSTSService(api_key="test_api_key", agent_id="test_agent_id")
        self.service._task_manager = MagicMock()
        self.service.push_frame = AsyncMock()
        self.mock_elevenlabs = mock_elevenlabs

    @patch('elevenlabs.conversational_ai.conversation.Conversation')
    def test_run_conversation(self, mock_conversation):
        # Mock the conversation object and its methods
        mock_convo_instance = MagicMock()
        mock_convo_instance.start_session = MagicMock()
        mock_convo_instance.wait_for_session_end = MagicMock()
        mock_conversation.return_value = mock_convo_instance

        # Run the conversation
        asyncio.run(self.service.run_conversation())

        # Check that the conversation was started
        mock_conversation.assert_called_with(
            self.service._client,
            self.service._agent_id,
            audio_interface=self.service._audio_interface,
            callback_agent_response=self.service._on_agent_response,
            callback_agent_response_correction=self.service._on_agent_response_correction,
            callback_user_transcript=self.service._on_user_transcript,
            requires_auth=False
        )
        mock_convo_instance.start_session.assert_called_once()

    async def test_process_frame(self):
        self.service._audio_interface.input_callback = MagicMock()
        audio_frame = AudioRawFrame(b"audio data", 16000, 1)
        await self.service.process_frame(audio_frame, None)
        self.service._audio_interface.input_callback.assert_called_with(b"audio data")
        self.service.push_frame.assert_called_with(audio_frame)

    def test_on_user_transcript(self):
        self.service._on_user_transcript("hello world")
        self.service.push_frame.assert_called_once()

    def test_custom_audio_interface_output(self):
        audio_data = b"test audio"
        self.service._audio_interface.output(audio_data)
        self.service._task_manager.create_task.assert_called_once()

if __name__ == '__main__':
    unittest.main()