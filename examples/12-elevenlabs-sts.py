import asyncio
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.services.elevenlabs.sts import ElevenLabsSTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from loguru import logger

from dotenv import load_dotenv
load_dotenv()

async def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID")
    room_url = os.environ.get("DAILY_ROOM_URL")

    if not api_key or not agent_id or not room_url:
        logger.error("Missing environment variables")
        return

    transport = DailyTransport(
        room_url,
        None,
        "Respond bot",
        DailyParams(
            audio_out_enabled=True,
            transcription_enabled=True,
        ),
    )

    sts = ElevenLabsSTSService(
        api_key=api_key,
        agent_id=agent_id,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
    ]

    pipeline = Pipeline(
        [
            transport.input(),
            sts,
            LLMUserResponseAggregator(messages),
            transport.output(),
        ]
    )

    runner = PipelineRunner()

    await runner.run(
        pipeline,
        PipelineParams(
            allow_interruptions=True,
            enable_profiling=True,
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())