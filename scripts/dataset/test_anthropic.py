import os
import asyncio
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENMODEL_API_KEY") or os.environ.get("API_KEY")
base = os.environ.get("API_BASE_URL", "https://api.openmodel.ai")
if base.endswith("/v1"):
    base = base[:-3]
elif base.endswith("/v1/"):
    base = base[:-4]

print(f"Using Base URL: {base}")
client = AsyncAnthropic(api_key=api_key, base_url=base)

async def main():
    try:
        response = await client.messages.create(
            model="deepseek-v4-flash",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("Success!", response.content)
    except Exception as e:
        print("Error:", repr(e))
        if hasattr(e, 'response'):
            print("Response:", getattr(e.response, "text", str(e.response)))

asyncio.run(main())
