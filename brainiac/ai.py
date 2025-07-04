import json

import openai
from openai.types.responses import ResponseFormatTextJSONSchemaConfigParam

from brainiac.model import InterestMetadataGeneratedResponse, MetadataGeneratedResponse


class Ai:
    def __init__(self, api_key: str, model: str):
        client = openai.AsyncClient(api_key=api_key)
        self._client = client
        self._model = model

    async def create_metadata_fields(self, article: str) -> MetadataGeneratedResponse:
        schema = MetadataGeneratedResponse.model_json_schema(mode="serialization")
        # Add the required additionalProperties: false
        schema["additionalProperties"] = False

        text_config: ResponseFormatTextJSONSchemaConfigParam = {
            "name": MetadataGeneratedResponse.__name__,
            "schema": schema,
            "type": "json_schema",
            "description": "Metadata of the article represented in json",
            "strict": True,
        }
        instructions = """You are an expert editorial assistant. When given article content, you must produce four outputs in JSON format:
        1. "title":
           – A concise, engaging headline (≤ 60 characters) that captures the essence of the topic and the flavor of its genre.

        2. "description":
           – A 1–2 sentence summary (≤ 200 characters) that expands on the core idea and clearly reflects the tone and perspective of the chosen genre.

        3. "keywords":
           – An array of 5–8 relevant, high-impact keywords or short phrases that will help optimize discoverability.

        4. "genre":
           – One of exactly these three strings: OPINION, TECHNOLOGY, LIFESTYLE.
           – Choose the genre that best fits the topic and ensure both "title" and "description" are written in that genre’s style.

        Requirements:
        - Always output valid JSON with exactly these four keys.
        - Do not include any additional keys or commentary.
        - Ensure the tone of the title and description matches the genre:
          • OPINION: bold, personal, provocative
          • TECHNOLOGY: precise, informative, forward-looking
          • LIFESTYLE: warm, relatable, experiential
        """

        generated = await self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=article,
            text={"format": text_config},
        )

        return MetadataGeneratedResponse.model_validate_json(generated.output_text)

    async def create_interest_metadata_fields(
        self, agg_metadata: str, article: str
    ) -> InterestMetadataGeneratedResponse:
        schema = InterestMetadataGeneratedResponse.model_json_schema(
            mode="serialization"
        )
        # Add the required additionalProperties: false
        schema["additionalProperties"] = False

        text_config: ResponseFormatTextJSONSchemaConfigParam = {
            "name": InterestMetadataGeneratedResponse.__name__,
            "schema": schema,
            "type": "json_schema",
            "description": "Interest metadata on the article represented in json",
            "strict": True,
        }

        instructions = """
        You are an editorial assistant. You will receive a JSON object with two fields:
          • “articles”: an array of article metadata objects
          • “target”: a single article metadata object

        Each metadata object has:
          – title (string)
          – description (string)
          – keywords (array of strings)
          – slug (string)
          – genre (one of OPINION, TECHNOLOGY, LIFESTYLE)

        Your job is to compute semantic similarity between “target” and each object in “articles” (using title, description, keywords, and genre), then return the two most related articles.
        Output **only** a JSON array of their slugs, ordered from most to next-most related. No extra text.

        USER (example payload):
        ```json
        {
          "articles": [
            {
              "title": "How AI is Transforming Fintech",
              "description": "An overview of AI applications in financial services...",
              "keywords": ["AI", "fintech", "automation"],
              "slug": "ai-fintech",
              "genre": "TECHNOLOGY"
            },
            {
              "title": "My Morning Routine",
              "description": "A personal look into my healthy habits...",
              "keywords": ["wellness", "morning", "routine"],
              "slug": "morning-routine",
              "genre": "LIFESTYLE"
            },
            {
              "title": "Opinion: The Future of Work",
              "description": "Why remote collaboration will redefine careers...",
              "keywords": ["remote", "work", "opinion"],
              "slug": "future-of-work",
              "genre": "OPINION"
            }
          ],
          "target":"**The Rise of AuroraMind: AI’s Next Frontier**
          In a quiet lab on the outskirts of Copenhagen, researchers have unveiled AuroraMind, an AI system capable of composing symphonies in the style of Beethoven and analyzing weather patterns to predict local microclimates with uncanny precision. Unlike its predecessors, AuroraMind blends a transformer-based language model with a novel temporal-spatial neural network, enabling it to learn both linguistic nuance and real-world dynamics simultaneously.
            "
        }
        """
        generated = await self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=json.dumps({"articles": agg_metadata, "target": article}),
            text={"format": text_config},
        )

        return InterestMetadataGeneratedResponse.model_validate_json(
            generated.output_text
        )
