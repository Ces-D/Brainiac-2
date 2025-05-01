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
            "description": "Metadata on the article represented in json",
            "strict": True,
        }
        generated = await self._client.responses.create(
            model=self._model,
            instructions="You are a highly skilled editorial analyst. Your task is to extract structured information from articles while maintaining a neutral, accurate tone. Your responses must be concise, consistent, and suitable for indexing and content classification.",
            input=f"Analyze the following article and extract the following:\n\n1. **Keywords**: Provide 5 to 10 relevant keywords that represent the core themes, topics, or emotional tones of the article. Avoid overly generic words (e.g., 'article', 'topic') and do not repeat word stems (e.g., 'technology' and 'technological').\n\n2. **Genre**: Label the genre or type of article (e.g., opinion, feature, editorial, satire, instructional). Be specific.\n\n3. **Title**: Create a clear and compelling title that reflects both the tone and content. Avoid vague or generic phrasing. Do not use colons unless essential.\n\n4. **Summary**: Write a 2–5 sentence summary that:\n   - Captures the core idea or message\n   - Reflects the emotional tone (e.g., hopeful, critical, nostalgic)\n   - Describes the likely intention of the author (e.g., to inform, to persuade, to entertain)\n\n**Guardrails:**\n- Do NOT include your own opinions.\n- Avoid exaggeration or clickbait-style wording.\n- Keep language professional and neutral.\n- Assume the audience is informed but not expert.\n\nThe article is delimited by triple quotes:\n\n\"\"\"\n{article}\n\"\"\"",
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
        generated = await self._client.responses.create(
            model=self._model,
            instructions="You are an assistant that processes article metadata and generates structured relationship mappings between articles based on shared interests and themes.You will receive a json collection of articles, each with its own slug, title, description, and interest metadata. Your task is to match the article to a list of related articles. You will consider the following factors:\n\n1. **Keywords**: Articles with similar keywords are likely to be related.\n2. **Genre**: Articles of the same genre may share themes or topics.\n3. **Title**: Articles with similar titles may be related.\n4. **Description**: Articles with similar descriptions may be related.",
            input=f'YOUR OUTPUT MUST FOLLOW THESE STRICT RULES:\nReturn only a single valid JSON object that contains a list of related articles.\nEach item of the list  must be a slug from the provided list of articles.\nThere must be a list of objects, each the slug of a related article provided\n Never recommend the same article as a related article (i.e., the slug of the article must not appear in its own related_articles list).\n Do not include any text, explanation, or markdown — only return the JSON object.\nIt is possible to not have any related articles.\n\nThe aggregate collection fo articles is delimited by triple quotes:\n\n"""\n{agg_metadata}\n"""\n\nThe article is delimited by triple quotes:\n\n"""\n{article}\n"""',
            text={"format": text_config},
        )

        return InterestMetadataGeneratedResponse.model_validate_json(
            generated.output_text
        )
