import asyncio
import datetime
import json
from pathlib import Path

import click

from brainiac.ai import Ai
from brainiac.model import (
    DATETIME_FORMAT,
    AnalyticsMetadata,
    BrainiacConfig,
    ConfigKey,
    InterestMetadata,
    Metadata,
    MetadataFile,
)
from brainiac.utils import (
    convert_to_slug,
    copy_file,
    get_reading_time_in_minutes,
    get_word_count,
    read_file,
    write_file,
)


@click.group()
def cli():
    """Brainiac-2 CLI"""
    pass


async def copy(src: str) -> None:
    config = BrainiacConfig()
    model = Ai(
        api_key=config.get(ConfigKey.OPENAI_API_KEY),
        model=config.get(ConfigKey.OPENAI_MODEL),
    )

    src_path = Path(src)
    if not src_path.exists():
        raise click.BadParameter(f"Source file {src} does not exist.")
    if not src_path.is_file():
        raise click.BadParameter(f"Source path {src} is not a file.")

    output_directory_path = Path(config.get(ConfigKey.OUTPUT_DIRECTORY))
    aggregate_metadata_path = output_directory_path.joinpath(
        config.get(ConfigKey.METADATA_STORAGE_NAME)
    )
    if aggregate_metadata_path.exists():
        aggregate_metadata = MetadataFile.model_validate_json(
            read_file(aggregate_metadata_path)
        )
    else:
        aggregate_metadata = MetadataFile(metadata={})

    article_content = read_file(src_path)

    [generated_meta, interest_meta] = await asyncio.gather(
        model.create_metadata_fields(article_content),
        model.create_interest_metadata_fields(
            agg_metadata=json.dumps(
                list(
                    map(lambda x: x.model_dump(), aggregate_metadata.metadata.values())
                )
            ),
            article=article_content,
        ),
    )

    metadata = Metadata(
        title=generated_meta.title,
        description=generated_meta.description,
        author=config.get(ConfigKey.AUTHOR),
        slug=convert_to_slug(generated_meta.title),
        analytics=AnalyticsMetadata(
            created_at=datetime.datetime.now().strftime(DATETIME_FORMAT),
            length_in_words=get_word_count(article_content),
            reading_time_in_minutes=get_reading_time_in_minutes(article_content),
        ),
        interest=InterestMetadata(
            keywords=generated_meta.keywords,
            genre=generated_meta.genre,
            related_articles=interest_meta.related_articles,
        ),
    )

    aggregate_metadata.push(metadata)
    write_file(file_path=aggregate_metadata_path, content=aggregate_metadata)

    copied_file_path = output_directory_path.joinpath(f"{metadata.slug}.md")
    copy_file(dest=copied_file_path, content=article_content)
    print(metadata.model_dump_json(indent=2))
    ## TODO: test


@click.command()
@click.argument("src", required=1)
def copy_async(src):
    """
    Copy a file from one location to another while adding the metadata to it

    SRC: Location of the file to be copied
    """
    asyncio.run(copy(src))


cli.add_command(copy_async)

if __name__ == "__main__":
    asyncio.run(cli())
