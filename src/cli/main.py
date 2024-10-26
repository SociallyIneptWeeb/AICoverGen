"""
Module which defines the command-line interface for the Ultimate RVC
project.
"""

import typer

from cli.generate_song_cover import app as song_cover_app

app = typer.Typer(
    name="urvc-cli",
    no_args_is_help=True,
    help="CLI for the Ultimate RVC project",
    rich_markup_mode="markdown",
)

app.add_typer(song_cover_app)