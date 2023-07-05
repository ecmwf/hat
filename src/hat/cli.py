import click
import typer


def prettyprint(
    text,
    color="white",
    bold=False,
    background=None,
    first_line_empty=False,
    last_line_empty=False,
):
    """Pretty print text using click formating"""
    if first_line_empty:
        print()
    typer.echo(click.style(text, fg=color, bg=background, bold=bold))
    if last_line_empty:
        print()


def warning(text, color="yellow", bold=False):
    """A text warning"""
    prettyprint(text, color=color, bold=bold)


def title(text, **kwargs):
    """A pretty title"""
    print("")
    prettyprint(text, **kwargs)
    print("-" * len(text))
