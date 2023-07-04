import typer


def mars_request_cli(config: str = ""):
    """
    MARS requests demo cli
    """

    if not config:
        print("Configuration required. User --config")
        return

    print("Configuration = ", config)
    print("<START SUPER LONG PROCESS>")
    print("...")


def main():
    typer.run(mars_request_cli)


if __name__ == "__main__":
    main()
