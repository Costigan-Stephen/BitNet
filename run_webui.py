import sys


def main() -> None:
    try:
        import uvicorn
    except Exception:
        print(
            "Missing web UI dependencies. Install with:\n"
            "python -m pip install -r webui/requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    uvicorn.run("webui.app:app", host="127.0.0.1", port=7860, reload=False, log_level="info")


if __name__ == "__main__":
    main()

