"""Entry point:  python -m lumeide"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from .launcher import open_window, print_banner
from .server import serve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lumeide", description="Lume Studio - the IDE for Lume")
    parser.add_argument("--port", type=int, default=0, help="port to listen on (0 = pick a free one)")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument("--workspace", default=os.getcwd(), help="folder with your Lume projects")
    parser.add_argument("--no-window", action="store_true", help="do not open a browser window")
    parser.add_argument("--open", action="store_true", help="force opening the browser")
    args = parser.parse_args(argv)

    workspace = os.path.abspath(args.workspace)
    os.makedirs(workspace, exist_ok=True)
    server = serve(workspace=workspace, host=args.host, port=args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"

    if args.no_window:
        print_banner(url, workspace)
    else:
        print_banner(url, workspace)
        print("  " + open_window(url) + "\n")

    stop = {"now": False}

    def handler(signum, frame):
        stop["now"] = True

    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, AttributeError):
        pass

    try:
        while not stop["now"]:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    print("  studio stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
