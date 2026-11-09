import os
import socket
import sys


def main():
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    with socket.create_connection((host, port), timeout=2):
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError:
        sys.exit(1)
