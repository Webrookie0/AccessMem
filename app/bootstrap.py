"""Create a local .env with a random JWT secret, preserving existing settings."""

import os
import secrets
from pathlib import Path


def create_env(root: Path) -> bool:
    target = root / ".env"
    if target.exists():
        return False
    template = (root / ".env.example").read_text(encoding="utf-8")
    template = template.replace("change-me-to-a-long-random-string", secrets.token_urlsafe(48))
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(template)
    return True


def main() -> None:
    created = create_env(Path(__file__).resolve().parent.parent)
    print("Created .env with a random JWT secret." if created else "Existing .env preserved.")


if __name__ == "__main__":
    main()
