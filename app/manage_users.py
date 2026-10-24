"""Local account provisioning without exposing an unauthenticated signup API."""

import argparse
import getpass

from .auth.models import ROLE_CLEARANCE
from .auth.users import create_user, disable_user, list_users


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage AccessMem accounts")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("username")
    create.add_argument("--role", required=True, choices=sorted(ROLE_CLEARANCE))
    create.add_argument("--tenant", required=True)
    disable = commands.add_parser("disable")
    disable.add_argument("username")
    commands.add_parser("list")
    args = parser.parse_args()
    try:
        if args.command == "create":
            password = getpass.getpass("Password (at least 12 characters): ")
            if password != getpass.getpass("Confirm password: "):
                raise ValueError("Passwords do not match")
            user = create_user(args.username, args.role, args.tenant, password)
            print(f"Created {user.username} in tenant {user.tenant_id} with role {user.role}")
        elif args.command == "disable":
            disable_user(args.username)
            print(f"Disabled {args.username}; existing tokens can no longer authenticate")
        else:
            for user in list_users():
                print(
                    f"{user['username']}  {user['role']}  {user['tenant_id']}  "
                    f"{'disabled' if user['disabled'] else 'active'}"
                )
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
