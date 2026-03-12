import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.services.email_service import send_deletion_confirmation_email


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SMTP configuration for Gestura.")
    parser.add_argument(
        "--to",
        dest="recipient",
        default="",
        help="Recipient email to send a test confirmation message.",
    )
    args = parser.parse_args()

    if not settings.smtp_host:
        print("SMTP_HOST is not configured.")
        return 1

    if not args.recipient:
        print("SMTP settings are present. Provide --to to send a test email.")
        return 0

    test_url = f"{settings.delete_confirm_base_url}?token=smtp-verification-token"

    try:
        send_deletion_confirmation_email(args.recipient, test_url)
    except Exception as exc:
        print(f"SMTP verification failed: {exc}")
        return 1

    print(f"SMTP verification email sent to {args.recipient}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
