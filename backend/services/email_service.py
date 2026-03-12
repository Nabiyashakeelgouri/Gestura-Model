import smtplib
from email.message import EmailMessage

from backend.config import settings


def send_deletion_confirmation_email(recipient_email: str, confirmation_url: str) -> None:
    if not settings.smtp_host:
        raise RuntimeError("SMTP_HOST is not configured.")

    message = EmailMessage()
    message["Subject"] = "Confirm your Gestura account deletion"
    message["From"] = settings.smtp_user or "no-reply@gestura.local"
    message["To"] = recipient_email
    message.set_content(
        "You requested to delete your Gestura account.\n"
        f"Click this link to confirm deletion:\n{confirmation_url}\n\n"
        "If you did not request this, ignore this email."
    )

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
        if settings.smtp_use_tls:
            smtp.starttls()

        if settings.smtp_user and settings.smtp_pass:
            smtp.login(settings.smtp_user, settings.smtp_pass)

        smtp.send_message(message)
