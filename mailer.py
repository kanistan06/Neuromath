"""Minimal transactional email service with safe production defaults."""

from __future__ import annotations

import html
import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr

import config


logger = logging.getLogger(__name__)


class EmailDeliveryError(RuntimeError):
    pass


def _smtp_ready() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_FROM_EMAIL)


def _public_auth_link(action: str, raw_token: str) -> str:
    """Build a fragment link so the secret token is not sent in HTTP access logs."""
    return f"{config.PUBLIC_BASE_URL.rstrip('/')}/#{action}={raw_token}"


def send_email(*, to_email: str, subject: str, text_body: str, html_body: str) -> None:
    """Send one transactional message without exposing credentials or tokens in logs."""
    mode = config.EMAIL_DELIVERY_MODE
    if mode == "disabled":
        raise EmailDeliveryError("Transactional email is disabled.")
    if mode == "console":
        # Development-only mode deliberately avoids printing message bodies or links.
        print(f"Email suppressed in console mode: subject={subject}")
        return
    if mode != "smtp":
        raise EmailDeliveryError("Transactional email is not configured.")
    if not _smtp_ready():
        raise EmailDeliveryError("SMTP configuration is incomplete.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr((config.SMTP_FROM_NAME, config.SMTP_FROM_EMAIL))
    message["To"] = to_email
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    try:
        if config.SMTP_USE_SSL:
            smtp = smtplib.SMTP_SSL(
                config.SMTP_HOST,
                config.SMTP_PORT,
                timeout=config.SMTP_TIMEOUT_SECONDS,
                context=ssl.create_default_context(),
            )
        else:
            smtp = smtplib.SMTP(
                config.SMTP_HOST,
                config.SMTP_PORT,
                timeout=config.SMTP_TIMEOUT_SECONDS,
            )
        with smtp:
            if config.SMTP_USE_TLS and not config.SMTP_USE_SSL:
                smtp.starttls(context=ssl.create_default_context())
            if config.SMTP_USERNAME:
                smtp.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
            smtp.send_message(message)
    except Exception as exc:
        logger.error(
            "SMTP delivery failed (host=%s, port=%s, recipient_domain=%s, error=%s)",
            config.SMTP_HOST,
            config.SMTP_PORT,
            to_email.rsplit("@", 1)[-1] if "@" in to_email else "invalid",
            type(exc).__name__,
        )
        raise EmailDeliveryError("Transactional email could not be delivered.") from exc


def send_verification_email(to_email: str, name: str, raw_token: str) -> None:
    link = _public_auth_link("verify", raw_token)
    safe_name = html.escape(name or "Student")
    send_email(
        to_email=to_email,
        subject="Verify your NeuroMath email",
        text_body=(
            f"Hello {name or 'Student'},\n\n"
            f"Verify your NeuroMath email using this link:\n{link}\n\n"
            f"This link expires in {config.EMAIL_VERIFICATION_TTL_HOURS} hours. "
            "If you did not create this account, ignore this email."
        ),
        html_body=(
            f"<p>Hello {safe_name},</p>"
            "<p>Verify your NeuroMath email address to activate your account.</p>"
            f'<p><a href="{html.escape(link, quote=True)}">Verify email address</a></p>'
            f"<p>This link expires in {config.EMAIL_VERIFICATION_TTL_HOURS} hours.</p>"
            "<p>If you did not create this account, ignore this email.</p>"
        ),
    )


def send_password_reset_otp(to_email: str, name: str, otp: str) -> None:
    """Send the short-lived reset OTP without including a password-reset link."""
    safe_name = html.escape(name or "Student")
    safe_otp = html.escape(str(otp))
    send_email(
        to_email=to_email,
        subject="Your NeuroMath password-reset code",
        text_body=(
            f"Hello {name or 'Student'},\n\n"
            f"Your NeuroMath password-reset code is: {otp}\n\n"
            f"This code expires in {config.PASSWORD_RESET_TTL_MINUTES} minutes and can be used once. "
            "If you did not request this change, ignore this email."
        ),
        html_body=(
            f"<p>Hello {safe_name},</p>"
            "<p>We received a request to reset your NeuroMath password.</p>"
            f'<p style="font-size: 24px; font-weight: bold; letter-spacing: 6px">{safe_otp}</p>'
            f"<p>This code expires in {config.PASSWORD_RESET_TTL_MINUTES} minutes and can be used once.</p>"
            "<p>If you did not request this change, ignore this email.</p>"
        ),
    )


def send_password_changed_notice(to_email: str, name: str) -> None:
    safe_name = html.escape(name or "Student")
    send_email(
        to_email=to_email,
        subject="Your NeuroMath password was changed",
        text_body=(
            f"Hello {name or 'Student'},\n\nYour NeuroMath password was changed. "
            "If this was not you, contact the NeuroMath administrator immediately."
        ),
        html_body=(
            f"<p>Hello {safe_name},</p><p>Your NeuroMath password was changed.</p>"
            "<p>If this was not you, contact the NeuroMath administrator immediately.</p>"
        ),
    )
