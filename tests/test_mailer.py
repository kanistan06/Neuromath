import mailer


class FakeSMTP:
    sent_messages = []

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def starttls(self, context):
        self.tls_context = context

    def login(self, username, password):
        self.credentials = (username, password)

    def send_message(self, message):
        self.sent_messages.append(message)


def test_smtp_verification_link_and_password_reset_otp(monkeypatch):
    FakeSMTP.sent_messages = []
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(mailer.config, "EMAIL_DELIVERY_MODE", "smtp")
    monkeypatch.setattr(mailer.config, "PUBLIC_BASE_URL", "https://math.example.com/")
    monkeypatch.setattr(mailer.config, "SMTP_HOST", "smtp.resend.com")
    monkeypatch.setattr(mailer.config, "SMTP_PORT", 587)
    monkeypatch.setattr(mailer.config, "SMTP_USERNAME", "resend")
    monkeypatch.setattr(mailer.config, "SMTP_PASSWORD", "re_test_key")
    monkeypatch.setattr(mailer.config, "SMTP_FROM_EMAIL", "learn@neuromath.io")
    monkeypatch.setattr(mailer.config, "SMTP_FROM_NAME", "NeuroMath")
    monkeypatch.setattr(mailer.config, "SMTP_USE_TLS", True)
    monkeypatch.setattr(mailer.config, "SMTP_USE_SSL", False)

    mailer.send_verification_email("student@example.com", "Student", "verify-token")
    mailer.send_password_reset_otp("student@example.com", "Student", "482019")

    assert len(FakeSMTP.sent_messages) == 2
    verification_body = FakeSMTP.sent_messages[0].get_body(preferencelist=("plain",)).get_content()
    reset_body = FakeSMTP.sent_messages[1].get_body(preferencelist=("plain",)).get_content()
    assert "https://math.example.com/#verify=verify-token" in verification_body
    assert "482019" in reset_body
    assert "#reset=" not in reset_body
    assert FakeSMTP.sent_messages[1]["From"] == "NeuroMath <learn@neuromath.io>"
