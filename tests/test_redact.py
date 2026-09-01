"""Tests for mempalace.redact — conservative ingest-time secret redaction.

Every kind gets two tests: a positive test with a REALISTIC-SHAPED fake
secret (never a real credential) asserting it is replaced and does not
survive in the output, and a negative test proving ordinary prose near it
(or entirely on its own) comes back byte-identical — the "never mangle
prose" requirement.
"""

from __future__ import annotations

from mempalace.redact import redact_secrets


# ── empty / falsy input ─────────────────────────────────────────────────


def test_redact_secrets_empty_string_returns_unchanged():
    redacted, counts = redact_secrets("")
    assert redacted == ""
    assert counts == {}


def test_redact_secrets_none_returns_unchanged():
    redacted, counts = redact_secrets(None)
    assert redacted is None
    assert counts == {}


# ── negative: plain prose is never touched ──────────────────────────────


def test_redact_secrets_plain_prose_is_byte_identical():
    text = (
        "We decided to move the deploy to Vultr Singapore and keep using "
        "Caddy in front of it. Nothing here looks like a credential at all."
    )
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


def test_redact_secrets_code_without_secrets_is_byte_identical():
    text = "def load_key():\n    key = os.environ.get('API_KEY')\n    return key\n"
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


# ── anthropic_key ────────────────────────────────────────────────────────


def test_redact_anthropic_key():
    fake_key = "sk-ant-api03-AbCdEfGh12345678901234567890ijklMNOPqrstUVWXyz9876"
    text = f"Here is my key: {fake_key} -- please don't share it."
    redacted, counts = redact_secrets(text)
    assert fake_key not in redacted
    assert "[redacted:anthropic_key]" in redacted
    assert counts == {"anthropic_key": 1}
    assert redacted == "Here is my key: [redacted:anthropic_key] -- please don't share it."


def test_redact_anthropic_key_prose_survives_around_it():
    fake_key = "sk-ant-api03-AbCdEfGh12345678901234567890ijklMNOPqrstUVWXyz9876"
    prefix = "Before the key. "
    suffix = " After the key."
    text = f"{prefix}{fake_key}{suffix}"
    redacted, _ = redact_secrets(text)
    assert redacted.startswith(prefix)
    assert redacted.endswith(suffix)


# ── openai_key ────────────────────────────────────────────────────────────


def test_redact_openai_project_key():
    fake_key = "sk-proj-AbCdEfGh12345678901234567890ijklMNOPqrstUVWXyz-9876ABCD"
    text = f"OPENAI_API_KEY={fake_key}"
    redacted, counts = redact_secrets(text)
    assert fake_key not in redacted
    assert counts == {"openai_key": 1}
    assert redacted == "OPENAI_API_KEY=[redacted:openai_key]"


def test_redact_openai_project_key_negative_prose():
    text = "sk-proj is the prefix OpenAI uses for project-scoped API keys."
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


# ── aws_access_key_id ──────────────────────────────────────────────────────


def test_redact_aws_access_key_id():
    # AKIAIOSFODNN7EXAMPLE is AWS's own published placeholder key ID from
    # their documentation, not a real credential.
    fake_key = "AKIAIOSFODNN7EXAMPLE"
    text = f"aws_access_key_id = {fake_key}"
    redacted, counts = redact_secrets(text)
    assert fake_key not in redacted
    assert counts == {"aws_access_key_id": 1}
    assert redacted == "aws_access_key_id = [redacted:aws_access_key_id]"


def test_redact_aws_access_key_id_negative_prose():
    text = "AKIA is the standard prefix for AWS access key IDs, always 20 chars long."
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


# ── github_pat (classic) ───────────────────────────────────────────────────


def test_redact_github_pat_classic():
    fake_token = "ghp_" + "aB1cD2eF3gH4iJ5kL6mN7oP8qR9sT0uV1wX2y3Z4"[:36]
    assert len(fake_token) == 40  # ghp_ (4) + 36
    text = f"token: {fake_token}"
    redacted, counts = redact_secrets(text)
    assert fake_token not in redacted
    assert counts == {"github_pat": 1}
    assert redacted == "token: [redacted:github_pat]"


def test_redact_github_pat_classic_negative_prose():
    text = "GitHub classic tokens start with the literal prefix ghp_ followed by 36 characters."
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


# ── github_fine_grained_pat ─────────────────────────────────────────────────


def test_redact_github_fine_grained_pat():
    fake_token = "github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJ"
    text = f"export GH_TOKEN={fake_token}"
    redacted, counts = redact_secrets(text)
    assert fake_token not in redacted
    assert counts == {"github_fine_grained_pat": 1}
    assert redacted == "export GH_TOKEN=[redacted:github_fine_grained_pat]"


def test_redact_github_fine_grained_pat_negative_prose():
    text = "Fine-grained tokens use the github_pat_ prefix instead of ghp_."
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


# ── slack_token ─────────────────────────────────────────────────────────────


def test_redact_slack_bot_token():
    fake_token = "xoxb-1234567890123-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"
    text = f"SLACK_BOT_TOKEN={fake_token}"
    redacted, counts = redact_secrets(text)
    assert fake_token not in redacted
    assert counts == {"slack_token": 1}
    assert redacted == "SLACK_BOT_TOKEN=[redacted:slack_token]"


def test_redact_slack_token_negative_prose():
    text = "Slack bot tokens start with xoxb-, app tokens with xoxa-."
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


# ── google_api_key ────────────────────────────────────────────────────────


def test_redact_google_api_key():
    fake_key = "AIzaSyD-9bT3xYzABCDEFghijklmnopQRSTUv12"
    assert len(fake_key) == 39  # AIza (4) + 35
    text = f"GOOGLE_MAPS_API_KEY={fake_key}"
    redacted, counts = redact_secrets(text)
    assert fake_key not in redacted
    assert counts == {"google_api_key": 1}
    assert redacted == "GOOGLE_MAPS_API_KEY=[redacted:google_api_key]"


def test_redact_google_api_key_negative_prose():
    text = "Google API keys are recognizable by their AIza prefix."
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


# ── jwt ───────────────────────────────────────────────────────────────────


def test_redact_jwt():
    # The canonical jwt.io example JWT — a widely published sample token,
    # not tied to any real account or secret.
    fake_jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    text = f"Authorization: Bearer {fake_jwt}"
    redacted, counts = redact_secrets(text)
    assert fake_jwt not in redacted
    assert counts == {"jwt": 1}
    assert redacted == "Authorization: Bearer [redacted:jwt]"


def test_redact_jwt_negative_prose():
    text = "A JWT has three dot-separated parts: header, payload, and signature."
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


# ── private_key (PEM block) ─────────────────────────────────────────────────


_FAKE_PEM_BLOCK = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEr6+eS984zaHFEr6+eS984zaHFEXA\n"
    "MPLEbase64linesthatarenotarealkeyatalljustshapedlikeoneEXAMPLEEX\n"
    "-----END RSA PRIVATE KEY-----"
)


def test_redact_pem_private_key_block():
    text = f"Here's the key:\n{_FAKE_PEM_BLOCK}\nThat's all of it."
    redacted, counts = redact_secrets(text)
    assert "-----BEGIN" not in redacted
    assert "MIIBOgIBAAJB" not in redacted
    assert counts == {"private_key": 1}
    assert redacted == "Here's the key:\n[redacted:private_key]\nThat's all of it."


def test_redact_pem_private_key_block_negative_prose():
    text = "PEM files wrap key material between BEGIN and END marker lines."
    redacted, counts = redact_secrets(text)
    assert redacted == text
    assert counts == {}


# ── multiple secrets in one document ────────────────────────────────────────


def test_redact_multiple_secrets_and_counts():
    fake_anthropic = "sk-ant-api03-AbCdEfGh12345678901234567890ijklMNOPqrstUVWXyz9876"
    fake_aws = "AKIAIOSFODNN7EXAMPLE"
    text = (
        f"ANTHROPIC_API_KEY={fake_anthropic}\n"
        "Some unrelated prose about deploy plans.\n"
        f"AWS_ACCESS_KEY_ID={fake_aws}\n"
        f"AWS_ACCESS_KEY_ID_BACKUP={fake_aws}\n"
    )
    redacted, counts = redact_secrets(text)
    assert fake_anthropic not in redacted
    assert fake_aws not in redacted
    assert "Some unrelated prose about deploy plans." in redacted
    assert counts == {"anthropic_key": 1, "aws_access_key_id": 2}
