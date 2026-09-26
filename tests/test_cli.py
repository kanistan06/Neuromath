import json

import app as app_module
import config
import main


def test_generate_template_command_reports_persisted_template(monkeypatch, capsys):
    versions = app_module._artifact_versions()
    assert versions["generation_provider"] == "runpod"
    assert versions["model_id"] == config.RUNPOD_MCQ_GENERATION_MODEL
    assert versions["embedding_provider"] == "hf-inference"
    assert versions["question_bank_version"] == "none"
    fingerprint = "f" * 64
    with app_module.app.app_context():
        app_module.db.session.add(
            app_module.AssessmentTemplate(
                fingerprint=fingerprint,
                paper_json="[]",
                **versions,
            )
        )
        app_module.db.session.commit()

    monkeypatch.setattr(app_module, "_student_diagnostic_syllabus", lambda _user: {})
    monkeypatch.setattr(app_module, "_diagnostic_fingerprint", lambda _value: fingerprint)
    monkeypatch.setattr(
        app_module,
        "_generate_and_save_paper",
        lambda _user, force: {"count": 25},
    )

    main.cmd_generate_template()

    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is True
    assert result["fingerprint"] == fingerprint
    assert result["question_count"] == 25
