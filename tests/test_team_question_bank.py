import json
from fractions import Fraction
from pathlib import Path

import pytest

import config
from rag.question_bank import load_question_bank, questions_for_concept, question_bank_version
from rag.question_bank_import import import_question_bank, record_digest, read_team_dataset, REVIEW_FILE
from rag.quality import validate_generated_mcq


BANK_DIR = Path(__file__).resolve().parents[1] / "data" / "past_papers"


def test_shipped_bank_contains_only_reviewed_self_contained_tasks():
    bank = load_question_bank(BANK_DIR, require_reviewed=True)
    review = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    assert len(bank) == 46
    assert len({q.concept_id for q in bank}) == 23
    held = {key for key, row in review["records"].items() if row["status"] == "held"}
    assert len(held) == 15
    for question in bank:
        assert question.source_record_id not in held
        assert question.source_record_sha256 == review["records"][question.source_record_id]["source_sha256"]
        assert question.answer_review_evidence and question.mapping_evidence
        assert question.source_verification == "dataset_text_only"
        assert "required_for_answer" not in question.stem


def test_contradictory_probability_answer_is_replaced_by_verified_computation():
    bank = {question.question_id: question for question in load_question_bank(BANK_DIR)}
    question = bank['OL_2024_2025_MATH_B_05.iv']
    one_boy = Fraction(4 * 3 + 1 * 2, 5 * 5)
    losing = one_boy * Fraction(6, 10) + (1 - one_boy) * Fraction(7, 10)
    assert question.original_answer == "149/250"
    assert Fraction(question.answer) == losing == Fraction(161, 250)
    assert question.correction and question.source_reference()['correction']


def test_graph_answer_rounds_exact_intersections_not_rounded_endpoints():
    import math
    bank = {question.question_id: question for question in load_question_bank(BANK_DIR)}
    question = bank['OL_2024_2025_MATH_II_A_02.iv_b']
    assert question.answer == f'{2 * math.sqrt(3):.1f} units'
    assert question.answer != '3.4 units'
    assert question.correction


def test_tax_and_solid_volume_workings_match_the_stated_givens():
    bank = {question.question_id: question for question in load_question_bank(BANK_DIR)}
    salary = Fraction(118800 - 6000, 1) / Fraction(94, 100)
    assert salary == 120000
    assert '20000' in bank['OL_2024_2025_MATH_B_01.iv'].solution
    volume = Fraction(314, 100) * Fraction(1725, 1000)**3 / 2
    assert float(volume) == pytest.approx(8.05873640625)
    assert '8.05873640625' in bank['OL_2024_2025_MATH_II_B_09.side_length'].solution
    assert Fraction(2 * 2**3) - Fraction(2 * 2**3, 3) == Fraction(32, 3)


def _input_and_review(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    meta = {'paper_id': 'fixture', 'examination_year': '2024(2025)', 'paper': 'Paper II',
            'source_question_paper': 'fixture.pdf'}
    original = {'question_id': 'fixture-paper-01', 'question_number': 1,
                'question_text': 'A square has side 3 cm. Find its perimeter.',
                'answers': {'final_answer': '12 cm'},
                'image_references': {'question_paper_page': 1}}
    (source / 'fixture.json').write_text(json.dumps({'paper_meta': meta, 'parts': {'Part A': {'questions': [original]}}}))
    (source / 'schema.json').write_text(json.dumps({'questions': [{'question_id': 'example-do-not-import'}]}))
    record = {'suffix': '', 'grade': 10, 'concept_id': 'g10_t1_01_perimeter', 'concept_name': 'Perimeter',
              'question': original['question_text'], 'answer': '12 cm', 'solution': '4*3=12 cm.',
              'mapping_evidence': 'Sum four equal sides of a square.', 'answer_review_evidence': '4*3=12 cm.'}
    review = {'review_version': 'fixture-v1', 'reviewed_by': 'test-review', 'reviewed_at': '2026-09-16T00:00:00+00:00',
              'source_verification': 'dataset_text_only', 'records': {original['question_id']: {
                  'source_dataset': 'fixture.json', 'source_sha256': record_digest(meta, original),
                  'status': 'accepted', 'reason': 'Checked from supplied text.', 'records': [record]}}}
    review_file = tmp_path / 'review.json'
    review_file.write_text(json.dumps(review))
    return source, review_file


def test_nested_import_skips_schema_and_is_idempotent(tmp_path):
    source, review = _input_and_review(tmp_path)
    output = tmp_path / 'bank'
    result = import_question_bank(source, output_dir=output, review_file=review)
    target = Path(result['output'])
    before = target.read_bytes(), target.stat().st_mtime_ns
    assert result['source_question_count'] == result['imported_question_count'] == 1
    second = import_question_bank(source, output_dir=output, review_file=review)
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert load_question_bank(output)[0].canonical_answer == '12 cm'


def test_changed_source_cannot_reuse_old_review_or_replace_existing_bank(tmp_path):
    source, review = _input_and_review(tmp_path)
    output = tmp_path / 'bank'
    result = import_question_bank(source, output_dir=output, review_file=review)
    target = Path(result['output'])
    before = target.read_bytes()
    raw = json.loads((source / 'fixture.json').read_text())
    raw['parts']['Part A']['questions'][0]['question_text'] = 'Changed mathematical givens.'
    (source / 'fixture.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='No reviewed questions'):
        import_question_bank(source, output_dir=output, review_file=review)
    assert target.read_bytes() == before


def test_duplicate_source_ids_and_unsafe_report_location_fail(tmp_path):
    source, review = _input_and_review(tmp_path)
    (source / 'duplicate.json').write_bytes((source / 'fixture.json').read_bytes())
    with pytest.raises(ValueError, match='Duplicate source question_id'):
        read_team_dataset(source)
    with pytest.raises(ValueError, match='outside'):
        import_question_bank(source, review_file=review, output_dir=tmp_path/'bank', report_path=source/'report.json')


def test_student_source_exclusions_and_grades_apply_to_actual_bank():
    bank = load_question_bank(BANK_DIR)
    source = next(q for q in bank if q.question_id.endswith('B_05.iv'))
    matches = questions_for_concept(source.concept_id, grade=11, questions=bank, seed='alice', limit=100)
    assert source in matches
    assert all(q.grade == 11 for q in matches)
    assert not questions_for_concept(source.concept_id, grade=10, questions=bank)
    unseen = questions_for_concept(source.concept_id, grade=11, questions=bank,
                                   exclude_question_ids={source.question_id}, seed='alice', limit=100)
    assert source not in unseen
    assert question_bank_version(bank) != 'none'


def test_generated_mcq_rejects_wrong_source_answer_grade_and_concept():
    source = next(q for q in load_question_bank(BANK_DIR) if q.question_id.endswith('B_05.iv'))
    question = {'question': 'Two teams are sampled independently. After applying the conditional win rules, which total probability describes a non-win for the younger team?',
                'options': ['149/250', '159/250', '161/250', '163/250'], 'answer': 'A',
                'explanation': 'Correct result: 149/250. Combine the disjoint branches.',
                'topic_id': source.concept_id, 'grade': 11, 'source_question_id': source.question_id}
    assert 'correct option does not match the verified source answer' in validate_generated_mcq(question, past_questions=[source.prompt_payload()])
    question.update(answer='C', explanation='Correct result: 161/250. Add 84/250 and 77/250.', grade=10, topic_id='another-topic')
    reasons = validate_generated_mcq(question, past_questions=[source.prompt_payload()])
    assert 'past-paper source belongs to a different grade' in reasons
    assert 'past-paper source belongs to a different concept' in reasons


def test_edited_imported_answer_invalidates_its_review(tmp_path):
    source, review = _input_and_review(tmp_path)
    output = tmp_path / 'bank'
    result = import_question_bank(source, output_dir=output, review_file=review)
    target = Path(result['output'])
    bank = json.loads(target.read_text())
    bank['questions'][0]['answer'] = '13 cm'
    target.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match='changed after review'):
        load_question_bank(output)
