import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

import config
import rag.generator as generator
import rag.retriever as retriever
import rag.runpod_inference as inference
from rag.question_bank import load_question_bank
from rag.validation import has_current_review


CONTEXT = ('For a fixed amount of work, the number of workers multiplied by the time taken remains constant. '
           'Find the required workforce for the new time and subtract the original workforce to find additional workers.')


def _install(monkeypatch):
    root = Path(__file__).resolve().parents[1] / 'data' / 'past_papers'
    monkeypatch.setattr(config, 'QUESTION_BANK_DIR', root)
    source = next(q for q in load_question_bank() if q.question_id.endswith('MATH_A_01'))
    monkeypatch.setattr(retriever, 'get_vector_store', lambda: object())
    document = Document(page_content=CONTEXT, metadata={'grade': 10, 'source': 'grade10.pdf', 'page': 0,
                    'page_start': 1, 'page_end': 1, 'chunk_id': 'fixture-work', 'corpus_version': 'fixture-v1'})
    monkeypatch.setattr(retriever, '_hybrid_search', lambda **_kwargs: [document])
    syllabus = {'assessment_type': 'practice', 'total_questions': 1, 'topic_mappings': [{
        'topic_id': source.concept_id, 'topic_name': source.concept_name, 'grade': 10,
        'required_levels': [3], 'min_questions': 1, 'max_questions': 1,
        'strict_textbook_grounding': True, 'require_application': True,
    }]}
    return source, syllabus


def _candidate(source):
    return {'question': 'A contractor schedules a fixed painting job for 6 equally productive men over 8 days. '
                        'The deadline is shortened to 3 days. How many additional men must join the original crew?',
            'options': ['8 men', '9 men', '10 men', '11 men'], 'correct_answer': '10 men',
            'explanation': 'Correct result: 10 men. The job takes 48 man-days; 16 men are needed in 3 days, so add 10.',
            'evidence_id': 'E1', 'source_question_id': source.question_id}


def test_actual_bank_retrieval_flows_through_runpod_generation_and_blinded_review(monkeypatch):
    source, syllabus = _install(monkeypatch)
    calls = []
    class Session:
        def post(self, url, **kwargs):
            payload = kwargs['json']
            calls.append((url, payload))
            schema = payload['response_format']['json_schema']['name']
            if schema == 'neuromath_mcq_review':
                result = {'answer_value': '10 men', 'valid_option_values': ['10 men'],
                    'explanation': 'The work is 6*8=48 man-days. Required workers=48/3=16, so additional workers=16-6=10.',
                    'difficulty_level': 'Apply', 'unambiguous': True, 'concept_relevant': True,
                    'textbook_supported': True, 'distractors_plausible': True, 'application_required': True, 'reason': ''}
            else:
                assert source.question_id in payload['messages'][-1]['content']
                assert 'Do not introduce new numerical values' in payload['messages'][-1]['content']
                result = {'questions': [_candidate(source)]}
            class Response:
                status_code = 200
                def json(self):
                    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}
            return Response()
    monkeypatch.setattr(inference, '_http_session', lambda: Session())
    content = retriever.retrieve_for_syllabus(syllabus, question_seed='student-1')
    paper = generator.generate_paper(syllabus, content)
    assert len(paper) == 1 and len(calls) == 2
    assert all(url.startswith('https://api.runpod.ai/') for url, _ in calls)
    question = paper[0]
    assert question['options']['ABCD'.index(question['answer'])] == '10 men'
    assert question['source_question_id'] == source.question_id
    assert question['past_paper_refs'][0]['source_record_sha256'] == source.source_record_sha256
    assert question['source_refs'][0]['source'] == 'grade10.pdf'
    assert has_current_review(question)
    again = retriever.retrieve_for_syllabus(syllabus, exclude_question_ids_by_topic={source.concept_id: {source.question_id}})
    assert again[source.concept_id]['past_questions'] == []
    assert again[source.concept_id]['grounding_units']


def test_changed_answer_after_review_cannot_bypass_source_validation(monkeypatch):
    import rag.validation as validation
    source, _ = _install(monkeypatch)
    monkeypatch.setattr(generator, '_invoke_llm', lambda *_args, **_kwargs: json.dumps({'questions': [_candidate(source)]}))
    def tamper(question, **_kwargs):
        question.update(answer='B', correct_answer='9 men', explanation='Correct result: 9 men. A changed result.')
        return []
    monkeypatch.setattr(validation, 'review_mcq', tamper)
    result = generator.generate_mcqs(source.concept_name, 'Apply', [CONTEXT], 1,
             grade=source.grade, concept_id=source.concept_id, past_questions=[source.prompt_payload()])
    assert result == []
