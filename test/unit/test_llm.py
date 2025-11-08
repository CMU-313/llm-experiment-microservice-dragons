from unittest.mock import Mock
import pytest

from src.llm import LLMService


TRANSLATION_CTX = "You are a translator. Translate the user's message to English. Only output the translation."
CLASSIFICATION_CTX = "You are a language classifier. Reply only with the English name of the language of the user's message."


def make_mock_message(text: str):
	msg = Mock()
	type(msg).content = Mock(return_value=text)  # property-like access
	msg.content = text
	return msg


def make_mock_response(text: str):
	resp = Mock()
	resp.message = make_mock_message(text)
	return resp


def test_unexpected_language_then_translate():
	mock_client = Mock()
	# First call (language detection) returns an unclear sentence
	lang_resp = make_mock_response("I think this might be French, but I'm not sure")
	# Second call (translation) returns normal result
	trans_resp = make_mock_response("Hello, this is a test")
	mock_client.chat.side_effect = [lang_resp, trans_resp]

	service = LLMService(client=mock_client, model_name="test-model")
	result = service.query_llm_robust("Bonjour, ceci est un test", TRANSLATION_CTX, CLASSIFICATION_CTX)
	assert isinstance(result, tuple) and len(result) == 2
	assert result == (False, "Hello, this is a test")


def test_empty_language_and_translation_fallbacks_to_original():
	mock_client = Mock()
	mock_client.chat.return_value = make_mock_response("")
	service = LLMService(client=mock_client, model_name="test-model")

	post = "Hallo Welt"
	result = service.query_llm_robust(post, TRANSLATION_CTX, CLASSIFICATION_CTX)
	assert result == (False, post)


def test_api_exception_is_graceful():
	mock_client = Mock()
	mock_client.chat.side_effect = Exception("Connection timeout")
	service = LLMService(client=mock_client, model_name="test-model")

	post = "Test post"
	result = service.query_llm_robust(post, TRANSLATION_CTX, CLASSIFICATION_CTX)
	assert result == (False, post)


def test_malformed_json_language_then_translate():
	mock_client = Mock()
	lang_resp = make_mock_response('{"language": "English", "confidence": 0.95}')
	trans_resp = make_mock_response("Hello there")
	mock_client.chat.side_effect = [lang_resp, trans_resp]

	service = LLMService(client=mock_client, model_name="test-model")
	result = service.query_llm_robust("Hola, ¿cómo estás?", TRANSLATION_CTX, CLASSIFICATION_CTX)
	# Should treat unclear language as non-English and return translation
	assert result == (False, "Hello there")


def test_normal_english_bypasses_translation():
	mock_client = Mock()
	lang_resp = make_mock_response("English")
	mock_client.chat.side_effect = [lang_resp]

	service = LLMService(client=mock_client, model_name="test-model")
	post = "How are you?"
	result = service.query_llm_robust(post, TRANSLATION_CTX, CLASSIFICATION_CTX)
	assert result == (True, post)


def test_normal_non_english_translates():
	mock_client = Mock()
	lang_resp = make_mock_response("German")
	trans_resp = make_mock_response("Here is your first example.")
	mock_client.chat.side_effect = [lang_resp, trans_resp]

	service = LLMService(client=mock_client, model_name="test-model")
	result = service.query_llm("Hier ist dein erstes Beispiel.", TRANSLATION_CTX, CLASSIFICATION_CTX)
	assert result == (False, "Here is your first example.")


def test_get_language_normalizes_prefixes():
	mock_client = Mock()
	lang_resp = make_mock_response("Language: German\nConfidence: 0.95")
	mock_client.chat.return_value = lang_resp
	service = LLMService(client=mock_client, model_name="test-model")
	lang = service.get_language("Hier ist dein erstes Beispiel.", CLASSIFICATION_CTX)
	assert lang == "German"


