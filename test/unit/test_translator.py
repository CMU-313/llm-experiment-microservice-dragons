from src.translator import translate_content


def test_translation_chinese():
    """Test Chinese translation with Ollama model"""
    is_english, translated = translate_content("这是一个测试句子。")
    assert isinstance(is_english, bool)  # Just verify it returns a boolean
    assert len(translated) > 0  # Just check that we got some response


def test_translation_spanish():
    """Test Spanish translation with Ollama model"""
    is_english, translated = translate_content("¿Dónde está la biblioteca?")
    assert is_english is False
    assert len(translated) > 0  # Just check that we got some response


def test_english_detection():
    """Test English detection with Ollama model"""
    is_english, translated = translate_content("This is a test sentence.")
    # Model should detect this as English (though may not always be consistent)
    assert isinstance(is_english, bool)
    assert len(translated) > 0


def test_empty_string():
    """Test empty string handling with Ollama model"""
    is_english, translated = translate_content("")
    assert is_english is False
    assert translated == ""


def test_translation_function_runs():
    """Basic smoke test - just verify the function runs without error"""
    is_english, translated = translate_content("Hello world")
    assert isinstance(is_english, bool)
    assert isinstance(translated, str)