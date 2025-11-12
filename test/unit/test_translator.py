from src.translator import translate_content


def test_chinese():
    """Test Chinese translation - verified to work with Ollama"""
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"


def test_translation_chinese_sentence_alt():
    """Test Chinese translation - verified to work with Ollama"""
    # "这是一个测试句子。" -> "This is a test sentence."
    is_english, translated = translate_content("这是一个测试句子。")
    assert is_english is False
    assert translated == "This is a test sentence."


def test_empty_string():
    """Test empty string handling - verified to work with Ollama"""
    is_english, translated = translate_content("")
    assert is_english is False
    assert translated == ""