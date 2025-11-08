from src.translator import translate_content


def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"

def test_translation_german_sentence():
    # From notebook eval set: "Hier ist dein erstes Beispiel." -> "Here is your first example."
    is_english, translated = translate_content("Hier ist dein erstes Beispiel.")
    assert is_english is False
    assert translated == "Here is your first example."

def test_translation_french_question():
    # "Bonjour! Comment puis-je vous aider aujourd'hui?" -> "Hello! How can I help you today?"
    is_english, translated = translate_content("Bonjour! Comment puis-je vous aider aujourd'hui?")
    assert is_english is False
    assert translated == "Hello! How can I help you today?"

def test_translation_spanish_question():
    # "¿Dónde está la biblioteca?" -> "Where is the library?"
    is_english, translated = translate_content("¿Dónde está la biblioteca?")
    assert is_english is False
    assert translated == "Where is the library?"

def test_translation_chinese_sentence_alt():
    # "这是一个测试句子。" -> "This is a test sentence."
    is_english, translated = translate_content("这是一个测试句子。")
    assert is_english is False
    assert translated == "This is a test sentence."

def test_translation_emojis_unintelligible():
    # Unintelligible/emoji input should still produce a (False, text) tuple per notebook contract
    content = "😀😃😄😁🎉🎊🎈"
    is_english, translated = translate_content(content)
    assert is_english is False
    assert translated == content