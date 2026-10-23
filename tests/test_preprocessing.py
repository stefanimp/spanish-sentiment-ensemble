import pytest

from sentiment_ensemble.preprocessing import clean_text


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("robertuito_uncased", "hola @usuario hashtag feliz url !!! 😊"),
        ("cased", "Hola @usuario Feliz URL !!! 😊"),
    ],
)
def test_social_text_normalization(mode, expected):
    assert clean_text(" Hola @Ana #Feliz https://example.org !!!!!😊 ", mode) == expected


def test_missing_text_is_empty():
    assert clean_text(None, "cased") == ""
