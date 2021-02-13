from jtran import JTran
from unicodedata import name as un_name


def _is_hiragana(char) -> bool:
    return 'HIRAGANA' == un_name(char).split()[0]


def _is_katakana(char) -> bool:
    return 'KATAKANA' == un_name(char).split()[0]


def _is_hira_or_kata(char) -> bool:
    return _is_hiragana(char) or _is_katakana(char) or char in ['ー', '々']


def _is_kanji(char) -> bool:
    return 'CJK' == un_name(char).split()[0]


def _is_cyr_or_lat(char) -> bool:
    return un_name(char).split()[0] in ['CYRILLIC', 'LATIN']


def _latin_to_hiragana(latin: str, colons_to_double_vowel: bool = True) -> str:
    return JTran.transliterate_from_latn_to_hrkt(latin, colons_to_double_vowel=colons_to_double_vowel)


def _latin_to_katakana(latin: str) -> str:
    return JTran.transliterate_from_hira_to_kana(JTran.transliterate_from_latn_to_hrkt(latin, False))


def _hiragana_to_latin(hiragana: str) -> str:
    return JTran.transliterate_from_hira_to_latn(hiragana)


def _distance(this: str, other: str) -> float:
    this_temp = this * 1
    common_chars = 0
    diff_chars = 0
    for char in other:
        if char in this_temp:
            common_chars += 1
            this_temp = this_temp.replace(char, '', 1)
        else:
            diff_chars += 1
    dist = float(common_chars) / max(len(this), len(other)) - (diff_chars + len(this_temp))

    this_kj = [ch for ch in this if _is_kanji(ch)]
    other_kj = [ch for ch in other if _is_kanji(ch)]
    common_kj = 0
    for kj in other_kj:
        if kj in this_kj:
            common_kj += 1
            this_kj.remove(kj)

    dist -= len(other_kj) / max((common_kj + len(this_kj)), 1) - 5 if not this_kj else 0

    return dist
