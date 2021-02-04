import os
import pathlib
import pickle
import re
import sqlite3
from dataclasses import dataclass
from operator import itemgetter
from typing import List, Tuple

from tqdm import tqdm

from src.entry import YarxiEntry
from src.reference import YarxiReference
from src.search import SearchMode, SearchResult
from src.utils import _is_kanji, _is_hira_or_kata, _is_hiragana, _distance, _latin_to_hiragana, _latin_to_katakana


class YarxiDictionary:
    _entries: List[YarxiEntry]

    def lookup(self, lexeme: str, reading: str = '', search_mode: SearchMode = SearchMode.consecutive,
               order: int = 1) -> List[SearchResult]:
        if order < 1:
            raise Exception(f'Error: order value ({order}) is less than 1')
        if int(search_mode.value) not in [0, 1, 2]:
            raise Exception(f'Error: unknown search mode ({search_mode})')
        if any(not (_is_kanji(char) or _is_hira_or_kata(char)) for char in lexeme):
            raise Exception(f'Error: bad characters in lexeme ({lexeme})')
        if reading and any(not _is_hiragana(char) for char in reading):
            raise Exception(f'Error: bad characters in reading ({reading})')

        res = []
        suitable_entries = []
        if search_mode != SearchMode.deep_only:
            suitable_entries = [e for e in self._entries if lexeme in e.lexeme]
            if reading:
                suitable_entries = [e for e in suitable_entries if reading in e.reading]

        if search_mode != SearchMode.shallow_only and not suitable_entries:
            lex_kanji = [k for k in lexeme if _is_kanji(k)]
            if lex_kanji:
                for char in lexeme:
                    if _is_kanji(char):
                        res.extend(
                            [entry for entry in self._entries if any(lex for lex in entry.lexeme if char in lex)])

                res = list(set(res))
                if reading:
                    narrowed_res = [r for r in res if reading in r.reading]
                    if narrowed_res:
                        res = narrowed_res

                weighted_res = []

                for r in res:
                    weighted_res.append([0.0, r.eid])

                if len(weighted_res) > 1:
                    for w_r in weighted_res:
                        for lex in self._get_entry_by_eid(w_r[1]).lexeme:
                            w_r[0] = _distance(lexeme, lex)

                weighted_res.sort(key=itemgetter(0), reverse=True)
                orders = sorted(list(set([r[0] for r in weighted_res])), reverse=True)
                suitable_entries = [self._get_entry_by_eid(e[1]) for e in weighted_res if
                                    e[0] >= orders[min(len(orders) - 1, order - 1)]]
            else:
                suitable_entries = [e for e in self._entries if lexeme in e.lexeme or lexeme in e.reading]

        final_res = []

        for entry in suitable_entries:
            final_res.append(
                SearchResult(reading=entry.reading, lexeme=entry.lexeme,
                             translation=self._translate_by_eid(entry.eid)))
        return final_res

    def lookup_translations_only(self, lexeme: str, reading: str = '', search_mode: SearchMode = SearchMode.consecutive,
                                 order: int = 1) -> List[str]:
        return list(
            set(sum([tr.translation for tr in self.lookup(lexeme, reading, search_mode, order)], [])))

    def _get_entry_by_eid(self, eid: str) -> YarxiEntry:
        return [e for e in self._entries if e.eid == eid][0]

    def __init__(self, entries: List[YarxiEntry]):
        self._entries = entries

    def _translate_by_eid(self, eid: str) -> List[str]:
        def _retrieve_from_references(refs: List[YarxiReference], visited_references: List[str]) -> List[str]:
            if not refs:
                return []

            usable_refs = [ref for ref in refs if ref.usable]

            referenced_entries = [(usable_ref.verified, usable_ref.mode, self._get_entry_by_eid(usable_ref.eid)) for
                                  usable_ref in
                                  usable_refs]

            translations = []
            usable_refs = []

            for reference_verified, reference_mode, referenced_entry in referenced_entries:
                if referenced_entry.eid in visited_references:
                    continue
                if reference_verified:
                    translations.extend([' '.join([reference_mode, tr]).strip() for tr in referenced_entry.translation])
                elif self._unreliable_translations:
                    translations.extend(
                        [self._unreliable_translations_pref + reference_mode + tr for tr in
                         referenced_entry.translation])

                usable_refs.extend([ref for ref in referenced_entry.references if ref.usable])
                visited_references.append(referenced_entry.eid)

            return translations + _retrieve_from_references(usable_refs, visited_references)

        entry = self._get_entry_by_eid(eid)
        return entry.translation + _retrieve_from_references(entry.references, [entry.eid])

    def save(self, fname: str = "default"):
        if fname == 'default':
            fname = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.joinpath('dictionaries/yarxi.jtdb')
        pickle.dump(self, open(fname, "wb"))


class YarxiLoader:
    @dataclass
    class _Kanji:
        kanji: str
        kun: str
        on: str
        rus: str
        rus_nick: str

    _entries: List[YarxiEntry]
    _kanji_db: {str: (str, str, str, str)}

    _unreliable_translations: bool = True
    _unreliable_translations_pref: str = '〈возм. перевод〉 '
    _in_compounds_pref: str = '〈в сочет.〉 '

    _last_eid: int = -1

    _collocations_right: {str: str} = {'*1': 'suru', '*2': 'na', '**2': 'no aru', '*3': 'no', '*=56': 'shite aru',
                                       '*7': 'taru',
                                       '*=28': 'naku', '**1': 'ga aru', '*5': 'de', '**3': 'no nai', '**7': 'ni suru',
                                       '**0': 'wo suru',
                                       '*=32': 'teki', '***8': 'to shita', '**4': 'de aru', '**8': 'ni naru',
                                       '*0': 'shita',
                                       '*4': 'ni', '*6': 'to', '*=30': 'ni (~de)', '***1': 'kara', '*=02': 'atte',
                                       '*=01': 'aru',
                                       '**9': 'to shite', '*=03': 'ga nai', '*=16': 'mo nai', '**6': 'da',
                                       '*=43': '[ni] suru',
                                       '*=53': 'wo yaru', '*=66': 'to [shite]', '*8': 'shite', '***5': 'to suru',
                                       '*=11': 'desu ka?', '*=07': 'demo', '***9': 'to shite iru', '*=06': 'ga suru',
                                       '*=31': 'ni mo', '*=86': 'subeki', '*=55': 'sareru', '*=54': 'saseru',
                                       '*=69': 'to nareba',
                                       '***3': 'mo', '*=04': 'ga atte', '*=14': 'ka', '*9': 'to shite',
                                       '*=24': 'na[ru]',
                                       '*=91': 'to natte', '*=62': 'to naru', '***2': 'made', '*=15': 'made mo',
                                       '*=34': 'ni [shite]',
                                       '*=23': 'narazu', '*=33': 'ni shite', '*=08': 'de wa', '*=13': 'de suru',
                                       '*=19': 'nagara',
                                       '*=37': 'ni natte iru', '*=51': 'wo shite iru', '*=71': 'e', '*=84': 'wo shite',
                                       '*=79': 'naraba',
                                       '*=59': 'shinai', '*=36': 'ni natte', '*=09': 'de nai', '*=81': 'narashimeru',
                                       '**5': 'desu',
                                       '*=25': 'nasai', '*=10': 'de (~ni)', '*=18': 'mo naku', '***4': 'wa',
                                       '*=73': 'ga gotoshi',
                                       '*=12': 'deshita', '*=29': 'ni (~wa)', '***0': 'naru', '*=50': '[wo] suru',
                                       '*=45': '[no] aru',
                                       '*=27': 'naki', '*=35': 'ni nai', '*=21': 'na (~no)', '*=38': 'ni aru',
                                       '*=82': 'ni [natte]',
                                       '*=87': 'sureba', '*=20': 'nai', '*=00': '!', '*=65': '[to] shita',
                                       '*=92': 'suru na',
                                       '*=67': 'to mo', '*=60': 'sezu ni', '*=74': 'de kara', '*=42': 'sarete',
                                       '*=40': 'ni iru',
                                       '*=58': 'shimasu', '*=44': 'ni yaru', '*=88': 'shite mo', '*=90': 'yaru',
                                       '*=57': 'su',
                                       '*=41': 'ni naranai', '*=17': '[mo] nai', '*=49': 'wo', '*=93': 'ni oite',
                                       '*=78': 'su[ru]',
                                       '*=85': 'shinagara', '*=26': 'nashi no', '*=52': 'wo shita', '*=39': 'ni sareru',
                                       '*=48': 'no suru', '*=70': 'to [naku]', '***6': 'yori', '*=80': 'naranu',
                                       '***7': 'ga shite iru',
                                       '*=89': 'to mo shinai', '*=77': 'mono de', '*=46': 'no shita',
                                       '*=47': 'no shinai',
                                       '*=61': 'seru', '*=72': 'sarete iru', '*=83': 'wo shinai', '*=22': 'narazaru',
                                       '*=75': 'dake no', '*=05': 'ga shite aru', '*=68': 'to mo sureba',
                                       '*=63': 'to saseru'}
    _collocations_left: {str: str} = {'*-6': 'to', '*-3': 'no', '*-4': 'ni', '*-7': 'wo', '*-0': 'wa', '*-8': 'ga',
                                      '*-1': '(kara)',
                                      '*-5': '(de)', '*-9': '(suru)'}

    def _get_entry_by_eid(self, eid: str) -> YarxiEntry:
        return [e for e in self._entries if e.eid == eid][0]

    def _get_kanji(self, code: str) -> str:
        return self._kanji_db[code].kanji

    def _get_next_eid(self) -> str:
        self._last_eid += 1
        return str(self._last_eid)

    def _load_kanji_db(self, fname, show_progress: bool = True) -> {str: str}:
        conn = sqlite3.connect(fname)
        cursor = conn.cursor()
        res = {'0': self._Kanji('', '', '', '', ''), '-1': self._Kanji('', '', '', '', '')}

        kanji = cursor.execute('SELECT Nomer, Uncd, Kunyomi, Onyomi, Russian, RusNick from Kanji').fetchall()
        for eid, uncd, kun, on, rus, rus_nick in tqdm(kanji, desc='[Yarxi] Building kanji database'.ljust(34),
                                                      disable=not show_progress):
            res[str(eid)] = self._Kanji(kanji=chr(int(uncd)), kun=kun, on=on, rus=rus, rus_nick=rus_nick.lower())
        conn.close()

        return res

    def _resolve_references(self, show_progress: bool) -> [YarxiReference]:
        for entry in tqdm(self._entries, desc='[Yarxi] Resolving references'.ljust(34), disable=not show_progress):
            refs = entry.references
            if refs:
                original_length = len(refs)
                for i in range(0, original_length):
                    if refs[i].eid:
                        refs[i].verified = True
                        refs[i].usable = True
                    else:
                        fitting = [et for et in self._entries if any(lex in et.lexeme for lex in refs[i].lexeme)]
                        if len(fitting) > 1:
                            fitting = [f for f in fitting if list(set(f.reading) & set(entry.reading))]
                        if len(fitting) == 1:
                            refs[i].eid = fitting[0].eid
                            refs[i].lexeme = ''
                            refs[i].verified = True
                            refs[i].usable = True
                        else:
                            for fit in [et for et in self._entries if refs[i].lexeme in et.lexeme]:
                                refs.append(YarxiReference(eid=fit.eid, verified=False, usable=True, mode=refs[i].mode))

    def _load_db(self, fname, show_progress: bool = True) -> List[YarxiEntry]:
        conn = sqlite3.connect(fname)
        cursor = conn.cursor()
        res = []

        tango = cursor.execute(f"SELECT K1, K2, K3, K4, Kana, Reading, Russian, Nomer, Hyphens FROM Tango").fetchall()[
                :-1]
        for k1, k2, k3, k4, kana, reading, translation, eid, hyphens in tqdm(tango,
                                                                             desc='[Yarxi] Building word database'.ljust(
                                                                                 34),
                                                                             disable=not show_progress):
            res.extend(self._convert_to_entry_tango(
                (str(k1), str(k2), str(k3), str(k4), kana, reading, translation, str(eid), hyphens)))
        conn.close()

        self._last_eid = int(res[-1].eid)

        return res

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'^_', '', text)
        text = re.sub(r'\[\\\'\'\\\^(.*)\\\'\'\\\]', lambda m: f'(от «{m.group(1)}»)', text)
        text = re.sub(r'(\^[\^|@])', '', text)
        text = re.sub(r'([^\s])_([^\s])', r'\1, \2', text)
        text = re.sub(r'^!', '', text)
        text = re.sub(r"-\\''\\\w+\\''", '', text)
        text = text.replace('@3(', ' и т.п. (')
        text = text.replace('@3', ' и т.п.')
        text = re.sub(r'(\(#)(.*?)(#\))', r'(\2)', text)
        text = re.sub(r'#(.*?)#', r'\\《\1》\\', text)
        text = text.replace('#', '')
        text = re.sub(r'(\\\'\'\\)([^\s].*)(\\\'\'\\)', r'\\«\2»\\', text)
        text = text.replace('\'', '')
        for _ in [1, 2]:
            text = re.sub(r'([^\(\[《\-\s])\\(《|[\\|\(|#|\w|\d|«|\[])', r'\1 \2', text)
        text = text.replace(')(', ') (')
        text = text.replace('》(', '》 (')
        text = text.replace('\\', '')
        text = text.replace('^^', '')
        text = text.replace('= ', ' ')
        text = re.sub(r'^[^\[]*(\*\d+])', r'[\1', text)
        text = re.sub(r'(\s?)(\*+=?\d+)',
                      lambda m: f" 〈~{self._collocations_right[m.group(2)]}〉",
                      text)
        text = re.sub(r'(\s?)(\*+-\d+)',
                      lambda m: f" 〈-{self._collocations_left[m.group(2)]}〉",
                      text)
        text = text.replace('[ 〈', '[〈')
        return text.lower().strip()

    def _convert_to_entry_tango(self, info: Tuple[str, str, str, str, str, str, str, str, str]) -> List[YarxiEntry]:
        def _resolve_readings(reading: str, variable: bool, hyphens: str) -> List[str]:
            def _resolve_hh(reading: str, hyphens: str) -> str:
                for h_pos in re.findall(r'\d+', hyphens):
                    if 'hh' not in reading[:int(h_pos)] + reading[int(h_pos) + 1:]:
                        reading = reading[:int(h_pos)] + reading[int(h_pos) + 1:]
                return reading

            reading = re.sub(r'(Q\d)', '', reading)
            reading = reading.replace('$', '')
            reading = reading.replace('L1', '').replace('L2', '')
            reading = reading.replace('=', '')
            reading = reading.replace(' ', '')

            if variable:
                split_reading = reading.split('*(*')
                full_reading = [r for r in split_reading[0].split('*') if r]
                if len(split_reading) > 1:
                    base_reading = [r for r in split_reading[1].split('*') if r][-len(full_reading):]
                else:
                    base_reading = []
                res = full_reading + base_reading
            else:
                res = [r for r in reading.split('*(*')[0].split('*')]

            if 'hh' in res[0]:
                for i in range(0, len(res)):
                    res[i] = _resolve_hh(res[i], hyphens)

            return sorted([_latin_to_hiragana(rd.lower()).strip() for rd in res if rd], key=len)

        def _resolve_lexemes(lexeme: str) -> (List[str], List[str], bool):
            kanji = [('1', self._get_kanji(k1)), ('2', self._get_kanji(k2)), ('3', self._get_kanji(k3)),
                     ('4', self._get_kanji(k4))]
            if not lexeme:
                return [k[1] for k in kanji if k[1]], [''.join([k[1] for k in kanji])], False
            if not [kj for kj in lexeme if kj.isnumeric() and kj != '0']:
                lexeme += '1234'

            other = {'0': '', "'": 'っ'}

            additional_kanji = re.findall(r'#(\d+)#', lexeme)
            for add_kj in additional_kanji:
                kanji.append((len(kanji), self._get_kanji(add_kj)))
            kanji_for_entry = [k[1] for k in kanji if k[1]]
            lexeme = re.sub(r'(#\d+#)', '', lexeme)

            lex_temp = ''
            cur_pos = 0
            while True:
                if cur_pos == len(lexeme):
                    lexeme = lex_temp + ''.join([kj[1] for kj in kanji])
                    break
                if lexeme[cur_pos].isnumeric():
                    if kanji and int(lexeme[cur_pos]) > int(kanji[0][0]):
                        for i in range(int(lexeme[cur_pos]), int(kanji[0][0]) - 1, -1):
                            if not kanji:
                                break
                            lex_temp += kanji[0][1]
                            kanji = kanji[1:]
                        cur_pos += 1
                    elif kanji and lexeme[cur_pos] == kanji[0][0]:
                        lex_temp += kanji[0][1]
                        kanji = kanji[1:]
                        cur_pos += 1
                    else:
                        cur_pos += 1
                        continue
                else:
                    lex_temp += lexeme[cur_pos]
                    cur_pos += 1

            for key, value in other.items():
                lexeme = lexeme.replace(key, value)

            if '^' in lexeme:
                lexeme = re.sub(r'\^([\w:]+)@?', lambda x: _latin_to_katakana(x.group(1)), lexeme)
                lexeme = re.sub(r'([\w:]*)', lambda x: _latin_to_hiragana(x.group(1)), lexeme)
            else:
                lexeme = _latin_to_hiragana(lexeme, False)

            if '[' in lexeme:
                return kanji_for_entry, [re.sub(r'(\[(.*?)\])', '', lexeme).strip(),
                                         re.sub(r'\[(.*?)\]', r'\1', lexeme).strip()], False
            elif '(' in lexeme:
                return kanji_for_entry, [re.sub(r'(\((.*?)\))', '', lexeme).strip(),
                                         re.sub(r'\((.*?)\)', r'\1', lexeme).strip()], True

            return kanji_for_entry, [lexeme.strip()], False

        def _extract_references(translation: str) -> (List[str], List[str]):
            def _extract_reference(translation: str) -> (str, str):
                res = []

                word_refs = re.findall(r'\^{2}10*(\d+)(?:\_)?', translation)
                word_refs.extend(re.findall(r'\^0+(\d+)(?:\_)?', translation))
                word_refs.extend(re.findall(r'\^20*(\d+)(?:\_)?', translation))

                if word_refs:
                    if re.search(r'^\\?#\\?.*?\\?#\\?\^\d+\\_', translation) is not None:
                        mode = self._clean_text(re.search(r'^(\\?#\\?.*?\\?#\\?)\^\d+\\_', translation).group(1))
                        translation = ''
                    else:
                        mode = ''

                    for word_ref in word_refs:
                        res.append(YarxiReference(eid=re.search(r'0*(\d+)', word_ref).group(1), mode=mode))

                kanji_refs = []
                kanji_refs.extend(re.findall(r'\^0-0*(\d+)-?(?:\\\'\'\\([\w|\s]+)\\\'\')?', translation))
                kanji_refs.extend(re.findall(r'\^2-0*(\d+)-?(?:\\\'\'\\([\w|\s]+)\\\'\')?', translation))

                if kanji_refs:
                    if re.search(r'^\\?#\\?.*?\\?#\\?\^\d+\\_', translation) is not None:
                        mode = self._clean_text(re.search(r'^(\\?#\\?.*?\\?#\\?)\^\d+\\_', translation).group(1))
                        translation = ''
                    else:
                        mode = ''

                    for kanji_ref in kanji_refs:
                        res.append(
                            YarxiReference(
                                lexeme=[self._get_kanji(kanji_ref[0]) + _latin_to_hiragana(kanji_ref[1].strip())],
                                eid='', mode=mode))

                return res, re.sub(r'(\^+\d-?\d+)(-\\\'\'\\([\w|\s]+)\\\'\'\\)?(?:\\_)?', '', translation)

            def _split_translation(translation: str) -> List[str]:
                translation = re.sub(r'^>{2}', '', translation)
                translation = translation.replace('$', '')
                translation = re.sub(r'\(!\d+\)', '', translation)
                translation = translation.replace('^#', '')

                common_part = ''
                cnt = -1
                while True:
                    cnt += 1
                    if cnt == len(translation) or translation[cnt] == '&':
                        break
                    else:
                        common_part += translation[cnt]

                numbered_translations = [tr for tr in translation[cnt + 1:].split('&') if tr]
                if numbered_translations and len(numbered_translations[0]) == len(translation) - 1:
                    return numbered_translations

                if len(numbered_translations) <= 1:
                    return [common_part]

                return [common_part + tr for tr in numbered_translations]

            ref_tr = [_extract_reference(tr.strip()) for tr in _split_translation(translation)]

            return sum([elem[0] for elem in ref_tr], []), [elem[1] for elem in ref_tr if elem[1]]

        def _resolve_translations(translation: List[str]) -> List[str]:
            def _clean_translation(translation: str) -> str:
                generic_translations = {'1': 'мужское имя', '11': 'мужское имя',
                                        '2': 'женское имя', '22': 'женское имя',
                                        '3': 'фамилия', '33': 'фамилии',
                                        '4': 'псевдоним', '44': 'псевдонимы',
                                        '5': 'топоним', '55': 'топонимы'}

                g_tr = re.search(r'>(\d+)', translation)
                temp = []
                if g_tr is not None:
                    for tr in g_tr.group(1):
                        if tr != '0':
                            temp.append(generic_translations[tr])
                    translation = f"《{' или '.join(temp)}》"
                else:
                    translation = self._clean_text(translation)

                return translation

            res = []

            for tr in translation:
                cleaned = _clean_translation(tr)
                if cleaned:
                    res.append(cleaned)

            return res

        res = []

        k1, k2, k3, k4, lexeme_schema, reading, translation, eid, hyphens = info

        kanji, lexemes, variable_reading = _resolve_lexemes(lexeme_schema)
        readings = _resolve_readings(reading, variable_reading, hyphens)
        references, translations = _extract_references(translation)
        if any('^^' in tr for tr in translations):
            lexemes.extend(readings)
        translations = _resolve_translations(translations)

        res.append(
            YarxiEntry(reading=readings, lexeme=lexemes, translation=translations, eid=eid, references=references,
                       kanji=kanji))

        return res

    def _extract_compound_values(self, kanji: {str: _Kanji}, show_progress: bool):
        def _split_and_clean_compound_translations(translations: str, rus_nick: str):
            translations = re.sub(r'{!.*}', '-', translations)
            translations = translations.replace('{^^^}', '')
            translations = translations.replace('^#', '')

            if translations == '-':
                return [([], '')]

            reading_numbers = []
            final_trs = []
            translations = re.sub(r'\{\^*\w\d+\}', '', translations)
            translations = re.sub(r'(\[!.*?\])', '', translations)
            translations = re.sub(r'(\+{\(.*\)})', '', translations)

            generic_translations = {'@9': '《тж. счетный суффикс》', '@3': '', '@7': '', '@6': rus_nick, '@4': rus_nick,
                                    '@2': rus_nick, '@1': rus_nick, '@5': '《встречается в географических названиях》',
                                    '@8': '《в сочетаниях идиоматичен》', '@l': '《употребляется в летоисчислении》',
                                    '@0': '《употребляется фонетически》'}

            for translation in [tr for tr in translations.split('/') if tr]:
                nums_present = re.search(r'\((\d*)\)', translation)
                if nums_present is None:
                    reading_numbers.append(['0'])
                else:
                    reading_numbers.append(list(nums_present.groups()))

                for g_tr in generic_translations.keys():
                    if g_tr in translation:
                        translation = translation.replace(g_tr, f' {generic_translations[g_tr]} ').strip()

                final_trs.append(self._clean_text(re.sub(r'(\(\d*\))', '', translation)))

            return list(zip(reading_numbers, [fin_tr for fin_tr in final_trs if fin_tr]))

        extension = []

        for kj in tqdm(list(kanji.values())[2:], desc="[Yarxi] Updating kanji database".ljust(34), disable=not
        show_progress):

            comp_readings = {'0': [_latin_to_hiragana(on) for on in re.split(r'[*,;\)\(-]', kj.on) if on]}

            cleaned_kun = kj.kun.split('||$')[0]

            comp_readings_temp = re.search(r'\|(.*)', cleaned_kun)
            if comp_readings_temp:
                for sp_c_r in comp_readings_temp.group(1).split('/'):
                    readings = re.split(r'[_|,]', re.sub(r'q\d', 'い', _latin_to_hiragana(
                        sp_c_r.replace('-', '').replace(' ', '').lower())))
                    comp_readings[str(len(comp_readings))] = readings

            if not comp_readings['0']:
                continue

            comp_translations = kj.rus.split('|')

            if len(comp_translations) > 1:
                comp_translations = _split_and_clean_compound_translations(comp_translations[1], kj.rus_nick)
                for tr in comp_translations:
                    if not tr[0]:
                        translation = re.sub(r'(.*\*#\*)(.*)(\*)', r'\2', kj.rus_nick)
                        translation = re.sub(r'\'\'(.*)\'\'', lambda match: f'«{_latin_to_hiragana(match.group(1))}»',
                                             translation)
                        already_there = [e for e in extension if
                                         e.reading == [cr for cr in comp_readings['0'] if cr] and e.lexeme == [
                                             kj.kanji]]
                        if not already_there:
                            extension.append(
                                YarxiEntry(reading=[cr for cr in comp_readings['0'] if cr], lexeme=[kj.kanji],
                                           translation=[self._in_compounds_pref + translation],
                                           eid=str(self._get_next_eid()), references=[], kanji=kj.kanji))
                        else:
                            extension[extension.index(already_there[0])].translation.extend(
                                [self._in_compounds_pref + translation])
                    else:
                        for tr_r in tr[0]:
                            already_there = [e for e in extension if
                                             e.reading == [cr for cr in comp_readings[tr_r] if cr] and e.lexeme == [
                                                 kj.kanji]]
                            if not already_there:
                                extension.append(
                                    YarxiEntry(reading=[cr for cr in comp_readings[tr_r] if cr], lexeme=[kj.kanji],
                                               translation=[self._in_compounds_pref + tr[1]],
                                               eid=str(self._get_next_eid()), references=[],
                                               kanji=kj.kanji))
                            else:
                                extension[extension.index(already_there[0])].translation.extend(
                                    [self._in_compounds_pref + tr[1]])
            else:
                nom_val = re.search(r'~=(.+)', kj.rus)
                if nom_val is not None:
                    already_there = [e for e in extension if
                                     e.reading == [cr for cr in comp_readings['0'] if cr] and e.lexeme == [kj.kanji]]
                    if not already_there:
                        extension.append(YarxiEntry(reading=[cr for cr in comp_readings['0'] if cr], lexeme=[kj.kanji],
                                                    translation=[
                                                        self._in_compounds_pref + self._clean_text(nom_val.group(1))],
                                                    eid=str(self._get_next_eid()), references=[], kanji=kj.kanji))
                    else:
                        extension[extension.index(already_there[0])].translation.extend([
                            self._in_compounds_pref + self._clean_text(nom_val.group(1))])
                nom_val = re.search(r'~\$(.+)', kj.rus)
                if nom_val is not None:
                    already_there = [e for e in extension if
                                     e.reading == [cr for cr in comp_readings['0'] if cr] and e.lexeme == [kj.kanji]]
                    if not already_there:
                        extension.append(YarxiEntry(reading=[cr for cr in comp_readings['0'] if cr], lexeme=[kj.kanji],
                                                    translation=[self._in_compounds_pref + kj.rus_nick],
                                                    eid=str(self._get_next_eid()), references=[], kanji=kj.kanji))
                    else:
                        extension[extension.index(already_there[0])].translation.extend(
                            [self._in_compounds_pref + kj.rus_nick])

        for ext in tqdm(extension, desc="[Yarxi] Expanding word database".ljust(34), disable=not
        show_progress):
            already_there = [e for e in self._entries if
                             e.reading == ext.reading and e.lexeme == ext.lexeme]
            if not already_there:
                self._entries.append(ext)
            else:
                self._entries[self._entries.index(already_there[0])].translation.extend(ext.translation)

    def rescan(self, fname: str = "../dictionaries/yarxi_19.08.2020.db", show_progress: bool = True) -> YarxiDictionary:
        self._kanji_db = self._load_kanji_db(fname, show_progress)
        self._entries = self._load_db(fname, show_progress)
        self._resolve_references(show_progress)
        self._extract_compound_values(self._kanji_db, show_progress)

        return YarxiDictionary(self._entries)

    def load(self, fname: str = "default") -> YarxiDictionary:
        if fname == 'default':
            fname = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.joinpath('dictionaries/yarxi.jtdb')
        return pickle.load(open(fname, "rb"))
