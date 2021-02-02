import collections
import unittest
import time

from src.search import SearchMode
from src.yarxi import YarxiDictionary, YarxiLoader


class YarxiTests(unittest.TestCase):
    yd: YarxiDictionary = YarxiLoader().load()

    def test_garbage_numbers_in_lexeme_schema(self):
        self.assertEqual(self.yd.lookup_translations_only('悪貨は良貨を駆逐する', 'あっかはりょうかをくちくする'),
                         ['《эк.》 «худшие деньги вытесняют из обращения лучшие» (закон грешема)'])

    def test_simple_translation(self):
        self.assertEqual(self.yd.lookup_translations_only('亜塩素酸塩', 'あえんそさんえん'),
                         ['《хим.》 хлорит'])

    def test_multiple_wordforms_and_readings(self):
        self.assertEqual(self.yd.lookup_translations_only('青図面', 'あおずめん'),
                         self.yd.lookup_translations_only('青図', 'あおず'))

        # not really correct, but must work nonetheless
        self.assertEqual(self.yd.lookup_translations_only('青図', 'あおずめん'),
                         self.yd.lookup_translations_only('青図面', 'あおず'))

        self.assertEqual(self.yd.lookup_translations_only('赤鳥赤腹', 'あかこっこ'),
                         self.yd.lookup_translations_only('鳥赤腹', 'あかこっこ'))

        self.assertEqual(self.yd.lookup_translations_only('赤鳥赤腹', 'あかこっこ'),
                         self.yd.lookup_translations_only('鳥赤腹', 'あかこっこ'))

        self.assertEqual(self.yd.lookup_translations_only('匕首', 'あいくち'), self.yd.lookup_translations_only('匕首', 'ひしゅ'))

        self.assertEqual(self.yd.lookup_translations_only('エートス', 'ええとす'),
                         self.yd.lookup_translations_only('エトス', 'えとす'))

        self.assertEqual(self.yd.lookup_translations_only('クーリングオフ期間', 'くうりんぐおふきかん'),
                         self.yd.lookup_translations_only('クーリングオフ', 'くうりんぐおふ'))

        self.assertEqual(self.yd.lookup_translations_only('アイスクリームサンデー', 'あいすくりいむさんでえ'),
                         self.yd.lookup_translations_only('クリームサンデー', 'くりいむさんでえ'))

    def test_retrieving_references(self):
        self.assertEqual(self.yd.lookup_translations_only('蚶', 'あかがい'), self.yd.lookup_translations_only('赤貝', 'あかがい'))
        self.assertCountEqual(self.yd.lookup_translations_only('合口', 'あいくち'),
                         ['общий язык, возможность приятно побеседовать; приятный '
                          'собеседник', 'кинжал без гарды'])

    def test_translation(self):
        # complete synonyms
        self.assertEqual(self.yd.lookup_translations_only('漢民族', 'かんみんぞく'),
                         self.yd.lookup_translations_only('漢族', 'かんぞく'))
        self.assertEqual(self.yd.lookup_translations_only('亜ぐ', 'つぐ'), self.yd.lookup_translations_only('次ぐ', 'つぐ'))
        self.assertEqual(self.yd.lookup_translations_only('慣用句', 'かんようく'),
                         self.yd.lookup_translations_only('慣用語句', 'かんようごく'))
        self.assertEqual(self.yd.lookup_translations_only('懽楽', 'かんらく'), self.yd.lookup_translations_only('歓楽', 'かんらく'))
        self.assertEqual(self.yd.lookup_translations_only('有り得べき', 'ありうべき'),
                         self.yd.lookup_translations_only('有り得る', 'ありうる'))

        self.assertCountEqual(self.yd.lookup_translations_only('宇', 'のき'),
                         ['свисающий край крыши, стреха', '《перен.》 крыша над головой, '
                                                          'дом'])

        self.assertCountEqual(self.yd.lookup_translations_only('総角', 'あげまき'),
                         ['старинная прическа мальчика (собранные в петли волосы '
                          'крепились над ушами, образуя «рожки»)', 'женская прическа '
                                                                   'конца xix в.('
                                                                   'волосы '
                                                                   'закручивались в '
                                                                   'высокий узел и '
                                                                   'закреплялись '
                                                                   'булавками)',
                          'узел агэмаки, бант с тремя петлями и двумя концами',
                          'китайская ракушка-бритва, sinonovacula constricta'])
        self.assertEqual(self.yd.lookup_translations_only('総角', 'あげまき'), self.yd.lookup_translations_only('揚巻', 'あげまき'))

        # multiple meanings
        self.assertIn(self.yd.lookup_translations_only('アイブローペンシル', 'あいぶろうぺんしる')[0],
                      self.yd.lookup_translations_only('アイブロー', 'あいぶろう'))
        self.assertIn(self.yd.lookup_translations_only('眉', 'まゆ')[0],
                      self.yd.lookup_translations_only('アイブロー', 'あいぶろう'))

        self.assertEqual(self.yd.lookup_translations_only('鵺的', 'ぬえてき'), ['〈~na〉 таинственный, загадочный'])

        # literary variants are marked as such
        self.assertIn(self.yd.lookup_translations_only('飽きる', 'あきる')[0],
                      self.yd.lookup_translations_only('飽く', 'あく')[0])

        # obsolete variants are marked as such
        self.assertIn(self.yd.lookup_translations_only('侮る', 'あなどる')[0],
                      self.yd.lookup_translations_only('侮る', 'あなずる')[0])

        # different suggestions for different on'yomi
        self.assertCountEqual(self.yd.lookup_translations_only('音', 'おん'),
                              ['звук', '«он» (японизированное китайское чтение иероглифа)', '〈в сочет.〉 звук',
                               '〈в сочет.〉 музыкальный звук', '〈в сочет.〉 звук [как единица речи]',
                               '〈в сочет.〉 он (китайское чтение иероглифа)', '〈в сочет.〉 весть'])
        self.assertCountEqual(self.yd.lookup_translations_only('音', 'いん'),
                         ['〈в сочет.〉 звук [как единица речи]', '〈в сочет.〉 он (китайское чтение иероглифа)',
                          '〈в сочет.〉 весть'])

        # entry with erroneously put numeration
        self.assertCountEqual(self.yd.lookup_translations_only('紙垂', 'しで'),
                         ['《синт.》 зигзагообразная бумажная лента (исп. в различных '
                          'ритуалах)'])

        self.assertEqual(self.yd.lookup_translations_only('春眠暁を覚えず'),
                         ['《посл.》 весной спишь так, как будто рассвет никогда не наступит'])
        self.assertCountEqual(self.yd.lookup_translations_only('暁', 'あかつき') +
                              self.yd.lookup_translations_only('暁', 'ぎょう'),
                              self.yd.lookup_translations_only('暁'))

        self.assertEqual(self.yd.lookup_translations_only('椦'), ['《бот.》 сумах (от «nurude»)'])

    def test_general(self):
        # all entries contain at least one reading
        self.assertEqual(len([e for e in self.yd._entries if e.reading == []]), 0)

        # all entry ids are distinct
        self.assertEqual(
            len([item for item, count in collections.Counter([e.eid for e in self.yd._entries]).items() if count > 1]),
            0)

        # every entry has a translation or a reference to another entry
        self.assertEqual(len([e for e in self.yd._entries if not e.references and not e.translation]), 0)

        # mma -> nma
        self.assertNotEqual(self.yd.lookup('ガンマ線', reading='がんません'), [])
        self.assertNotEqual(self.yd.lookup('干満の差', reading='かんまんのさ'), [])

        # no unnecessary quotes; unneccessary whitespaces are stripped from lexemes in references
        self.assertEqual(self.yd.lookup_translations_only('感無量', 'かんむりょう'),
                         ['безмерность (невыразимость) чувств; 〈~ni naru〉, 〈~de '
                          'aru〉 быть переполненным чувствами'])
        self.assertEqual(self.yd.lookup_translations_only('寒夜', 'かんや'), ['《кн.》 холодная (зимняя) ночь'])
        self.assertEqual(self.yd.lookup_translations_only('紡ぎ', 'つむぎ'), ['шелковая ткань типа чесучи'])
        self.assertCountEqual(self.yd.lookup_translations_only('阿', 'あ'),
                         ['«а», 《первая буква санскритского алфавита》',
                          '〈в сочет.〉 льстить, угодничать',
                          '〈в сочет.〉 《употребляется фонетически》',
                          '〈в сочет.〉 《сокр.》 африка'])

        # no exclamation marks in the beginning of line
        self.assertEqual(self.yd.lookup_translations_only('紡ぎ歌', 'つむぎうた'), ['песня, исполняемая в процессе прядения'])

        # removed short 'i' marks
        self.assertNotIn('q1', self.yd.lookup('甲斐甲斐しい', 'かいがいしい')[0].reading)
        self.assertNotIn('q2', self.yd.lookup('甲斐甲斐しい', 'かいがいしい')[0].reading)

        # removed 'usually in katakana' marks
        self.assertEqual(self.yd.lookup_translations_only('伯林', 'べるりん'), ['берлин'])
        self.assertEqual(self.yd.lookup_translations_only('越南', 'べとなむ'), ['вьетнам'])
        self.assertEqual(self.yd.lookup_translations_only('越南', 'えつなん'), ['вьетнам'])

        # correct lexeme when both hiragana and katakana are present
        self.assertEqual(self.yd.lookup_translations_only('聖エルモの火'), ['огни святого эльма'])

        # nested translation
        for tr in self.yd.lookup_translations_only('ちょっかいを出す'):
            self.assertIn(tr, self.yd.lookup_translations_only('ちょっかい'))

    def test_likeliness_score(self):
        self.assertEqual(self.yd.lookup_translations_only('聖エルモ'), ['огни святого эльма'])
        self.assertEqual(self.yd.lookup_translations_only('乱麻'), ['решить проблему быстро и решительно, разрубить гордиев узел'])

    def test_double_h(self):
        self.assertEqual(self.yd.lookup_translations_only('暁には', 'あかつきには'), ['когда; в случае'])
        self.assertEqual(self.yd.lookup_translations_only('バッハ', 'ばっは'),
                         ['[иоган себастьян] бах (немецкий композитор,1685-1750 гг.; '
                          'нем. 《johann sebastian bach》)'])

    def test_no_empty_translations(self):
        # for entry in tqdm(self.yd._entries, desc='Checking that translations are never empty'):
        #     if not self.yd.translate(entry.reading[0], entry.lexeme[0]):
        #         print(entry)

        # edits made in order to ensure that translation is never empty:

        # 合せる -> 合わせる
        self.assertCountEqual(self.yd.lookup_translations_only('勠せる', 'あわせる'),
                         ['[при] соединять', 'согласовывать', 'сличать, сопоставлять', 'подсекать (рыбу)'])

        # 裡 -> edited の裡に at tango table into just 裡
        self.assertEqual(self.yd.lookup_translations_only('裏', 'うち'), ['〈-no〉 〈~ni〉 в (условиях чего-л.)'])
        self.assertEqual(self.yd.lookup_translations_only('裡', 'うち'), self.yd.lookup_translations_only('裏', 'うち'))

        # 逆う -> 逆らう
        self.assertEqual(self.yd.lookup_translations_only('忤う', 'さからう'), ['идти против 《чего-л.》'])

        # 華しい -> 華々しい
        self.assertEqual(self.yd.lookup_translations_only('花々しい', 'はなばなしい'), ['〈~na〉 яркий, цветущий, прекрасный'])

        # 捻くれる -> 捻れる
        self.assertEqual(self.yd.lookup_translations_only('拈くれる', 'ひねくれる'), ['быть искривленным; быть замысловатым; '
                                                                             'быть извращенным'])

        # 芽む -> 芽ぐむ
        self.assertEqual(self.yd.lookup_translations_only('萌む', 'めぐむ'), ['давать почки; распускаться'])

        # 別かれ -> 別れ
        self.assertCountEqual(self.yd.lookup_translations_only('分かれ', 'わかれ'),
                         ['отделение, ответвление; развилка', 'расставание, разлука'])

        # ちじみあがる -> ちぢみあがる
        self.assertCountEqual(self.yd.lookup_translations_only('縮み上がる', 'ちぢみあがる'), ['сжиматься; съеживаться'])

    def test_matching_lexemes(self):
        self.assertEqual(self.yd.lookup_translations_only('繰り返す', 'くりかえす'), self.yd.lookup_translations_only('繰返す', 'くりかえす'))
        self.assertEqual(self.yd.lookup_translations_only('相打ち', 'あいうち'), self.yd.lookup_translations_only('相打', 'あいうち'))
        self.assertEqual(self.yd.lookup_translations_only('空地', 'あきち'), self.yd.lookup_translations_only('空き地', 'あきち'))
        self.assertEqual(self.yd.lookup_translations_only('お前', 'おまえ'), ['《грубо》 ты'])
        self.assertCountEqual(self.yd.lookup_translations_only('前', 'まえ'), ['перед; 〈~ni〉 впереди; 〈~no〉 передний', 'в '
                                                                                                               '《чьем-л.》 присутствии, перед 《кем-л.》',
                                                                       '〈~ni〉 раньше; 〈~no〉 прежний, прошлый',
                                                                       '〈~ni〉 заранее', 'гениталии',
                                                                       '〈в сочет.〉 передний; впереди; перед 《чем-л.》',
                                                                       '〈в сочет.〉 раньше 《чего-л.》',
                                                                       '〈в сочет.〉 заранее', '〈в сочет.〉 порция; доля',
                                                                       '〈в сочет.〉 《суффикс после имен придворных дам》',
                                                                       '〈в сочет.〉 《идиоматические сочетания》'])

        # print(self.yd.lookup_translations_only('前', 'まえ'), self.yd.lookup_translations_only('前', 'ぜん'))

    def test_search_modes(self):
        # しまう is not explicitly listed in the dictionary, but 仕舞う is marked with 'usually in hiragana'
        # so its hiragana spelling is also added to its list of lexemes
        self.assertEqual(self.yd.lookup_translations_only('仕舞う', 'しまう'), self.yd.lookup_translations_only('しまう', 'しまう'))
        self.assertEqual(self.yd.lookup_translations_only('終う', 'しまう'), self.yd.lookup_translations_only('了う', 'しまう'))
        self.assertEqual(self.yd.lookup_translations_only('了う', 'しまう'), self.yd.lookup_translations_only('しまう', 'しまう'))

        # 'ra' is missing from okurigana here, but deep search allows to retrieve translation nevertheless
        self.assertEqual(self.yd.lookup_translations_only('逆う', 'さからう', search_mode=SearchMode.shallow_only), [])
        self.assertEqual(self.yd.lookup_translations_only('逆う', 'さからう', search_mode=SearchMode.deep_only), ['идти против 《чего-л.》'])
        self.assertEqual(self.yd.lookup_translations_only('逆う', 'さからう', search_mode=SearchMode.consecutive), ['идти против 《чего-л.》'])

    def test_orders(self):
        # order defines the degree of likeliness that the retrieved entry describes the required word
        # order = 1 (default): the candidate(s) with the topmost probability score
        self.assertCountEqual(self.yd.lookup_translations_only('繰返す', order=1), ['повторять, делать снова'])

        # order = 2: the candidates with the two topmost probability scores
        self.assertCountEqual(self.yd.lookup_translations_only('繰返す', order=2), ['рефрен, припев', 'повторение', 'повторять, делать снова'])

    def test_exceptions(self):
        # bad lexeme
        with self.assertRaises(Exception):
            self.yd.lookup_translations_only('一1')

        # bad reading (must be exclusively hiragana)
        with self.assertRaises(Exception):
            self.yd.lookup_translations_only('一', 'ichi')

        # bad order value
        with self.assertRaises(Exception):
            self.yd.lookup_translations_only('一', 'いち', order=0)

        # bad search mode value
        with self.assertRaises(Exception):
            self.yd.lookup_translations_only('一', 'いち', search_mode=SearchMode(4))


    # Time tests:
    #
    # Average time spent in shallow-only search mode: 0.0054999589920043945
    # Average time spent in consecutive search mode (not involving deep search): 0.00520021915435791
    #
    # Average time spent in deep-only search mode: 1.5742049932479858
    # Average time spent in consecutive search mode (involving deep search): 1.5545949935913086
    #
    # Conclusion:
    # consecutive search mode is the best of the three
    # or highly comparable to shallow/deep-only in their respective cases.
    #
    # def test_measure_time(self):
    #     start = time.time()
    #     self.yd.lookup_translations_only('繰り返す', search_mode=SearchMode.shallow_only)
    #     self.yd.lookup_translations_only('相打ち', search_mode=SearchMode.shallow_only)
    #     self.yd.lookup_translations_only('空き地', search_mode=SearchMode.shallow_only)
    #     end = time.time()
    #     print(f'Average time spent in shallow-only search mode: {(end - start) / 10}')
    #
    #     start = time.time()
    #     self.yd.lookup_translations_only('繰り返す', search_mode=SearchMode.consecutive)
    #     self.yd.lookup_translations_only('相打ち', search_mode=SearchMode.consecutive)
    #     self.yd.lookup_translations_only('空き地', search_mode=SearchMode.consecutive)
    #     end = time.time()
    #     print(f'Average time spent in consecutive search mode (not involving deep search): {(end - start) / 10}')
    #
    #     start = time.time()
    #     self.yd.lookup_translations_only('繰り返す', search_mode=SearchMode.deep_only)
    #     self.yd.lookup_translations_only('相打ち', search_mode=SearchMode.deep_only)
    #     self.yd.lookup_translations_only('空き地', search_mode=SearchMode.deep_only)
    #     end = time.time()
    #     print(f'Average time spent in deep-only search mode: {(end - start) / 10}')
    #
    #     start = time.time()
    #     self.yd.lookup_translations_only('繰返す')
    #     self.yd.lookup_translations_only('相打')
    #     self.yd.lookup_translations_only('空地')
    #     end = time.time()
    #     print(f'Average time spent in consecutive search mode (involving deep search): {(end - start) / 10}')


if __name__ == '__main__':
    unittest.main()
