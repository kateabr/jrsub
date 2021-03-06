import collections
import re
import unittest

from jrsub import SearchMode, YarxiDictionary, YarxiLoader
from jrsub.utils import _is_hiragana


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
                                                                        'конца xix в. ('
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
                              ['〈в сочет.〉 звук [как единица речи]', '〈в сочет.〉 он (китайское чтение иероглифа)',
                               '〈в сочет.〉 звук', 'звук', '〈в сочет.〉 весть', '〈в сочет.〉 музыкальный звук',
                               '«он» (японизированное китайское чтение иероглифа)'])
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

        for e in [e for e in self.yd._entries if not e.references and not e.translation]:
            print(e)

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

        self.assertEqual(self.yd.lookup_translations_only('淫りがわしい', 'みだりがわしい'),
                         self.yd.lookup_translations_only('淫ら', 'みだら'))

        # general translations with complicated markdown
        self.assertEqual(self.yd.lookup_translations_only('加圧', 'かあつ'),
                         ['подача под давлением, нагнетание; 〈~suru〉 подавать под давлением, нагнетать; 〈~shite〉,'
                          ' 〈[~no]〉 под давлением'])

        self.assertCountEqual(self.yd.lookup_translations_only('結構', 'けっこう'),
                              ['структура, композиция, построение, архитектоника',
                               'вполне, довольно, весьма; 〈[~desu]〉 достаточно, хватит, больше не нужно, спасибо',
                               '〈~na〉 прекрасный, превосходный, великолепный; 〈~desu〉 хорошо!, '
                               'прекрасно!, замечательно!'])

        self.assertEqual(self.yd.lookup_translations_only('重畳', 'ちょうじょう'),
                         ['〈~de aru〉 громоздиться; 〈~desu!〉 превосходно!, прекрасно!'])

        self.assertEqual(self.yd.lookup_translations_only('どうやって', 'どうやって'),
                         ['как?, каким образом?'])

        self.assertEqual(self.yd.lookup_translations_only('何方', 'どなた'),
                         ['《вежл.》 кто?'])

        self.assertEqual(self.yd.lookup_translations_only('儲かりまっか', 'もうかりまっか'),
                         ['«как заработки?», стандартное приветствие в осакском диалекте, искаженное  «mo:karimas ka»'])

        self.assertCountEqual(self.yd.lookup_translations_only('出だす', 'いだす'),
                              ['《уст.》 помещать (в печати)', '《уст.》 посылать, отправлять',
                               '《уст.》 выставлять; высовывать; вынимать; вытаскивать', '《уст.》 давать, выдавать',
                               '《уст.》 демонстрировать; предъявлять'])

        self.assertCountEqual(self.yd.lookup_translations_only('赤裸', 'あかはだか'),
                              ['〈~no〉 совершенно голый; 〈~de〉 голым', '〈~no〉 абсолютный голый'])

        self.assertEqual(self.yd.lookup_translations_only('葫蘆', 'ころ'),
                         ['《редк.》 тыква-горлянка'])

        self.assertCountEqual(self.yd.lookup_translations_only('アウトコース'),
                              ['внешняя беговая дорожка (стадиона и т. п.)',
                               'внешняя сторона, наружная часть (англ. 《outside》); 《в сочетаниях》 наружный, снаружи',
                               '《бейсбол》 внешнее пространство (для броска; англ. 《out course》)'])

        self.assertCountEqual(self.yd.lookup_translations_only('一寸', 'ちょっと'),
                              ['〈~shita〉 маленький, слабый, пустяковый', '〈~shita〉 приличный, порядочный, неплохой',
                               '《с отриц.》 никак, нелегко, вряд ли', 'немножко, чуточку; на минутку',
                               '《обращение》 эй! послушайте!'])

        self.assertEqual(self.yd.lookup_translations_only('曳々', 'えいえい'),
                         ['〈~to〉 (тянуть с выкриком) «взяли!», «дружно!», «еще раз!»'])

        self.assertCountEqual(self.yd.lookup_translations_only('合点', 'がてん'),
                              ['〈~suru〉 соглашаться; кивать; 〈~ka〉 согласен?, нет возражений?',
                               'понимание; 〈~suru〉 понимать'])

        self.assertCountEqual(self.yd.lookup_translations_only('石', 'こく'),
                              ['мера емкости [риса] (180 л, 10 то, 150 кг риса)',
                               'мера объема [пиломатериалов] (0,278 куб. м., 10 куб. сяку)'])

        # there are two entries in which both lexeme and reading equal よう: [よう]:[よう] and [よう, 様]:[よう]
        self.assertCountEqual(self.yd.lookup_translations_only('よう', 'よう'),
                              ['《частица》 〈~na〉 такой как, похожий, вроде; 〈~ni〉, '
                               '〈~desu〉 [так] как, подобно; будто; по-видимому, похоже; '
                               '〈[~ni]〉 [так] чтобы; 〈~ni naru〉 《после глагола показывает, '
                               'что действие начало совершаться》',
                               '《частица》 《в начале》 браво!, здорово!, молодцы!',
                               '《частица》 《заключительная восклицательная частица》',
                               '〈~na〉 такой как, похожий, вроде; 〈~ni〉, 〈~desu〉 [так] как, подобно; '
                               'будто; по-видимому, похоже; 〈[~ni]〉 [так] чтобы; 〈~ni naru〉 '
                               '《после глагола показывает, что действие начало совершаться》',
                               '《частица》 《в конце》 ну пожалуйста!, ну же!'])

        self.assertCountEqual(self.yd.lookup_translations_only('どっこい'),
                              ['《прост. межд.》 стой!, погоди!',
                               '《прост. межд.》 взяли!; 〈~to〉 с большим трудом (поднимать); устало (плюхнуться, повалиться)'])

        self.assertCountEqual(self.yd.lookup_translations_only('まあまあ'),
                              [
                                  '〈[~no]〉 неплохой, годный, удовлетворительный, более-менее; 〈~da〉 нормально, сойдет, ничего',
                                  'полноте, довольно, ладно уже', '《межд.》 ох!, ах!'])

        self.assertCountEqual(self.yd.lookup_translations_only('ヤード', 'やあど'),
                              ['ярд (91,44 см)', 'площадка для хранения, склад (англ. 《yard》)'])

        # some compound translations
        self.assertCountEqual(self.yd.lookup_translations_only('尭', 'ぎょう'),
                              ['〈в сочет.〉 император яо',
                               'яо, мифический китайский император, четвертый из пяти совершенномудрых государей древности (2356-2258 до н. э.)'])

        self.assertCountEqual(self.yd.lookup_translations_only('諄', 'じゅん'),
                              ['〈в сочет.〉 назойливый', '《мужское имя》'])

        self.assertEqual(self.yd.lookup_translations_only('吾', 'ご'),
                         self.yd.lookup_translations_only('吾', 'あが'))

        self.assertEqual(self.yd.lookup_translations_only('垢', 'こう'), ['〈в сочет.〉 грязь'])

        self.assertEqual(self.yd.lookup_translations_only('明け六つ', 'あけむつ'), ['《уст.》 шесть часов утра'])

    def test_likeliness_score(self):
        self.assertEqual(self.yd.lookup_translations_only('聖エルモ'), ['огни святого эльма'])
        self.assertEqual(self.yd.lookup_translations_only('乱麻'),
                         ['решить проблему быстро и решительно, разрубить гордиев узел'])

    def test_double_h(self):
        self.assertEqual(self.yd.lookup_translations_only('暁には', 'あかつきには'), ['когда; в случае'])
        self.assertEqual(self.yd.lookup_translations_only('バッハ', 'ばっは'),
                         ['[иоган себастьян] бах (немецкий композитор, 1685-1750 гг.; '
                          'нем. 《johann sebastian bach》)'])

    def test_no_empty_translations(self):
        # edits made in order to ensure that translation is never empty:

        # 合せる -> 合わせる
        self.assertCountEqual(self.yd.lookup_translations_only('勠せる', 'あわせる'),
                              ['[при] соединять', 'согласовывать', 'сличать, сопоставлять', 'подсекать (рыбу)'])

        # 裡 -> edited の裡に at tango table into just 裡
        self.assertEqual(self.yd.lookup_translations_only('裏', 'うち'), ['〈-no〉＿〈~ni〉 в (условиях чего-л.)'])
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

        # other edits

        # ちじみあがる -> ちぢみあがる
        self.assertCountEqual(self.yd.lookup_translations_only('縮み上がる', 'ちぢみあがる'), ['сжиматься; съеживаться'])

        # いんなーわあ -> いんなーうぇあ
        self.assertCountEqual(self.yd.lookup_translations_only(lexeme='インナーウェア', reading='いんなあうぇあ'),
                              ['нательное (нижнее) белье (англ. 《innerware》)', 'нижнее белье'])

        # дюйм\(\англ.#\inchi\#) -> дюйм\(\англ.#\inch\#)
        self.assertEqual(self.yd.lookup_translations_only('吋', 'いんち'), ['дюйм (англ. 《inchi》)'])

        # <<кино>> ср -> <<кино>> см
        self.assertCountEqual(self.yd.lookup_translations_only('アップ'),
                              ['повышение; улучшение (от англ. 《up》); 〈~suru〉 повышать [ся]; улучшать [ся]',
                               '《фото》 《перен.》 повышенное внимание; 〈~suru〉 привлекать внимание 《к чему-л.》; 〈~sareru〉 быть в центре внимания, выходить на передний план',
                               '《тех.》 кривошип (англ. 《crank》)', 'рукоятка (особ. кинокамеры старого образца)',
                               '《комп.》 загрузка, заливка (англ. 《upload》); 〈~suru〉 загружать',
                               'зачесанные кверху волосы (англ. 《up style》)',
                               '《фото》 《фото, кино》 крупный план (англ. 《close-up》); 〈~de〉 крупным планом; 〈~suru〉 снимать крупным планом; 〈~sareru〉 идти крупным планом',
                               '《кино》 завершение съемки кинофильма (англ. 《crank up》; букв. «рукоятку вверх»)',
                               '《муз.》 кранк (стиль хип-хопа; англ. 《crunk》)'])

        # кольцо\,,\круг\;*3\кольцевидный\,\круглый -> кольцо\,\круг\;*3\кольцевидный\,\круглый
        self.assertEqual(self.yd.lookup_translations_only('輪形', 'わがた'),
                         ['кольцо, круг; 〈~no〉 кольцевидный, круглый'])

        # とうる -> とおる
        self.assertEqual(self.yd.lookup_translations_only('透る', 'とおる'),
                         ['проникать, проходить сквозь'])
        self.assertCountEqual(self.yd.lookup_translations_only('亮', 'とおる'),
                              ['《мужское имя》', '〈в сочет.〉 ясный, очевидный'])

    def test_matching_lexemes(self):
        self.assertEqual(self.yd.lookup_translations_only('繰り返す', 'くりかえす'),
                         self.yd.lookup_translations_only('繰返す', 'くりかえす'))
        self.assertEqual(self.yd.lookup_translations_only('相打ち', 'あいうち'),
                         self.yd.lookup_translations_only('相打', 'あいうち'))
        self.assertEqual(self.yd.lookup_translations_only('空地', 'あきち'), self.yd.lookup_translations_only('空き地', 'あきち'))
        self.assertEqual(self.yd.lookup_translations_only('お前', 'おまえ'), ['《грубо》 ты'])
        self.assertCountEqual(self.yd.lookup_translations_only('前', 'まえ'),
                              ['〈в сочет.〉 порция; доля', '〈~ni〉 заранее', '〈в сочет.〉 предшествующий', 'гениталии',
                               '〈в сочет.〉 заранее', 'перед; 〈~ni〉 впереди; 〈~no〉 передний',
                               '〈в сочет.〉 《идиоматические сочетания》', '〈~ni〉 раньше; 〈~no〉 прежний, прошлый',
                               '〈в сочет.〉 передний; впереди; перед 《чем-л.》', '〈в сочет.〉 раньше 《чего-л.》',
                               '〈в сочет.〉 《суффикс после имен придворных дам》',
                               'в 《чьем-л.》 присутствии, перед 《кем-л.》'])
        self.assertCountEqual(self.yd.lookup_translations_only('前', 'ぜん'),
                              ['〈в сочет.〉 порция; доля', '〈в сочет.〉 предшествующий', '〈в сочет.〉 заранее',
                               '〈в сочет.〉 《идиоматические сочетания》', '〈в сочет.〉 раньше 《чего-л.》',
                               '〈в сочет.〉 передний; впереди; перед 《чем-л.》',
                               '〈в сочет.〉 《суффикс после имен придворных дам》'])

        # print(self.yd.lookup_translations_only('前', 'まえ'), self.yd.lookup_translations_only('前', 'ぜん'))

    def test_search_modes(self):
        # しまう is not explicitly listed in the dictionary, but 仕舞う is marked with 'usually in hiragana'
        # so its hiragana spelling is also added to its list of lexemes
        self.assertEqual(self.yd.lookup_translations_only('仕舞う', 'しまう'), self.yd.lookup_translations_only('しまう', 'しまう'))
        self.assertEqual(self.yd.lookup_translations_only('終う', 'しまう'), self.yd.lookup_translations_only('了う', 'しまう'))
        self.assertEqual(self.yd.lookup_translations_only('了う', 'しまう'), self.yd.lookup_translations_only('しまう', 'しまう'))

        # 'ra' is missing from okurigana here, but deep search allows to retrieve translation nevertheless
        self.assertEqual(self.yd.lookup_translations_only('逆う', 'さからう', search_mode=SearchMode.shallow_only), [])
        self.assertEqual(self.yd.lookup_translations_only('逆う', 'さからう', search_mode=SearchMode.deep_only),
                         ['идти против 《чего-л.》'])
        self.assertEqual(self.yd.lookup_translations_only('逆う', 'さからう', search_mode=SearchMode.consecutive),
                         ['идти против 《чего-л.》'])

    def test_orders(self):
        # order defines the degree of likeliness that the retrieved entry describes the required word
        # order = 1 (default): the candidate(s) with the topmost probability score
        self.assertCountEqual(self.yd.lookup_translations_only('繰返す', order=1), ['повторять, делать снова'])

        # order = 2: the candidates with the two topmost probability scores
        self.assertCountEqual(self.yd.lookup_translations_only('繰返す', order=2),
                              ['рефрен, припев', 'повторение', 'повторять, делать снова'])

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

    def test_bad_readings(self):
        bad = []
        for e in self.yd._entries:
            for rd in e.reading:
                if any(not _is_hiragana(ch) for ch in rd):
                    bad.append(e)
        self.assertEqual(bad, [])

        for e in self.yd._entries:
            for tr in e.translation:
                if re.search(r'\(\d+\)', tr) is not None:
                    bad.append(e)
        self.assertEqual(bad, [])

    def test_all_refs_verified(self):
        us_nv = []
        for entry in self.yd._entries:
            for ref in entry.references:
                if ref.usable and not ref.verified:
                    us_nv.append(entry)

        self.assertEqual(us_nv, [])


if __name__ == '__main__':
    unittest.main()
