import unittest

from jrsub.utils import _is_hiragana
from jrsub import WarodaiDictionary, WarodaiLoader


class MyTestCase(unittest.TestCase):
    wd: WarodaiDictionary = WarodaiLoader().load()

    # def test_ending_punct(self):
    #     self.assertEqual(self.wd.lookup_translations_only('相争う', 'あいあらそう'), ['спорить (ссориться) друг с другом'])
    #     self.assertEqual(self.wd.lookup_translations_only('愛', 'あい'), ['любовь'])
    #     self.assertEqual(self.wd.lookup_translations_only('ああ', 'ああ'), ['《при обращении》', '《ответное》', 'так', '《межд.》 ах!'])

    def test_readings_lexemes(self):
        # self.assertCountEqual(self.wd.lookup('嗚呼')[0].lexeme, ['ああ', '嗚呼'])
        # self.assertCountEqual(self.wd.lookup('藹藹')[0].lexeme, ['藹々', '藹藹'])
        #
        # print(self.wd.lookup('青大将'))
        # print(self.wd.lookup('黄頷蛇'))
        self.assertEqual(self.wd.lookup_translations_only('鳳輦'),
                         ['《кн.》 колесница с золотой птицей «хо» на крыше', '《кн.》 императорский экипаж'])
        self.assertEqual(self.wd.lookup_translations_only('北陸道'),
                         [
                             '《ист. область》 хокурикудо 《включала пров. вакаса, кага, ното, этидзэн, эттю, этиго, садо》'])
        self.assertEqual(self.wd.lookup_translations_only('明六つ'),
                         ['《ист.》 шесть часов утра', '《ист.》 шестичасовой утренний колокол'])
        self.assertEqual(self.wd.lookup_translations_only('星合い'),
                         [
                             '«встреча звезд» 《праздник, отмечаемый 7 числа 7 месяца по лунному календарю в честь '
                             'встречи двух влюбленных, согласно легенде о звездах «волопасе» и «ткачихе»》'])
        self.assertEqual(self.wd.lookup_translations_only('靄々'),
                         ['〈~taru〉 《кн.》 стелющийся 《о тумане, облаках》',
                          '〈~taru〉 《кн.》 густой, пышный 《напр. о растительности》',
                          '〈~taru〉 《кн.》 мирный, спокойный; безмятежный'])
        self.assertEqual(self.wd.lookup_translations_only('補助'),
                         ['помощь, поддержка; 〈~suru〉 оказывать помощь (поддержку)',
                          'дополнение, пополнение; 〈[~no]〉 дополнительный; вспомогательный; запасной',
                          'пособие, субсидия; дотация'])

        # for entry in [e for e in self.wd._entries if '1' in e.translation.keys() and len(e.translation['1']) == 1 and re.search(r'^《[^》]+》$', e.translation['1'][0])]:
        #     print(entry.eid, entry.translation['1'][0])

        self.assertEqual(self.wd.lookup_translations_only('約する'), ['сокращать 《дробь》', 'договориться, заключить '
                                                                                        'договор (контракт)',
                                                                   'обещать', 'сокращать; опускать', 'захватывать'])
        self.assertEqual(self.wd.lookup_translations_only('旅慣れる'), ['＿≪-ta≫ привыкший путешествовать', '＿≪-nai≫ '
                                                                                                       'непривычный к '
                                                                                                       'путешествиям'])
        self.assertEqual(self.wd.lookup_translations_only('坊主'), ['《буд.》 монах; 《обр.》 бритоголовый', '〈~ni naru〉 '
                                                                                                       'постричься [в'
                                                                                                       ' монахи]',
                                                                  '〈~ni naru〉, 〈~ni karu 【刈る】〉 《обр.》 коротко '
                                                                  'остричься, обрить голову'])
        self.assertEqual(self.wd.lookup_translations_only('干し海苔'),
                         ['нори 《тонкие листочки сушеной водоросли tenera, употребляются как приправа》'])
        self.assertEqual(self.wd.lookup_translations_only('乾し物'), ['что-либо высушенное на солнце'])
        self.assertEqual(self.wd.lookup_translations_only('ほじくり出す'),
                         ['《прост.》 копать, раскапывать; ковырять 《напр. в зубах, в носу》',
                          '《прост.》 раскапывать, допытываться'])
        self.assertEqual(self.wd.lookup_translations_only('保する'), ['《кн.》 гарантировать, обеспечивать'])
        self.assertEqual(self.wd.lookup_translations_only('細腕'),
                         ['《прост.》 тонкие (слабые) руки', '《прост. перен.》 неумение, неискусность'])
        self.assertEqual(self.wd.lookup_translations_only('布袋'),
                         ['хотэй 《бог изобилия с большим животом и мешком на спине》'])
        self.assertEqual(self.wd.lookup_translations_only('程'),
                         ['мера, границы 《тк. в устойчивых выражениях》; 〈~ga aru〉 есть границы '
                          '《чему-л., в чем-л.》',
                          '《вспомогательная частица после обозначения количества, размера, расстояния, '
                          'времени》',
                          '《послелог》; настолько, такой, как; так же, как; настолько (так), что…; 《в '
                          'обороте речи》 чем…, тем…'])
        self.assertEqual(self.wd.lookup_translations_only('ほんこ'), ['《прост.》 〈~de menko wo suru〉 играть в картинки '
                                                                   '《детская игра》'])
        self.assertEqual(self.wd.lookup_translations_only('嫌味'),
                         ['〈~no aru〉 неприятный; колкий; безвкусный', '〈~no nai〉 приятный; со вкусом'])
        self.assertEqual(self.wd.lookup_translations_only(lexeme='相', reading='あい'), ['взаимно, друг друга, обоюдно; '
                                                                                      'вместе'])
        self.assertEqual(self.wd.lookup_translations_only('青黄紛'), ['《смесь двух порошков》'])
        self.assertEqual(self.wd.lookup_translations_only('五つ時'), ['《ист.》 «пять часов» 《время от 8 до 10 часов '
                                                                   'вечера и утра》'])
        self.assertEqual(self.wd.lookup_translations_only('苺'), ['земляника, клубника', 'малина'])
        self.assertEqual(self.wd.lookup_translations_only('莓'), ['малина'])
        self.assertEqual(self.wd.lookup_translations_only('一年生'), ['первокурсник, первоклассник; 《обр.》 начинающий, '
                                                                   'новичок', '〈[~no]〉 однолетний 《о растениях》'])
        self.assertEqual(self.wd.lookup_translations_only('原水爆'), ['(《сокр.》 genshi 【原子】 bakudan 【爆弾】 + suiso 【水素】 '
                                                                   'bakudan 【爆弾】) атомная и водородная бомбы'])
        self.assertEqual(self.wd.lookup_translations_only('浮上げ彫'), ['барельеф, горельеф'])
        self.assertCountEqual(self.wd.lookup_translations_only('未だ'), ['《кн.》 еще [не]',
                                                                       'с 《отриц.》 еще не',
                                                                       '[все] еще',
                                                                       'еще, кроме того',
                                                                       'еще [только]'])

        self.assertEqual(self.wd.lookup_translations_only('新玉'), ['〈~no〉 《поэтическое определение к таким словам, '
                                                                  'как год, месяц и т. п.》', 'неограненный '
                                                                                             'драгоценный камень'])
        self.assertEqual(self.wd.lookup_translations_only('一元'), ['〈~no〉, 〈~teki 【的】〉 унифицированный; '
                                                                  'централизованный; 《филос.》 монистический',
                                                                  'первый (начальный) год 《годов правления》',
                                                                  'один юань'])
        self.assertEqual(self.wd.lookup_translations_only('一次'), ['〈~no〉 первый, первичный 《о повторяющихся явлениях》',
                                                                  '〈~no〉 《мат.》 первой степени'])
        self.assertEqual(self.wd.lookup_translations_only('正札付き'), ['〈~no〉 без запроса', '〈~no〉 《перен.》 отъявленный; '
                                                                                         'заведомый 《напр. о '
                                                                                         'мошеннике》'])
        self.assertEqual(self.wd.lookup_translations_only('入府'), ['〈~suru〉 въезжать в округ, въезжать в город',
                                                                  '〈~suru〉 《ист.》 въезжать в свое владение'])
        self.assertEqual(self.wd.lookup_translations_only('予算外'), ['〈~no〉 не предусмотренный в бюджете, не включенный '
                                                                   'в бюджет, внесметный', '〈~no〉 непредвиденный'])
        self.assertEqual(self.wd.lookup_translations_only('猥雑'), ['〈~na〉 неряшливый 《гл. обр. в одежде》',
                                                                  '〈~na〉 грубый, вульгарный, низкопробный'])

        self.assertCountEqual(self.wd.lookup_translations_only('側'), ['сторона, бок', 'ряд, шеренга',
                                                                      '《связ.》 〈~[no mono 【者】]〉 окружающие',
                                                                      'сторона 《в переговорах, конфликтах и т. п.; '
                                                                      'часто не переводится》', 'сторона',
                                                                      '…〈-[no]〉＿〈~ni〉 возле 《кого-чего-л.》, '
                                                                      'рядом с 《кем-чем-л.》; поблизости [от '
                                                                      '《кого-чего-л.》]', '《редко》 сторона',
                                                                      '《редко》 〈~kara〉 со стороны',
                                                                      'боковая сторона, бок; 〈~ni〉 сбоку, около, '
                                                                      'подле, возле, рядом; 〈~kara〉 сбоку'])
        self.assertCountEqual(self.wd.lookup_translations_only('火照らす'),
                              ['《связ.》 〈kao 【顔】 wo〉＿≪-shite≫ с пылающим лицом',
                               '《связ.》 〈makka ni kao 【顔】 wo〉＿≪-sete≫ раскрасневшись',
                               '《связ.》 〈hoho 【頰】 wo〉＿ щеки пылают, лицо залила краска'])
        self.assertCountEqual(self.wd.lookup_translations_only('骨節'), ['сустав', 'суставы'])
        self.assertCountEqual(self.wd.lookup_translations_only('阿彌陀'), ['(《санскр.》 amitābhā) будда амида (амитаба)',
                                                                        '《прост.》 «лотерея амиды» 《особый вид лотереи '
                                                                        'для покупки вещей в складчину》'])
        self.assertCountEqual(self.wd.lookup_translations_only('後生大事'), ['〈~ni〉 с религиозным рвением',
                                                                         '〈~ni〉 с величайшей заботой, весьма '
                                                                         'тщательно; ревностно, преданно; 〈~ni suru〉 '
                                                                         'беречь как зеницу ока'])
        self.assertEqual(self.wd.lookup_translations_only('裏書譲渡'),
                         ['〈~suru〉 делать передаточную надпись, индоссировать, жирировать'])

        self.assertCountEqual(self.wd.lookup_translations_only('匹', 'ひき'),
                              ['《счетный суф. для животных》', '《после числит.》 хики 《мера измерения тканей = 21,2 м》',
                               '《после числит.》 штука 《напр. шелка》', '= 2 тан = 21,2 м 《для измерения тканей》'])

        self.assertCountEqual(self.wd.lookup_translations_only('思わせる', 'おもわせる'),
                              ['〈hen 【変】 da to〉＿ производить странное впечатление; наводить на мысль, что 《что-л.》 странно',
                               'напоминать [собой]'])

        self.assertCountEqual(self.wd.lookup_translations_only('取れる', 'とれる'),
                              ['получаться 《о деньгах, плате》', 'получаться, добываться', 'ловиться',
                               'получаться, выходить 《о снимке》', 'отрываться; отламываться',
                               'сходить 《о пятне; перен.》 проходить'])

        self.assertCountEqual(self.wd.lookup_translations_only('領域', 'りょういき'),
                              ['территория, владения; район', '《перен.》 область, сфера 《чего-л.》'])

        self.assertCountEqual(self.wd.lookup_translations_only('起こさせる', 'おこさせる'),
                              ['вызывать, пробуждать 《у кого-л. какое-л. состояние, чувство》',
                               '《побуд. форма гл.》 поднимать 《лежащего, упавшего》',
                               '《побуд. форма гл.》 поднимать, восстанавливать, возрождать',
                               '《побуд. форма гл.》 будить 《спящего》',
                               '《побуд. форма гл.》 начинать, класть начало',
                               '《побуд. форма гл.》 открывать, учреждать',
                               '《побуд. форма гл.》 возбуждать, вызывать',
                               '《побуд. форма гл.》 《обозначает появление какой-л. эмоции, состояния, болезни》'])

        self.assertCountEqual(self.wd.lookup_translations_only('蛻け', 'もぬけ'),
                              ['линька 《животных》',
                               'сброшенная кожа 《пресмыкающихся, куколок насекомых》; о постели, в которой никто не '
                               'спал; о брошенном [жильцами] доме; о мертвом теле'])

        self.assertCountEqual(self.wd.lookup_translations_only('七草', 'ななくさ'),
                              ['«семь трав» 《семь полевых цветов》',
                               '(《сокр.》 nanakusa 【七草】 no sekku 【節句】) праздник «семи трав» 《7 января》'])

        self.assertCountEqual(self.wd.lookup_translations_only('働かす', 'はたらかす'),
                              ['работать 《чем-л.》; заставлять функционировать, приводить в движение 《что-либо》',
                               '《побуд. залог гл.》 работать, трудиться'])

        self.assertCountEqual(self.wd.lookup_translations_only('版籍', 'はんせき'),
                              ['(《сокр. от》 hanto 【版図】, koseki 【戸籍】) 《ист.》 страна и народ'])

        self.assertCountEqual(self.wd.lookup_translations_only('風趣', 'ふうしゅ'),
                              ['《кн.》 вкус; прелесть 《чего-л.》; 〈~no nai〉 безвкусный', '《кн.》 тон, вид'])

        self.assertCountEqual(self.wd.lookup_translations_only('見出す', 'みいだす'),
                              ['высмотреть, заметить, обнаружить, открыть', 'высматривать, выискивать; отбирать, выбирать'])

        self.assertCountEqual(self.wd.lookup_translations_only('浅草', 'あさくさ'),
                              ['асакуса 《уст., вошел в р-н дайто》', 'асакуса 《парк в токио》'])

        self.assertCountEqual(self.wd.lookup_translations_only('中共', 'ちゅうきょう'),
                              ['(《сокр.》 chuugoku 【中国】 kyousantou 【共産党】) кпк 《коммунистическая партия китая》',
                               '《географическое название》 《сокр.》 кнр', '《сокр.》 кнр'])

        self.assertCountEqual(self.wd.lookup_translations_only('阿羅漢', 'あらかん'),
                              ['(《санскр.》 arhan) 《буд.》 архан 《высшая степень духовного совершенства》',
                               '(《санскр.》 arhan) 《буд.》 архан 《подвижник, достигший этой степени совершенства》'])

        self.assertCountEqual(self.wd.lookup_translations_only('軈て', 'やがて'),
                              ['скоро, вскоре; немного спустя', 'почти', '〈~…de aru〉 《что-л.》 [есть] не что иное, как…'])
        self.assertCountEqual(self.wd.lookup_translations_only('敢えて', 'あえて'),
                              ['смело, решительно; 〈~…suru〉 отваживаться, осмеливаться 《что-л. сделать》; идти 《на что-л.》',
                               'определенно, твердо'])

    def test_bad_readings(self):
        bad = []
        for e in self.wd._entries:
            for rd in e.reading:
                if any(not _is_hiragana(ch) for ch in rd):
                    bad.append(e)
        for b in bad:
            print(b.eid, b.reading, b.translation)
        self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
