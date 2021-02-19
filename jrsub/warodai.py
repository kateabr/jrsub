import os
import pathlib
import pickle
import re
from dataclasses import dataclass
from operator import itemgetter
from typing import List

from jtran import JTran
from tqdm import tqdm

from jrsub import WarodaiEntry
from jrsub import WarodaiReference
from jrsub import SearchResult, SearchMode
from jrsub.utils import _hiragana_to_latin, _is_kanji, _is_hira_or_kata, _is_hiragana, _distance


@dataclass
class WarodaiEid:
    val1: int
    val2: int
    val3: int

    def inc(self) -> str:
        inc2, self.val3 = divmod(self.val3 + 1, 1000)
        inc1, self.val2 = divmod(self.val2 + inc2, 100)
        self.val1 += inc1
        return str(self)

    def __init__(self, eid: str):
        split_eid = eid.split('-')
        self.val1 = int(split_eid[0])
        self.val2 = int(split_eid[1])
        self.val3 = int(split_eid[2])

    def __str__(self):
        return f'{str(self.val1).rjust(3, "0")}-{str(self.val2).rjust(2, "0")}-{str(self.val3).rjust(2, "0")}'

    def __eq__(self, other: WarodaiReference):
        return str(self) == str(other)

    def __gt__(self, other: WarodaiReference):
        return str(self) > str(other)

    def __lt__(self, other: WarodaiReference):
        return str(self) < str(other)


class WarodaiDictionary:
    _entries: List[WarodaiEntry]

    def __init__(self, entries: List[WarodaiEntry]):
        self._entries = entries

    def _get_entry_by_eid(self, eid) -> WarodaiEntry:
        return [e for e in self._entries if e.eid == eid][0]

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

    def _translate_by_eid(self, eid: str) -> List[str]:
        def _retrieve_from_references(refs: [WarodaiReference], visited_references: List[str]) -> List[str]:
            if not refs:
                return []

            meanings = []
            rrefs = []

            for ref in refs:
                if ref.eid in visited_references:
                    continue
                temp_meanings = []
                temp_rrefs = []
                entry = self._get_entry_by_eid(ref.eid)
                if ref.meaning_number != ['-1']:
                    for m_n in ref.meaning_number:
                        if m_n in entry.translation.keys():
                            temp_meanings.extend(entry.translation[m_n])
                        if m_n in entry.references.keys():
                            temp_rrefs.extend(entry.references[m_n])
                else:
                    temp_meanings.extend(sum(list(entry.translation.values()), []))
                    temp_rrefs.extend(sum(list(entry.references.values()), []))

                if ref.prefix:
                    temp_meanings = '; '.join(temp_meanings).split('; ')
                    temp_meanings = [re.sub(r'(〈.+?[〉＿]\s)', '', meaning) for meaning in temp_meanings if
                                     meaning.startswith(ref.prefix)]

                    temp_rrefs = [r for r in temp_rrefs if ref.prefix in r.body]

                if ref.mode:
                    temp_meanings = [ref.mode + ' ' + meaning for meaning in temp_meanings]

                if ref.body:
                    temp_meanings = [ref.body + ' ' + meaning for meaning in temp_meanings]

                meanings.extend(temp_meanings)
                rrefs.extend(temp_rrefs)
                visited_references.append(ref.eid)

            return meanings + _retrieve_from_references(rrefs, visited_references)

        entry = self._get_entry_by_eid(eid)

        return sum(list(entry.translation.values()), []) + [m for m in _retrieve_from_references(
            sum(list(entry.references.values()), []), [entry.eid]) if m]

    def lookup_translations_only(self, lexeme: str, reading: str = '') -> List[str]:
        return list(dict.fromkeys(sum([tr.translation for tr in self.lookup(lexeme, reading)], [])))

    def save(self, fname: str = "default"):
        if fname == 'default':
            fname = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.joinpath(
                'dictionaries/warodai.jtdb')
        pickle.dump(self, open(fname, "wb"))


class WarodaiLoader:
    _entries: List[WarodaiEntry]
    _transliterate_collocations: bool = True
    _normalizer: {str: str} = {'noaru': 'no aru',
                               'no': 'no',
                               'taru': 'taru',
                               'suru': 'suru',
                               'nisuru': 'ni suru',
                               'ninaru': 'ni naru',
                               'shite': 'shite',
                               'deyaru': 'de yaru',
                               'na': 'na',
                               'nonai': 'no nai',
                               'wosuru': 'wo suru',
                               'su[ru]': 'su[ru]',
                               'toshiteiru': 'to shite iru',
                               'to': 'to',
                               '[to]shita': '[to] shita',
                               'de': 'de',
                               'gasuru': 'ga suru',
                               'ni': 'ni',
                               '[ni]ha': '[ni] wa',
                               '[no]shita': '[no] shita',
                               'dearu': 'de aru',
                               'gaaru': 'ga aru',
                               'aru': 'aru',
                               '[to]': '[to]',
                               '[ni]': '[ni]',
                               'kara': 'kara',
                               '[no]': '[no]',
                               'niaru': 'ni aru',
                               'he': 'e',
                               'made': 'made',
                               'ninatte': 'ni natte',
                               'tosuru': 'to suru',
                               'toshite': 'to shite',
                               'ganai': 'ga nai', 'tonaru': 'to naru', 'shita': 'shita', 'shinai': 'shinai',
                               'monai': 'mo nai',
                               '！': '!',
                               'niyaru': 'ni yaru',
                               'woyaru': 'wo yaru',
                               'sareru': 'sareru',
                               '[mo]naku': '[mo] naku',
                               'ba': 'ba',
                               'wo': 'wo',
                               'naru': 'naru',
                               'nimo': 'ni mo',
                               'saseru': 'saseru',
                               'tonaku': 'to naku 【鳴く】',
                               '[ka]no': '[ka] no',
                               '[ka]': '[ka]',
                               'subeki': 'subeki',
                               'toshinai': 'to shinai',
                               'gaatte': 'ga atte',
                               'yori': 'yori',
                               'mo': 'mo',
                               'tomonaku': 'to mo naku',
                               'heka': 'e ka',
                               'ka': 'ka',
                               '[toshite]': '[to shite]',
                               'naku': 'naku',
                               'na[ru]': 'na[ru]',
                               'ni[shite]': 'ni [shite]',
                               'tonatte': 'to natte',
                               'nishite': 'ni shite',
                               'wonasu': 'wo nasu',
                               'nishiteyaru': 'ni shite yaru',
                               'desuru': 'de suru',
                               '[ha]': '[wa]',
                               'ninatteiru': 'ni natte iru',
                               'woshiteiru': 'wo shite iru',
                               'nashi': 'nashi',
                               'naki': 'naki',
                               'shiteiru': 'shite iru',
                               '[na]': '[na]',
                               'noshita': 'no shita',
                               'to[shite]': 'to [shite]',
                               '[wo]suru': '[wo] suru',
                               '[no]aru': '[no] aru',
                               'narashimeru': 'narashimeru',
                               'ni(no)': 'ni (no)',
                               'monaku': 'mo naku',
                               'karano': 'kara no',
                               'da': 'da',
                               '[de]': '[de]',
                               'nosuru': 'no suru',
                               '[shite]': '[shite]',
                               'woshita': 'wo shita',
                               '[ni]suru': '[ni] suru',
                               'woshinai': 'wo shinai',
                               '[shita]': '[shita]',
                               'nishinai': 'ni shinai',
                               'seru': 'seru',
                               '[noaru]': '[no aru]',
                               'toshita': 'to shita',
                               'tosureba': 'to sureba',
                               'toshitemo': 'to shite mo',
                               'sureba': 'sureba',
                               'sarenai': 'sarenai',
                               'ninai': 'ni nai',
                               'tari': 'tari',
                               'nai': 'nai',
                               '[ga]suru': '[ga] suru',
                               '[to]shiteiru': '[to] shite iru',
                               'sarete': 'sarete',
                               'wosaseru': 'wo saseru',
                               'niatte': 'ni atte',
                               'woshite': 'wo shite',
                               'seshimeru': 'seshimeru',
                               'shite…wosaseru': 'shite… wo saseru',
                               '[kara]': '[kara]',
                               'ga': 'ga',
                               'tosuru(naru)': 'to suru (naru)',
                               'niyori': 'ni yori',
                               'ha': 'wa',
                               'nimonai': 'ni mo nai',
                               'kara[no]': 'kara [no]',
                               'nishiteiru': 'ni shite iru',
                               '[notame]ni': '[no tame] ni',
                               'ni[natte]': 'ni [natte]',
                               'nara': 'nara',
                               'deha': 'de wa',
                               '[taru]': '[taru]',
                               'ninarau': 'ni narau',
                               'ni！': 'ni!',
                               'denai': 'de nai',
                               'ga…aru': "ga… aru",
                               'shitearu': 'shite aru',
                               '[naru]': '[naru]',
                               'shiteinai': 'shite inai',
                               'sareta': 'sareta',
                               'desu': 'desu',
                               'tonaru(suru)': 'to naru (suru)',
                               'naran': 'naran',
                               'noshitearu': 'no shite aru',
                               'no(ninaru)': 'no (ni naru)',
                               'shinagara': 'shinagara',
                               'nagara': 'nagara',
                               'womotte': 'wo motte',
                               'niyotte': 'ni yotte',
                               'noshinai': 'no shinai',
                               'suru(naru)': 'suru (naru)',
                               'demo': 'demo',
                               'tarashimeru': 'tarashimeru',
                               'ni…wosuru': 'ni… wo suru',
                               'nika': 'ni ka',
                               '[de]mo': '[de]mo',
                               'temo': 'te mo',
                               'nisareru': 'ni sareru',
                               'to[naku]': 'to [naku]',
                               '[mo]': '[mo]',
                               'tonai': 'to nai',
                               'ninatta': 'ni natta',
                               'no(ni)': 'no (ni)',
                               'nishita': 'ni shita',
                               'denaku': 'de naku',
                               'sasete': 'sasete',
                               'gashiteiru': 'ga shite iru',
                               'shite…suru': 'shite… suru',
                               'ni…suru': 'ni… suru',
                               'deiru': 'de iru',
                               'ganaku': 'ga naku',
                               'ni[te]': 'ni[te]',
                               'no[aru]': 'no [aru]',
                               'wosuru(yaru)': 'wo suru (yaru)',
                               'gashinai': 'ga shinai',
                               'ni[oite]': 'ni [oite]',
                               'wo…to': 'wo… to',
                               'ninaranai': 'ni naranai',
                               '[mo]nai': '[mo] nai',
                               'nisaseru': 'ni saseru',
                               'ni…saseru': 'ni… saseru',
                               'ni[mo]': 'ni [mo]',
                               'sezaru': 'sezaru',
                               'naraba': 'naraba',
                               'nisuru(naru)': 'ni suru (naru)',
                               'sareteiru': 'sarete iru',
                               '的': 'teki 【的】',
                               '的no': 'teki 【的】 no',
                               '的ni': 'teki 【的】 ni',
                               '[的]': '[teki 【的】]',
                               '的[na]': 'teki 【的】 [na]',
                               '[的no]': '[teki 【的】 no]',
                               '的na': 'teki 【的】 na',
                               '的[no]': 'teki 【的】 [no]',
                               '[的]ni': '[teki 【的】] ni',
                               '～': '~',
                               '…': '-',
                               'gasuru(gaaru)': 'ga suru (ga aru)',
                               '[ga]aru': '[ga] aru',
                               'suru(dearu)': 'suru (de aru)',
                               'to(ni)': 'to (ni)',
                               'niiru': 'ni iru',
                               'ni(to)suru': 'ni (to) suru',
                               'ninaruto': 'ni naru to',
                               'toshite(suru)': 'to shite (suru)',
                               'dearu(ninaru)': 'de aru (ni naru)',
                               '[niha]': '[ni wa]',
                               '＝': '',
                               '｜': '|',
                               '落chiru': 'ochiru 【落ちる】',
                               'shite置ku': 'shite oku 【置く】',
                               'de飲mu': 'de nomu 【飲む】',
                               'nikakaru': 'ni kakaru',
                               'ni倒reru': 'ni taoreru 【倒れる】',
                               '最後no一周ha彼no': 'saigo 【最後】 no isshuu 【一周】 wa kare 【彼】 no',
                               'no観gaatta': 'no kan 【観】 ga atta',
                               'wo踏mu': 'wo fumu 【踏む】',
                               '二十円': 'nijuuen 【二十円】',
                               '禁zezu': 'kinzezu 【禁ぜず】',
                               'de行ku': 'de iku 【行く】',
                               '両極端ha': 'ryoukyokutan 【両極端】 wa',
                               'su': 'su',
                               'wo言u': 'wo iu 【言う】',
                               'wo打tsu': 'wo utsu 【打つ】',
                               '顔wo': 'kao 【顔】 wo',
                               'kotowo知ranu': 'koto wo shiranu 【知らぬ】',
                               'no笑iwo浮beru': 'no warai 【笑い】 wo ukaberu 【浮べる】',
                               '夜no': 'yoru 【夜】 no',
                               '空': 'sora 【空】',
                               '夜ga': 'yoru 【夜】 ga',
                               'ta': 'ta',
                               'douzo': 'douzo',
                               '[思tte下sai]': '[omotte 【思って】 kudasai 【下さい】]',
                               'wo厳重nisuru': 'wo genjuu 【厳重】 ni suru',
                               'gayoi': 'ga yoi',
                               '人': 'hito 【人】',
                               'wo使u': 'wo tsukau 【使う】',
                               '二人ha': 'futari 【二人】 wa',
                               'to逃geru': 'to nigeru 【逃げる】',
                               '消eteshimau': 'kiete 【消えて】 shimau',
                               'iu': 'iu',
                               'yaru': 'yaru',
                               'no雪': 'no yuki 【雪】',
                               'wohishigu': 'wo hishigu',
                               'wo抜ku': 'wo nuku 【抜く】',
                               'ni振ru舞u': 'ni furumau 【振る舞う】',
                               '目ni': 'me 【目】 ni',
                               '疑心': 'gishin 【疑心】',
                               'wo生zu': 'wo shouzu 【生ず】',
                               'tosuruni足ru': 'to suru ni taru 【足る】',
                               '意ni': 'i 【意】 ni',
                               '故人': 'kojin 【故人】',
                               'dakara君karayaritamae': 'dakara kimi 【君】 kara yaritamae',
                               'no概gaaru': 'no gai 【概】 ga aru',
                               'wosarasu': 'wo sarasu',
                               'noyoi': 'no yoi',
                               '夫婦ha': 'fuufu 【夫婦】 wa',
                               'no年頃': 'no toshigoro 【年頃】',
                               '[打tte]': '[utte 【打って】]',
                               'ni及bazu': 'ni oyobazu 【及ばず】',
                               'no中/ウチ/ni収meru': 'no uchi 【中】 ni osameru 【収める】',
                               'notsuwamono': 'no tsuwamono',
                               'no夢/ユメ/': 'no yume 【夢】',
                               '五分試meshinisuru': 'gobudameshi 【五分試めし】 ni suru',
                               'wo云u': 'wo iu 【云う】',
                               'ikimashou': 'ikimashou',
                               '地ni塗reru': 'chi 【地】 ni mamireru 【塗れる】',
                               'no力wo貸su': 'no chikara 【力】 wo kasu 【貸す】',
                               '変watta男': 'kawatta 【変わった】 otoko 【男】',
                               'wo以te労wo待tsu': 'wo motte 【以て】 rou 【労】 wo matsu 【待つ】',
                               '赤n坊ni': 'akanbou 【赤ん坊】 ni',
                               'woyatte見ru': 'wo yatte miru 【見る】',
                               '見eru': 'mieru 【見える】',
                               'nomamade買u': 'no mama de kau 【買う】',
                               '逃geru': 'nigeru 【逃げる】',
                               'to申su': 'to mousu 【申す】',
                               'ni出ru': 'ni deru 【出る】',
                               'no子': 'no ko 【子】',
                               'toiu所gaaru': 'to iu tokoro 【所】 ga aru',
                               'no月': 'no tsuki 【月】',
                               'no所': 'no tokoro 【所】',
                               'zutto': 'zutto',
                               '返事': 'henji 【返事】',
                               'no女': 'no onna 【女】',
                               'bakarini': 'bakari ni',
                               'areba水心/ミズゴコロ/': 'areba mizugokoro 【水心】',
                               '顔wosuru': 'kao 【顔】 wo suru',
                               'ni身wo沈meru': 'ni mi 【身】 wo shizumeru 【沈める】',
                               'no衆': 'no shuu 【衆】',
                               '得意no鼻wo': 'tokui 【得意】 no hana 【鼻】 wo',
                               'ni縛ru': 'ni shibaru 【縛る】',
                               'wo差su': 'wo sasu 【差す】',
                               'wo並be立teru': 'wo narabetateru 【並べ立てる】',
                               'ga上garanai': 'ga agaranai 【上がらない】',
                               'no小槌': 'no kozuchi 【小槌】',
                               '一点no': 'itten 【一点】 no',
                               'wo切ru': 'wo kiru 【切る】',
                               'no強i男': 'no tsuyoi 【強い】 otoko 【男】',
                               'detsuitahodonosukimonai': 'de tsuita hodo no suki mo nai',
                               'no尼': 'no ama 【尼】',
                               'narukana': 'naru ka na',
                               'nashini': 'nashi ni',
                               'nonaiyouni': 'no nai you ni',
                               'ni買i言葉': 'ni kaikotoba 【買い言葉】',
                               'ni泣ku': 'ni naku 【泣く】',
                               'no空': 'no sora 【空】',
                               'tomosuntomo言wanai': 'to mo sun to mo iwanai 【言わない】',
                               'mo言warenu': 'mo iwarenu 【言われぬ】',
                               'ni達suru': 'ni tassuru 【達する】',
                               '時wo': 'toki 【時】 wo',
                               'ni振舞u': 'ni furumau 【振舞う】',
                               'no沙汰/サタ/': 'no sata 【沙汰】',
                               'no知renai': 'no shirenai 【知れない】',
                               'ni入/イ/ru': 'ni iru 【入る】',
                               'no士': 'no shi 【士】',
                               'wo伸basu': 'wo nobasu 【伸ばす】',
                               '泣ku': 'naku 【泣く】',
                               'wo食u': 'wo kuu 【食う】',
                               'wokaku': 'wo kaku',
                               'ni言u': 'ni iu 【言う】',
                               'sonnakotogaarukashira！arutomo、arutomo': 'sonna koto ga aru kashira! aru tomo, arutomo',
                               'ni腐tte': 'ni kusatte 【腐って】',
                               'no鉄砲': 'no teppou 【鉄砲】',
                               'wo付keru': 'wo tsukeru 【付ける】',
                               'ni見ru': 'ni miru 【見る】',
                               'ni揉meru': 'ni momeru 【揉める】',
                               'nasai': 'nasai',
                               'ni入reru': 'ni ireru 【入れる】',
                               'no夫': 'no otto 【夫】',
                               '子供no': 'kodomo 【子供】 no',
                               'korede': 'kore de',
                               'dayo': 'da yo',
                               '申shimashita': 'moushimashita 【申しました】',
                               'wo吹kaseru': 'wo fukaseru 【吹かせる】',
                               'wo吹kasu': 'wo fukasu 【吹かす】',
                               'ga悪i': 'ga warui 【悪い】',
                               'de育teru': 'de sodateru 【育てる】',
                               'wo去ri実/ジツ/ni就ku': 'wo sari 【去り】 jitsu 【実】 ni tsuku 【就く】',
                               'no者': 'no mono 【者】',
                               'no恥wososogu': 'no haji 【恥】 wo sosogu',
                               'wo叫bu': 'wo sakebu 【叫ぶ】',
                               '人口ni': 'jinkou 【人口】 ni',
                               'no作者': 'no sakusha 【作者】',
                               'no動物': 'no doubutsu 【動物】',
                               'naranu': 'naranu',
                               'ga出来nai': 'ga dekinai 【出来ない】',
                               'no医者': 'no isha 【医者】',
                               'no式': 'no shiki 【式】',
                               '涙ni': 'namida 【涙】 ni',
                               'gaii': 'ga ii',
                               'nookori': 'no okori',
                               'woiu': 'wo iu',
                               'wo利ku': 'wo kiku 【利く】',
                               '[詐欺]wo働ku': '[sagi 【詐欺】] wo hataraku 【働く】',
                               '御': 'go 【御】',
                               'wo祈ru': 'wo inoru 【祈る】',
                               'no国ni遊bu': 'no kuni 【国】 ni asobu 【遊ぶ】',
                               'no典': 'no ten 【典】',
                               'noノ:ト': 'no NOUTO 【ノート】',
                               'wo取ru': 'wo toru 【取る】',
                               '風no': 'kaze 【風】 no',
                               'wotsuku': 'wo tsuku',
                               'ni乗seru': 'ni noseru 【乗せる】',
                               'to音wo立tete落chiru': 'to oto 【音】 wo tatete 【立てて】 ochiru 【落ちる】',
                               'no時計': 'no tokei 【時計】',
                               'no感gaaru': 'no kan 【感】 ga aru',
                               'no悪i': 'no warui 【悪い】',
                               'ni立tsu': 'ni tatsu 【立つ】',
                               'noyoi女': 'no yoi onna 【女】',
                               '髪no': 'kami 【髪】 no',
                               '鼻wo': 'hana 【鼻】 wo',
                               'to笑u': 'to warau 【笑う】',
                               '音wo立teru': 'oto 【音】 wo tateru 【立てる】',
                               '声wo': 'koe 【声】 wo',
                               '世ha': 'yo 【世】 wa',
                               'to乱reta': 'to midareta 【乱れた】',
                               'ga故ni': 'ga yue 【故】 ni',
                               '声ga': 'koe 【声】 ga',
                               '犬no': 'inu 【犬】 no',
                               '酒no': 'sake 【酒】 no',
                               'wosuru(付keru)': 'wo suru (tsukeru 【付ける】)',
                               'wo交jieru': 'wo majieru 【交じえる】',
                               'no論': 'no ron 【論】',
                               'no至ridearu': 'no itari 【至り】 de aru',
                               '事務wo': 'jimu 【事務】 wo',
                               '生死no': 'seishi 【生死】 no',
                               'no道': 'no michi 【道】',
                               'no祭/マツリ/': 'no matsuri 【祭】',
                               'mou': 'mou',
                               'no緒/o/ga切reta': 'no o 【緒】 ga kireta 【切れた】',
                               'no人': 'no hito 【人】',
                               'nakimade': 'naki made',
                               'no交wari': 'no majiwari 【交わり】',
                               'no政治家': 'no seijika 【政治家】',
                               '鳴ku': 'naku 【鳴く】',
                               '飲mu': 'nomu 【飲む】',
                               'ni帰suru': 'ni kisuru 【帰する】',
                               '言u': 'iu 【言う】',
                               'to鳴ru': 'to naru 【鳴る】',
                               '[no者]': '[no mono 【者】]',
                               'no男': 'no otoko 【男】',
                               'suru(shitai)': 'suru (shitai)',
                               'no下ni': 'no moto 【下】 ni',
                               'sorega': 'sore ga',
                               'datta': 'datta',
                               'no紋': 'no mon 【紋】',
                               '妙na': 'myou 【妙】 na',
                               'ga動ite': 'ga ugoite 【動いて】',
                               '鳴ru': 'naru 【鳴る】',
                               '矢nogotoshi': 'ya 【矢】 no gotoshi',
                               'no齢/ヨワイ/': 'no yowai 【齢】',
                               'ni入/ハイ/ru': 'ni hairu 【入る】',
                               '私ha': 'watashi 【私】 wa',
                               'no別re': 'no wakare 【別れ】',
                               '叫bu': 'sakebu 【叫ぶ】',
                               'no一毛/イチモウ/': 'no ichimou 【一毛】',
                               'no位/クライ/': 'no kurai 【位】',
                               '万事': 'banji 【万事】',
                               'no地': 'no chi 【地】',
                               '半/ナカバ/': 'nakaba 【半】',
                               'de指su': 'de sasu 【指す】',
                               'no巷': 'no chimata 【巷】',
                               'noエゾ': 'no EZO 【エゾ】',
                               'no戦術': 'no senjutsu 【戦術】',
                               'no徒': 'no to 【徒】',
                               'toshite退場suru': 'to shite taijousuru 【退場する】',
                               'toshite動kanai': 'to shite ugokanai 【動かない】',
                               'wo進meru': 'wo susumeru 【進める】',
                               '目wo': 'me 【目】 wo',
                               '眺meru': 'nagameru 【眺める】',
                               'no娘': 'no musume 【娘】',
                               'no栄': 'no ei 【栄】',
                               'no夢': 'no yume 【夢】',
                               '[no身]': '[no mi 【身】]',
                               '相和/アイワ/su[ru]': 'aiwasu[ru] 【相和す[る]】',
                               '更ni花wo添eru': 'sara 【更】 ni hana 【花】 wo soeru 【添える】',
                               'wo起su': 'wo okosu 【起す】',
                               '顔付': 'kaotsuki 【顔付】',
                               '言u音': 'iu 【言う】 oto 【音】',
                               'de進mu': 'de susumu 【進む】',
                               'dane': 'da ne',
                               'ni葬rareru': 'ni houmurareru 【葬られる】',
                               'wo肥/コ/yasu': 'wo koyasu 【肥やす】',
                               'to光ru': 'to hikaru 【光る】',
                               '歯wo鳴rasu': 'ha 【歯】 wo narasu 【鳴らす】',
                               '音gasuru': 'oto 【音】 ga suru',
                               'wonameru': 'wo nameru',
                               '毛no': 'ke 【毛】 no',
                               '彼ha': 'kare 【彼】 wa',
                               'ni落着iteiru': 'ni ochitsuite 【落着いて】 iru',
                               'nikenasu': 'ni kenasu',
                               'wo買tte出ru': 'wo katte 【買って】 deru 【出る】',
                               'na事wo言u': 'na koto 【事】 wo iu 【言う】',
                               '月光': 'gekkou 【月光】',
                               'no水': 'no mizu 【水】',
                               'no家': 'no ie 【家】',
                               'no貰i損': 'no moraizon 【貰い損】',
                               '日ga': 'hi 【日】 ga',
                               'no様na人dakari': 'no you 【様】 na hitodakari 【人だかり】',
                               '鹿児島': 'kagoshima 【鹿児島】',
                               'he行ku': 'e iku 【行く】',
                               'no音/ne/': 'no ne 【音】',
                               'nikowareru': 'ni kowareru',
                               '眠ru': 'nemuru 【眠る】',
                               '煮ru': 'niru 【煮る】',
                               'ni酔pparau': 'ni yopparau 【酔っぱらう】',
                               'ni行ku': 'ni iku 【行く】',
                               'no美人': 'no bijin 【美人】',
                               'wo直su': 'wo naosu 【直す】',
                               'no勢de': 'no ikioi 【勢】 de',
                               'ni伏奏su': 'ni fukusousu 【伏奏す】',
                               'wo補u': 'wo oginau 【補う】',
                               'no乱': 'no ran 【乱】',
                               'ni触reru': 'ni fureru 【触れる】',
                               '痩seru': 'yaseru 【痩せる】',
                               '笑u': 'warau 【笑う】',
                               'no生徒': 'no seito 【生徒】',
                               'ni当tte見ru': 'ni atatte 【当って】 miru 【見る】',
                               '泡/アワ/wo飛bashite': 'awa 【泡】 wo tobashite 【飛ばして】',
                               'no念': 'no nen 【念】',
                               'ni微笑wotataete': 'ni bishou 【微笑】 wo tataete',
                               '獅子no': 'shishi 【獅子】 no',
                               'wohineru': 'wo hineru',
                               'wo傾keru': 'wo katamukeru 【傾ける】',
                               'wokashigeru': 'wo kashigeru',
                               'tobakari': 'to bakari',
                               'ni待tsu': 'ni matsu 【待つ】',
                               'bakari': 'bakari',
                               'wo屈meru': 'wo kagameru 【屈める】',
                               '風邪wo': 'kaze 【風邪】 wo',
                               '居眠riwosuru': 'inemuri 【居眠り】 wo suru',
                               'ni話su': 'ni hanasu 【話す】',
                               'to頭wo打chitsukeru': 'to atama 【頭】 wo uchitsukeru 【打ちつける】',
                               'to音gashita': 'to oto 【音】 ga shita',
                               'te': 'te',
                               'no計(策)': 'no kei 【計】 (saku 【策】)',
                               'to鳴ku': 'to naku 【鳴く】',
                               '一tsunishita': 'hitotsu 【一つ】 ni shita',
                               '御通知申shi上gemasu': 'gotsuuchi 【御通知】 moushiagemasu 【申し上げます】',
                               'no民/タミ/': 'no tami 【民】',
                               'ni歩ku': 'ni aruku 【歩く】',
                               'wosukuu': 'wo sukuu',
                               'ni挟mu': 'ni hasamu 【挟む】',
                               '家/イエ/no': 'ie 【家】 no',
                               'to焼keteiru': 'to yakete 【焼けて】 iru',
                               'to飲mu': 'to nomu 【飲む】',
                               'no舞/マイ/': 'no mai 【舞】',
                               'to打tsu': 'to utsu 【打つ】',
                               '[no道/ドウ/]': '[no dou 【道】]',
                               'to寝転bu': 'to nekorobu 【寝転ぶ】',
                               '小刀wo': 'kogatana 【小刀】 wo',
                               'ni持tsu(取ru)': 'ni motsu 【持つ】 (toru 【取る】)',
                               'ga起ru': 'ga okoru 【起る】',
                               '事': 'koto 【事】',
                               '涙': 'namida 【涙】',
                               'no楼閣/ロウカク/': 'no roukaku 【楼閣】',
                               'wo決mekomu': 'wo kimekomu 【決めこむ】',
                               'to泣ku': 'to naku 【泣く】',
                               '言umo': 'iu 【言う】 mo',
                               '日/ヒ/既ni': 'hi 【日】 sude 【既】 ni',
                               'no礼wotoru': 'no rei 【礼】 wo toru',
                               'no才': 'no sai 【才】',
                               'no舌/シタ/wo振u': 'no shita 【舌】 wo furuu 【振う】',
                               'wo極meta': 'wo kiwameta 【極めた】',
                               'to包丁wo入reru': 'to houchou 【包丁】 wo ireru 【入れる】',
                               '[to]湯ni入/ハイ/ru': '[to] yu 【湯】 ni hairu 【入る】',
                               '[to]飛bi込mu': '[to] tobikomu 【飛び込む】',
                               '舌ni': 'shita 【舌】 ni',
                               'no二': 'no ni 【二】',
                               '身ni': 'mi 【身】 ni',
                               'no兔': 'no usagi 【兔】',
                               'no臣': 'no shin 【臣】',
                               'shitemo及banai': 'shite mo oyobanai 【及ばない】',
                               '両端wo持/ジ/su': 'ryoutan 【両端】 wo jisu 【持す】',
                               'wo誤razu': 'wo ayamarazu 【誤らず】',
                               'no誉/ホマレ/gaaru': 'no homare 【誉】 ga aru',
                               'ni乗ru': 'ni noru 【乗る】',
                               '夜ha': 'yoru 【夜】 wa',
                               'to更kete行ku': 'to fukete 【更けて】 iku 【行く】',
                               'wokakete言u': 'wo kakete iu 【言う】',
                               'nikotaeru': 'ni kotaeru',
                               'no物tosuru': 'no mono 【物】 to suru',
                               '[wo]踏mu': '[wo] fumu 【踏む】',
                               'no明/メイ/gaaru': 'no mei 【明】 ga aru',
                               'sono言ha今monao': 'sono koto 【言】 wa ima 【今】 mo nao',
                               'no従業員': 'no juugyouin 【従業員】',
                               'no職員': 'no shokuin 【職員】',
                               'no八九made': 'no hakku 【八九】 made',
                               'ni陥ru': 'ni ochiiru 【陥る】',
                               'notame': 'no tame',
                               'wo挙geru': 'wo ageru 【挙げる】',
                               'nouchi': 'no uchi',
                               'no節句/セック/': 'no sekku 【節句】',
                               'janai': 'ja nai',
                               '見ru': 'miru 【見る】',
                               'ni落chinai': 'ni ochinai 【落ちない】',
                               '[to]飛bu': '[to] tobu 【飛ぶ】',
                               'wo決suru': 'wo kessuru 【決する】',
                               '夫婦ninaru': 'fuufu 【夫婦】 ni naru',
                               '転bu': 'korobu 【転ぶ】',
                               'korobu': 'korobu',
                               'ni行ki過giru': 'ni ikisugiru 【行き過ぎる】',
                               'untomo': 'un to mo',
                               'tomo言wanai': 'to mo iwanai 【言わない】',
                               'ni切ru': 'ni kiru 【切る】',
                               'to倒reru': 'to taoreru 【倒れる】',
                               '[to]刺su': '[to] sasu 【刺す】',
                               'ni酔u': 'ni you 【酔う】',
                               'to落chiru': 'to ochiru 【落ちる】',
                               'ni力wo入reru': 'ni chikara 【力】 wo ireru 【入れる】',
                               'no権': 'no ken 【権】',
                               'ni起然tosuru': 'ni chouzen 【起然】 to suru',
                               'no遣ri方': 'no yarikata 【遣り方】',
                               'no谷': 'no tani 【谷】',
                               'wo着keru': 'wo tsukeru 【着ける】',
                               '面/ツラ/no皮ga': 'tsura 【面】 no kawa 【皮】 ga',
                               'ni暮su': 'ni kurasu 【暮す】',
                               '追(逐)tte': 'otte 【追って, 逐って】',
                               'wo傾倒suru': 'wo keitousuru 【傾倒する】',
                               '[御免下sai]': '[gomen 【御免】 kudasai 【下さい】]',
                               'no琴/コト/': 'no koto 【琴】',
                               'ni浮bu': 'ni ukabu 【浮ぶ】',
                               'ni現wareru': 'ni arawareru 【現われる】',
                               'no仁/ジン/': 'no jin 【仁】',
                               'toshite鳴ru': 'to shite naru 【鳴る】',
                               'no本': 'no hon 【本】',
                               'wokakeru': 'wo kakeru',
                               'no足': 'no ashi 【足】',
                               '模様wo': 'moyou 【模様】 wo',
                               '風ga': 'kaze 【風】 ga',
                               '吹ku': 'fuku 【吹く】',
                               'kara物wo見ru': 'kara mono 【物】 wo miru 【見る】',
                               'ni縛rareta': 'ni shibarareta 【縛られた】',
                               'no見物wosuru': 'no kenbutsu 【見物】 wo suru',
                               '済qi/セイセイ/': 'seisei 【済々】',
                               'no状態dearu': 'no joutai 【状態】 de aru',
                               'no中/ウチ/ni': 'no uchi 【中】 ni',
                               '小便wo': 'shouben 【小便】 wo',
                               'sono': 'sono',
                               'ni卵wo': 'ni tamago 【卵】 wo',
                               'nishite置ku': 'ni shite oku 【置く】',
                               'no如ku': 'no gotoku 【如く】',
                               '不信任no': 'fushinnin 【不信任】 no',
                               'no思igasuru': 'no omoi 【思い】 ga suru',
                               'ni進mu': 'ni susumu 【進む】',
                               'hasamu': 'hasamu',
                               'no味wo覚eru': 'no aji 【味】 wo oboeru 【覚える】',
                               'no一針': 'no isshin 【一針】',
                               '木ni止maru': 'ki 【木】 ni tomaru 【止まる】',
                               'wo出su': 'wo dasu 【出す】',
                               'ni侍/ジ/su': 'ni jisu 【侍す】',
                               'o': 'o',
                               'woshimashite申訳arimasen': 'wo shimashite moushiwake 【申訳】 arimasen',
                               'ga良i': 'ga ii 【良い】',
                               '口wo': 'kuchi 【口】 wo',
                               '五日': 'itsuka 【五日】',
                               '権威': "ken'i 【権威】",
                               'no為ni': 'no tame 【為】 ni',
                               '身wo': 'mi 【身】 wo',
                               '無shi': 'nashi 【無し】',
                               'wo認mezu': 'wo mitomezu 【認めず】',
                               'de歩ku': 'de aruku 【歩く】',
                               '歩ku': 'aruku 【歩く】',
                               'wo引ite待tteiru': 'wo hiite 【引いて】 matte 【待って】 iru',
                               'wo引ku': 'wo hiku 【引く】',
                               '酒wo': 'sake 【酒】 wo',
                               'ni取ru': 'ni toru 【取る】',
                               'no急': 'no kyuu 【急】',
                               '『済mimasen』': '『sumimasen 【済みません】』',
                               '事wo言u': 'koto 【事】 wo iu 【言う】',
                               '両刀wo': 'ryoutou 【両刀】 wo',
                               'wokamu': 'wo kamu',
                               'woshite寝ru': 'wo shite neru 【寝る】',
                               'no筆wo揮u': 'no fude 【筆】 wo furuu 【揮う】',
                               'to酔u': 'to you 【酔う】',
                               'no吏': 'no ri 【吏】',
                               'ga利ku': 'ga kiku 【利く】',
                               'mo無i': 'mo nai 【無い】',
                               '様na': 'you 【様】 na',
                               'yatsu': 'yatsu',
                               '写真wo': 'shashin 【写真】 wo',
                               'no友': 'no tomo 【友】',
                               '[wo(ni)]': '[wo (ni)]',
                               'ni落chiru': 'ni ochiru 【落ちる】',
                               '倒reru': 'taoreru 【倒れる】',
                               'wo割ru': 'wo waru 【割る】',
                               'urikosu': 'urikosu',
                               '買越/カイコ/su': 'kaikosu 【買越す】',
                               '天wo突ku': 'ten 【天】 wo saku 【突く】',
                               'wo決meru': 'wo kimeru 【決める】',
                               'no煩i': 'no wazurai 【煩い】',
                               'no憂i': 'no urei 【憂い】',
                               'no石持草/イシモチソ:/': 'no ishimochisou 【石持草】',
                               'ha両眼wo': 'wa ryougan 【両眼】 wo',
                               'ita': 'ita',
                               'monaihodo': 'mo nai hodo',
                               '事mo': 'koto 【事】 mo',
                               '思案no': 'shian 【思案】 no',
                               '髪wo': 'kami 【髪】 wo',
                               'woageru': 'wo ageru',
                               'wohagasu': 'wo hagasu',
                               'no寿': 'no ju 【寿】',
                               'no難事': 'no nanji 【難事】',
                               '怒ttano': 'okotta 【怒った】 no',
                               '思u': 'omou 【思う】',
                               'no所dearu': 'no tokoro 【所】 de aru',
                               'no武士': 'no bushi 【武士】',
                               'no戸': 'no to 【戸】',
                               'araserareru': 'araserareru',
                               'takenokonoyouni': 'takenoko no you ni',
                               '出teiru': 'dete 【出て】 iru',
                               'no一字': 'no ichiji 【一字】',
                               'ha記憶kara': 'wa kioku 【記憶】 kara',
                               'rareta': 'rareta',
                               '烏no': 'tori 【烏】 no',
                               'no松': 'no matsu 【松】',
                               'wo極meru': 'wo kiwameru 【極める】',
                               'wo飲mu': 'wo nomu 【飲む】',
                               'ni水': 'ni mizu 【水】',
                               'shita顔': 'shita kao 【顔】',
                               '一人': 'hitori 【一人】',
                               'no至ridesu': 'no itari 【至り】 desu',
                               'no陣/ジン/wo布ku': 'no jin 【陣】 wo shiku 【布く】',
                               '羽/ハネ/no': 'hane 【羽】 no',
                               'tta': 'tta',
                               'no人tonaru': 'no hito 【人】 to naru',
                               'kana': 'kana',
                               'de卸su': 'de orosu 【卸す】',
                               'no眉': 'no mayu 【眉】',
                               'no姓/カバネ/': 'no kabane 【姓】',
                               'wo行u': 'wo okonau 【行う】',
                               'no性格': 'no seikaku 【性格】',
                               'ni聞ku': 'ni kiku 【聞く】',
                               'ganaranai': 'ga naranai',
                               '鯛no': 'tai 【鯛】 no',
                               'ni散歩suru': 'ni sanposuru 【散歩する】',
                               'ni食beru': 'ni taberu 【食べる】',
                               'na女': 'na onna 【女】',
                               'yoroshikiwo得ta': 'yoroshiki wo eta 【得た】',
                               'de見ru': 'de miru 【見る】',
                               'ni構eru': 'ni kamaeru 【構える】',
                               '涙wofurutte': 'namida 【涙】 wo furutte',
                               'wo斬ru': 'wo kiru 【斬る】',
                               'no勇gaaru': 'no yuu 【勇】 ga aru',
                               'ni引kkaku': 'ni hikkaku 【引っかく】',
                               '止muwo得nakereba': 'yamu 【止む】 wo enakereba 【得なければ】',
                               'no涙wo注gu': 'no namida 【涙】 wo sosogu 【注ぐ】',
                               'wo全usuru': 'wo mattousuru 【全うする】',
                               '絶/タ/yu': 'tayu 【絶ゆ】',
                               '中no紅一点': 'chuu 【中】 no kouitten 【紅一点】',
                               '眼/me/wo': 'me 【眼】 wo',
                               '贔屓no': 'hiiki 【贔屓】 no',
                               'no死(最期)wo遂geru': 'no shi 【死】 (saigo 【最期】) wo togeru 【遂げる】',
                               'wo願imasu': 'wo negaimasu 【願います】',
                               'ni押shi寄seru': 'ni oshiyoseru 【押し寄せる】',
                               'ni走ru': 'ni hashiru 【走る】',
                               'de暮rasu': 'de kurasu 【暮らす】',
                               'no武器': 'no buki 【武器】',
                               'wo加eru': 'wo kuwaeru 【加える】',
                               'wo張ru': 'wo haru 【張る】',
                               '吹kaseteyaru': 'fukasete 【吹かせて】 yaru',
                               'noii': 'no ii',
                               'nowarui': 'no warui',
                               'hitori見enai(inai)': 'hitori mienai 【見えない】 (inai)',
                               '照ruto': 'teru 【照る】 to',
                               '脱/ヌ/gu': 'nugu 【脱ぐ】',
                               'hoshiidesune': 'hoshii desu ne',
                               '急ide(zatto)': 'isoide 【急いで】 (zatto)',
                               '浴biru': 'abiru 【浴びる】',
                               'toshite降ru': 'to shite furu 【降る】',
                               'no飲食': 'no inshoku 【飲食】',
                               '常/ツネ/ganai': 'tsune 【常】 ga nai',
                               '屁wo': 'onara 【屁】 wo',
                               'ni傚u': 'ni narau 【傚う】',
                               '啼ku': 'naku 【啼く】',
                               'tomoshinai': 'to mo shinai',
                               '水no中wo': 'mizu 【水】 no naka 【中】 wo',
                               '頭wo下geru': 'atama 【頭】 wo sageru 【下げる】',
                               '跳bu': 'tobu 【跳ぶ】',
                               'ni驚ku': 'ni odoroku 【驚く】',
                               'no灯(灯火)': 'no hi 【灯】 (tomoshibi 【灯火】)',
                               'タバコwo': 'TABAKO 【タバコ】 wo',
                               '夜wo': 'yoru 【夜】 wo',
                               'no客tonaru': 'no kyaku 【客】 to naru',
                               '複製': 'fukusei 【複製】',
                               '天地ni愧jizu': 'tenchi 【天地】 ni hajizu 【愧じず】',
                               '天地ni恥zuru所ganai': 'tenchi 【天地】 ni hazuru 【恥ずる】 tokoro 【所】 ga nai',
                               '裏面ni': 'rimen 【裏面】 ni',
                               'nomama検束sareru': 'no mama kensokusareru 【検束される】',
                               'no数': 'no suu 【数】',
                               'ni別reru': 'ni wakareru 【別れる】',
                               'to見rarenai[youna]': 'to mirarenai 【見られない】 [you na]',
                               '足wo': 'ashi 【足】 wo',
                               'ni足wo': 'ni ashi 【足】 wo',
                               '[足no]': '[ashi 【足】 no]',
                               '[足wo]': '[ashi 【足】 wo]',
                               'no法則': 'no housoku 【法則】',
                               'ni付suru': 'ni fusuru 【付する】',
                               'tono': 'to no',
                               '雨ga絶ezu': 'ame 【雨】 ga taezu 【絶えず】',
                               '気ga': 'ki 【気】 ga',
                               'gatsukanai': 'ga tsukanai',
                               'wo生yashiteiru': 'wo hayashite 【生やして】 iru',
                               'no原理': 'no genri 【原理】',
                               '下garu': 'sagaru 【下がる】',
                               'toshite日月no如shi': 'to shite jitsugetsu 【日月】 no gotoshi 【如し】',
                               'ni疲reru': 'ni tsukareru 【疲れる】',
                               'shaberu': 'shaberu',
                               'wohakatte': 'wo hakatte',
                               '用ga': 'you 【用】 ga',
                               'no礼': 'no rei 【礼】',
                               'douzo御': 'douzo go 【御】',
                               '下saimasuyou': 'kudasaimasu 【下さいます】 you',
                               'wo突ku': 'wo tsuku 【突く】',
                               'demenkowosuru': 'de menko wo suru',
                               'no閑': 'no kan 【閑】',
                               'to落chita': 'to ochita 【落ちた】',
                               'to飛bi込mu': 'to tobikomu 【飛び込む】',
                               '太tta': 'futotta 【太った】',
                               'no及bu所denai': 'no oyobu 【及ぶ】 tokoro 【所】 de nai',
                               '折reru': 'oreru 【折れる】',
                               'ga見enai': 'ga mienai 【見えない】',
                               'wosasu': 'wo sasu',
                               'tte': 'tte',
                               'eba': 'eba',
                               '方/koto/naki': 'koto 【方】 naki',
                               '[to]見ru': '[to] miru 【見る】',
                               'ni(kara)': 'ni (kara)',
                               '帽子wo': 'boushi 【帽子】 wo',
                               'ni被ru': 'ni kaburu 【被る】',
                               'notsukanu': 'no tsukanu',
                               'ni寝ru': 'ni neru 【寝る】',
                               'hagitorareru': 'hagitorareru',
                               'no混戦': 'no konsen 【混戦】',
                               'no争i': 'no arasoi 【争い】',
                               'ni暮rasu': 'ni kurasu 【暮らす】',
                               'yatte行ku': 'yatte iku 【行く】',
                               '鰯no': 'iwashi 【鰯】 no',
                               'no郷/サト/': 'no sato 【郷】',
                               'wo立teru': 'wo tateru 【立てる】',
                               'toha': 'to wa',
                               '人nimeimei': 'hito 【人】 ni meimei',
                               'no御宿/オヤド/': 'no oyado 【御宿】',
                               'ni納meru': 'ni osameru 【納める】',
                               'ni打tsu': 'ni utsu 【打つ】',
                               'ni撃tsu': 'ni utsu 【撃つ】',
                               'wo押su': 'wo osu 【押す】',
                               'no一夜wo過gosu': 'no ichiya 【一夜】 wo sugosu 【過ごす】',
                               'wo食waseru': 'wo kuwaseru 【食わせる】',
                               'wo投geru': 'wo nageru 【投げる】',
                               'gakiku': 'ga kiku',
                               '燃e上garu': 'moeagaru 【燃え上がる】',
                               'ano店ha': 'ano mise 【店】 wa',
                               '煙ga': 'kemuri 【煙】 ga',
                               '[to]出ru': '[to] deru 【出る】',
                               'no話de': 'no hanashi 【話】 de',
                               'ni身wo': 'ni mi 【身】 wo',
                               '気no': 'ki 【気】 no',
                               'no幸': 'no sachi 【幸】',
                               '袴no': 'hakama 【袴】 no',
                               'wo高kutoru': 'wo takaku 【高く】 toru',
                               'wo脱gu(脱ideiru)': 'wo nugu 【脱ぐ】 (nuide 【脱いで】 iru)',
                               '効/コウ/naku': 'kou 【効】 naku',
                               'woafuru': 'wo afuru',
                               '八幡/yawata/no': 'yawata 【八幡】 no',
                               'no大蛇/オロチ/': 'no orochi 【大蛇】',
                               'ano男ha何事mo': 'ano otoko 【男】 wa nanigoto 【何事】 mo',
                               'niataru': 'ni ataru',
                               'mo忘renai': 'mo wasurenai 【忘れない】',
                               'wo作ru': 'wo tsukuru 【作る】',
                               'desu(da)': 'desu (da)',
                               '[御座imasu]': '[gozaimasu 【御座います】]',
                               'no事de': 'no koto 【事】 de',
                               'nikakaeru': 'ni kakaeru',
                               'ni飛ndeyuku': 'ni tonde 【飛んで】 yuku',
                               'ni払u': 'ni harau 【払う】',
                               'aru人': 'aru hito 【人】',
                               'wo漏rasu': 'wo morasu 【漏らす】',
                               'kono本ha': 'kono hon 【本】 wa',
                               'no捨小舟/ステコブネ/': 'no sutekobune 【捨小舟】',
                               '[wo通tte]': '[wo tootte 【通って】]',
                               'no余地monai': 'no yochi 【余地】 mo nai',
                               'no机': 'no tsukue 【机】',
                               'to呼鈴wo鳴rasu': 'to yobirin 【呼鈴】 wo narasu 【鳴らす】',
                               '人間ha万物': 'ningen 【人間】 wa banbutsu 【万物】',
                               'no身': 'no mi 【身】',
                               'wo結bu': 'wo musubu 【結ぶ】',
                               '震eru': 'furueru 【震える】',
                               'no紙': 'no kami 【紙】',
                               'no話shiburi': 'no hanashiburi 【話しぶり】',
                               'no塔': 'no tou 【塔】',
                               'no灰/ハイ/': 'no hai 【灰】',
                               'ni持chi込mu': 'ni mochikomu 【持ち込む】',
                               '指導教師': 'shidou 【指導】 kyoushi 【教師】',
                               'kamo': 'ka mo',
                               'beshi': 'beshi',
                               '間違tteiru': 'machigatte 【間違って】 iru',
                               'o出denasai, o出de下sai': 'oide 【お出で】 nasai, oide 【お出で】 kudasai 【下さい】',
                               '正金銀行': 'shoukin 【正金】 ginkou 【銀行】',
                               '特別委員会': 'tokubetsu 【特別】 iinkai 【委員会】',
                               '特殊飲食店街': 'tokushu 【特殊】 inshoku 【飲食】 tengai 【店街】',
                               'モスリン': 'MOSURIN 【モスリン】',
                               'タクシ:': 'TAKUSHII 【タクシー】',
                               'アジ': 'AJI 【アジ】',
                               '海軍': 'kaigun 【海軍】',
                               '空軍': 'kuugun 【空軍】',
                               '陸軍': 'rikugun 【陸軍】',
                               '原子爆弾': 'genshi 【原子】 bakudan 【爆弾】',
                               '水素爆弾': 'suiso 【水素】 bakudan 【爆弾】',
                               '国民党': 'kokumintou 【国民党】',
                               '共産党': 'kyousantou 【共産党】',
                               '修正': 'shuusei 【修正】',
                               '校定': 'koutei 【校定】',
                               '時間': 'jikan 【時間】',
                               '空間': 'kuukan 【空間】',
                               '汽船': 'kisen 【汽船】',
                               '軍艦': 'gunkan 【軍艦】',
                               '早稲田': 'waseda 【早稲田】',
                               '慶応': 'keiou 【慶応】',
                               '理髪': 'rihatsu 【理髪】',
                               '美容': 'biyou 【美容】',
                               '礼儀': 'reigi 【礼儀】',
                               '音楽': 'ongaku 【音楽】',
                               '労働組合連合会': 'roudou 【労働】 kumiai 【組合】 rengoukai 【連合会】',
                               '労働組合連盟': 'roudou 【労働】 kumiai 【組合】 renmei 【連盟】',
                               'goneru': 'goneru',
                               '得': 'toku 【得】',
                               '青馬no節会/セチエ/': 'aouma 【青馬】 no sechie 【節会】',
                               '安全保障': 'anzen 【安全】 hoshou 【保障】',
                               '安全保障条約': 'anzen 【安全】 hoshou 【保障】 jouyaku 【条約】',
                               '言ukoto': 'iu 【言う】 koto',
                               '医学大学': 'igaku 【医学】 daigaku 【大学】',
                               '医学博士': 'igaku 【医学】 hakase 【博士】',
                               '委員会': 'iinkai 【委員会】',
                               '印度綿花': 'indo 【印度】 menka 【綿花】',
                               '運転休止': 'unten 【運転】 kyuushi 【休止】',
                               '英国資本': 'eikoku 【英国】 shihon 【資本】',
                               'o茶no水大学': 'ochanomizu 【お茶の水】 daigaku 【大学】',
                               '巡洋艦': "jun'youkan 【巡洋艦】",
                               '化学繊維': "kagaku 【化学】 sen'i 【繊維】",
                               '外国語大学': 'gaikokudo 【外国語】 daigaku 【大学】',
                               '僧伽藍摩': 'sougaranma 【僧伽藍摩】',
                               '機関銃': 'kikaijuu 【機関銃】',
                               '多伽羅': 'takara 【多伽羅】',
                               '旧暦': 'kyuureki 【旧暦】',
                               '九州大学': 'kyuushuu 【九州】 daigaku 【大学】',
                               '至急電報': 'shikyuu 【至急】 denpou 【電報】',
                               '共同漁業権': 'kyoudou 【共同】 gyogyouken 【漁業権】',
                               '京都大学': 'kyouto 【京都】 daigaku 【大学】',
                               '機械水雷': 'kikai 【機械】 suirai 【水雷】',
                               '空中爆撃': 'kuuchuu 【空中】 bakugeki 【爆撃】',
                               '航空母艦': 'koukuu 【航空】 bokan 【母艦】',
                               '空中回廊': 'kuuchuu 【空中】 kairou 【回廊】',
                               '九九no表': 'kuku 【九九】 no hyou 【表】',
                               '区域行政': 'kuiki 【区域】 gyousei 【行政】',
                               '軍備拡張': 'gunbi 【軍備】 kakuchou 【拡張】',
                               '軍事警察': 'gunji 【軍事】 keisatsu 【警察】',
                               '軍備縮小': 'gunbi 【軍備】 shukushou 【縮小】',
                               '軽機関銃': 'keikikanjuu 【軽機関銃】',
                               '軽飛行機': 'keihikouki 【軽飛行機】',
                               '刑事訴訟': 'keiji 【刑事】 soshou 【訴訟】',
                               '刑事訴訟法': 'keiji 【刑事】 soshouhou 【訴訟法】',
                               '結跏趺坐': 'kekkafuza 【結跏趺坐】',
                               '赤血球沈降速度': 'sekkekkyuu 【赤血球】 chinkou 【沈降】 sokudo 【速度】',
                               '県会議員': 'kenkai 【県会】 giin 【議員】',
                               '事件名称索引': 'jiken 【事件】 meishou 【名称】 sakuin 【索引】',
                               '現物no株式': 'genbutsu 【現物】 no kabushiki 【株式】',
                               '原子力研究': 'genshiryoku 【原子力】 kenkyuu 【研究】',
                               '原子力研究所': 'genshiryoku 【原子力】 kenkyuujo 【研究所】',
                               '現在no住職': 'genzai 【現在】 no juushoku 【住職】',
                               '現物提供': 'genbutsu 【現物】 teikyou 【提供】',
                               '公定価格': 'koutei 【公定】 kakaku 【価格】',
                               '銀行no金': 'ginkou 【銀行】 no kane 【金】',
                               '興業銀行': 'kougyou 【興業】 ginkou 【銀行】',
                               '高等工業学校': 'koutou 【高等】 kougyou 【工業】 gakkou 【学校】',
                               '高等学校生徒': 'koutou 【高等】 gakkou 【学校】 seito 【生徒】',
                               '高等裁判所': 'koutou 【高等】 saibansho 【裁判所】',
                               '高等師範学校': 'koutou 【高等】 shihan 【師範】 gakkou 【学校】',
                               '公債to社債': 'kousai 【公債】 + shasai 【社債】',
                               '高等商業学校': 'koutou 【高等】 shougyou 【商業】 gakkou 【学校】',
                               '高等専門学校': 'koutou 【高等】 senmon 【専門】 gakkou 【学校】',
                               '高等専門学校卒業': 'koutou 【高等】 senmon 【専門】 gakkou 【学校】 sotsugyou 【卒業】',
                               '工業大学': 'kougyou 【工業】 daigaku 【大学】',
                               '公家武家': 'kuge 【公家】 + buke 【武家】',
                               '高等文官': 'koutou 【高等】 bunkan 【文官】',
                               '高等文官試験': 'koutou 【高等】 bunkan 【文官】 shiken 【試験】',
                               '国民政府': 'kokumin 【国民】 seifu 【政府】',
                               '小口当座預金': 'koguchi 【小口】 touza 【当座】 yokin 【預金】',
                               '国家管理': 'kokka 【国家】 kanri 【管理】',
                               '国家宗教': 'kokka 【国家】 shuukyou 【宗教】',
                               '国家地方警察': 'kokka 【国家】 chihou 【地方】 keisatsu 【警察】',
                               '識見to度量': 'shikiken 【識見】 + doryou 【度量】',
                               '市会議': 'shikaigi 【市会議】',
                               '市会議員': 'shikaigiin 【市会議員】',
                               '市中銀行': 'shichuu 【市中】 ginkou 【銀行】',
                               '私立大学': 'shiritsu 【私立】 daigaku 【大学】',
                               '師団長': 'shidanchou 【師団長】',
                               '輜重部隊': 'shichou 【輜重】 butai 【部隊】',
                               '執行委員会': 'shikkou 【執行】 iinkai 【委員会】',
                               '市内電車': 'shinai 【市内】 densha 【電車】',
                               '支局to分局': 'shikyoku 【支局】 + bunkyoku 【分局】',
                               '衆議院to参議院': 'shuugiin 【衆議院】 + sangiin 【参議院】',
                               '秋季闘争': 'shuuki 【秋季】 tousou 【闘争】',
                               '輸出超過': 'yushutsu 【輸出】 chouka 【超過】',
                               '主婦連合会': 'shufu 【主婦】 rengoukai 【連合会】',
                               '春季闘争': 'shunki 【春季】 tousou 【闘争】',
                               '商業大学': 'shougyou 【商業】 daigaku 【大学】',
                               '公共職業安定所': 'koukyou 【公共】 shokugyouanteijo 【職業安定所】',
                               '進学適正検査': 'shingaku 【進学】 tekisei 【適正】 kensa 【検査】',
                               '自治体警察': 'jichitai 【自治体】 keisatsu 【警察】',
                               '機関銃部隊': 'kikanjuu 【機関銃】 butai 【部隊】',
                               '受信報知電報': 'jushin 【受信】 houchi 【報知】 denpou 【電報】',
                               '純粋系統': 'junsui 【純粋】 keitou 【系統】',
                               '通常国会': 'tsuujou 【通常】 kokkai 【国会】',
                               '女子高等師範学校': 'joshi 【女子】 koutou 【高等】 shihan 【師範】 gakkou 【学校】',
                               '人造絹糸': 'jinzou 【人造】 kenshi 【絹糸】',
                               '人糞肥料': 'jinpun 【人糞】 hiryou 【肥料】',
                               '人造肥料': 'jinzou 【人造】 hiryou 【肥料】',
                               '水上飛行機': 'suijou 【水上】 hikouki 【飛行機】',
                               '青年訓練': 'seinen 【青年】 kunren 【訓練】',
                               '政治結社': 'seiji 【政治】 kessha 【結社】',
                               '生活必需品': 'seikatsu 【生活】 hitsujuuhin 【必需品】',
                               '生命保険': 'seimei 【生命】 hoken 【保険】',
                               '世界銀行': 'sekai 【世界】 ginkou 【銀行】',
                               '戦争犯罪': 'sensou 【戦争】 hanzai 【犯罪】',
                               '放射線量': 'housha 【放射】 senryou 【線量】',
                               '早稲田大学': 'waseda 【早稲田】 daigaku 【大学】',
                               '日本労働総同盟': 'nihon 【日本】 roudou 【労働】 soudoumei 【総同盟】',
                               '日本労働組合総評議会': 'nihon 【日本】 roudou 【労働】 kumiai 【組合】 souhyou 【総評】 gikai 【議会】',
                               '短期大学': 'tanki 【短期】 daigaku 【大学】',
                               'dato言tte': 'da to itte 【言って】',
                               '堕落幹部': 'daraku 【堕落】 kanbu 【幹部】',
                               '団体協約': 'dantai 【団体】 kyouyaku 【協約】',
                               '団体交渉': 'dantai 【団体】 koushou 【交渉】',
                               '集団地域': 'shuudan 【集団】 chiiki 【地域】',
                               '地下核爆発停止': 'chika 【地下】 kakubakuhatsu 【核爆発】 teishi 【停止】',
                               '地方警察': 'chihou 【地方】 keisatsu 【警察】',
                               '地方検事局': 'chihou 【地方】 kenjikyoku 【検事局】',
                               '地方裁判所': 'chihou 【地方】 saibansho 【裁判所】',
                               '中央委員会': 'chuuou 【中央】 iinkai 【委員会】',
                               '中央委員[会]': 'chuuou 【中央】 iin[kai] 【委員[会]】',
                               '中国共産党': 'chuugoku 【中国】 kyousantou 【共産党】',
                               '中央執行委員会': 'chuuou 【中央】 shikkou 【執行】 iinkai 【委員会】',
                               '中央闘争委員会': 'chuuou 【中央】 tousou 【闘争】 iinkai 【委員会】',
                               '中央労働委員会': 'chuuou 【中央】 roudou 【労働】 iinkai 【委員会】',
                               '賃上ge闘争': "chin'age 【賃上げ】 tousou 【闘争】",
                               '通商産業省': 'tsuushou 【通商】 sangyoushou 【産業省】',
                               '通商産業大臣': 'tsuushou 【通商】 sangyou 【産業】 daijin 【大臣】',
                               '帝国大学': 'teikoku 【帝国】 daigaku 【大学】',
                               '電気分解': 'denki 【電気】 bunkai 【分解】',
                               '電気機械': 'denki 【電気】 kikai 【機械】',
                               '電気器具': 'denki 【電気】 kigu 【器具】',
                               '電気蓄音機': 'denki 【電気】 chikuonki 【蓄音機】',
                               '電気鉄道': 'denki 【電気】 tetsudou 【鉄道】',
                               '電気回路': 'denki 【電気】 kairo 【回路】',
                               '東京大学': 'toukyou 【東京】 daigaku 【大学】',
                               '特殊銀行': 'tokushu 【特殊】 ginkou 【銀行】',
                               '特高警察': 'tokkou 【特高】 keisatsu 【警察】',
                               'to言tte': 'to itte 【言って】',
                               '都内電車': 'tonai 【都内】 densha 【電車】',
                               '土木建築': 'doboku 【土木】 kenchiku 【建築】',
                               'nanakusanosekku【七草no節句】': 'nanakusa 【七草】 no sekku 【節句】',
                               '生意気': 'namaiki 【生意気】',
                               '生ビ:ル': 'nama 【生】 BIIRU 【ビール】',
                               '南無阿弥陀仏/ナムアミダブツ/': 'namu 【南無】 amidabutsu 【阿弥陀仏】',
                               '日本銀行': 'nihon 【日本】 ginkou 【銀行】',
                               '日本大学': 'nihon 【日本】 daigaku 【大学】',
                               '日本共産党': 'nihon 【日本】 kyousantou 【共産党】',
                               '日本航空': 'nihon 【日本】 koukuu 【航空】',
                               '輸入超過': 'yunyuu 【輸入】 chouka 【超過】',
                               '農業協同組合': 'nougyou 【農業】 kyoudou 【協同】 kumiai 【組合】',
                               '四国八十八個所': 'shikoku 【四国】 hachijuuhakkasho 【八十八個所】',
                               '発売禁止': 'hatsubai 【発売】 kinshi 【禁止】',
                               '万国博覧会': 'bankoku 【万国】 hakurankai 【博覧会】',
                               '風力程度': 'fuuryoku 【風力】 teido 【程度】',
                               '府会議員': 'fukaigiin 【府会議員】',
                               '婦人警官': 'fujin 【婦人】 keikan 【警官】',
                               '婦人選挙権': 'fujin 【婦人】 senkyoken 【選挙権】',
                               '普通選挙': 'futsuu 【普通】 senkyo 【選挙】',
                               '物資動員': 'busshi 【物資】 douin 【動員】',
                               '文部省検定試験': 'monbushou 【文部省】 kentei 【検定】 shiken 【試験】',
                               '文学博士': 'bungaku 【文学】 hakase 【博士】',
                               '砲兵to工兵': 'houhei 【砲兵】 + kouhei 【工兵】',
                               '貿易手形': 'boueki 【貿易】 tegata 【手形】',
                               '南満鉄道': 'nanman 【南満】 tetsudou 【鉄道】',
                               '居留民団': 'kyoryuumindan 【居留民団】',
                               '無線電信': 'musen 【無線】 denshin 【電信】',
                               '同盟休校': 'doumei 【同盟】 kyuukou 【休校】',
                               '明治大学': 'meiji 【明治】 daigaku 【大学】',
                               '名誉to利益': 'meiyo 【名誉】 + rieki 【利益】',
                               '落書no一首': 'rakusho 【落書】 no isshu 【一首】',
                               '陸軍大学校': 'rikugun 【陸軍】 daigakkou 【大学校】',
                               '流行性感冒': 'ryuukousei 【流行性】 kanbou 【感冒】',
                               '臨時急行': 'rinji 【臨時】 kyuukou 【急行】',
                               '労働組合': 'roudou 【労働】 kumiai 【組合】',
                               'ブランケット': 'BURANKETTO 【ブランケット】',
                               'サリチル酸・ソ:ダ': 'SARICHIRUsan SOUDA 【サリチル酸・ソーダ】',
                               'ストライキ': 'SUTORAIKI 【ストライキ】',
                               'ソヴィエト共産党': 'SOVIETO 【ソヴィエト】 kyousantou 【共産党】',
                               'クレ:プ・デシン': 'KUREEPU DESHIN 【クレープ・デシン】',
                               'ミステ:ク': 'MISUTEEKU 【ミステーク】',
                               'ラグビ': 'RAGUBI 【ラグビ】',
                               '入国管理事務所': 'nyuukoku 【入国】 kanri 【管理】 jimusho 【事務所】',
                               'nihiku': 'ni hiku',
                               'ni置ku': 'ni oku 【置く】',
                               '頰wo': 'hoho 【頰】 wo',
                               '頰ga': 'hoho 【頰】 ga',
                               'gatorenai': 'ga torenai',
                               'ni取rareru': 'ni torareru 【取られる】',
                               'aite': 'aite',
                               '息wo': 'iki 【息】 wo',
                               'no目wo抜ku[youna事wosuru]': 'no 【目】 wo nuku 【抜く】 [you na koto 【事】 wo suru]',
                               'no煙': 'no kemuri 【煙】',
                               'no命': 'no inochi 【命】',
                               'no望miwo抱ku': 'no nozomi 【望み】 wo idaku 【抱く】',
                               'no労/ロウ/': 'no rou 【労】',
                               'ni過ginai': 'ni suginai 【過ぎない】',
                               'narazu': 'narazu',
                               '[nokoto]': '[no koto]',
                               'ga尽kita': 'ga tsukita 【尽きた】',
                               'na奴da': 'na yatsu 【奴】 da',
                               '言tteiru': 'itte 【言って】 iru',
                               'no魚': 'no sakana 【魚】',
                               'no料理': 'no ryouri 【料理】',
                               'ni掛keru': 'ni kakeru 【掛ける】',
                               '心wo': 'kokoro 【心】 wo',
                               'nikakeru': 'ni kakeru',
                               'ni止maru': 'ni tomaru 【止まる】',
                               'ni過ginu': 'ni suginu 【過ぎぬ】',
                               'ni嵌ru': 'ni hamaru 【嵌る】',
                               '第一ni': 'daiichi 【第一】 ni',
                               'to書itearu': 'to kaite 【書いて】 aru',
                               'no偉業(事業)': 'no igyou 【偉業】 (jigyou 【事業】)',
                               'no力': 'no chikara 【力】',
                               'kachin': 'kachin',
                               '晴re上garu': 'hareagaru 【晴れ上がる】',
                               '揚geru': 'ageru 【揚げる】',
                               '日ga照tteiru': 'me 【日】 ga tette 【照って】 iru',
                               'ni怒ru': 'ni okoru 【怒る】',
                               '三': 'san 【三】',
                               'iu音': 'iu oto 【音】',
                               'to飲mi込mu': 'to nomikomu 【飲み込む】',
                               'to嚙mi付ku': 'to kamitsuku 【嚙み付く】',
                               '騒gu': 'sawagu 【騒ぐ】',
                               'to音gasuru': 'to oto 【音】 ga suru',
                               '嚙mu': 'kamu 【嚙む】',
                               '鳴rasu': 'narasu 【鳴らす】',
                               'no字': 'no ji 【字】',
                               'no祝': 'no iwai 【祝】',
                               'nonakiwo得nai': 'no naki wo enai 【得ない】',
                               'wo現wasu': 'wo arawasu 【現わす】',
                               'wo没suru': 'wo bossuru 【没する】',
                               'ga高i': 'ga takai 【高い】',
                               '気儘na': 'kimama 【気儘】 na',
                               'ni詰matteiru': 'ni tsumatte 【詰まって】 iru',
                               'no字no祝/イワイ/': 'no ji 【字】 no iwai 【祝】',
                               'ga折reru': 'ga oreru 【折れる】',
                               'no折reru': 'no oreru 【折れる】',
                               '言waseru': 'iwaseru 【言わせる】',
                               '参ru': 'mairu 【参いる】',
                               '鳴ru音': 'naru 【鳴る】 oto 【音】',
                               'to言u音': 'to iu 【言う】 oto 【音】',
                               'to開ku': 'to hiraku 【開く】',
                               '疲reru': 'tsukareru 【疲れる】',
                               '汗deカラ:ga': 'ase 【汗】 de KARAA 【カラー】 ga',
                               '目ga': 'me 【目】 ga',
                               '煮e立tsu': 'nietatsu 【煮え立つ】',
                               '空no色': 'sora 【空】 no iro 【色】',
                               '白百合/シラユリ/': 'shirayuri 【白百合】',
                               '嗅gu': 'kagu 【嗅ぐ】',
                               '引ku': 'hiku 【引く】',
                               '寝ru': 'neru 【寝る】',
                               '腹ga': 'hara 【腹】 ga',
                               'mo出nai': 'mo denai 【出ない】',
                               'mo出naiyounisuru': 'mo denai 【出ない】 you ni suru',
                               '回ru': 'mawaru 【回る】',
                               '巻ku': 'maku 【巻く】',
                               '巻kinisuru': 'maki 【巻き】 ni suru',
                               'ga付ku': 'ga tsuku 【付く】',
                               'no勝負woyaru': 'no shoubu 【勝負】 wo yaru',
                               'wo食wasu': 'wo kuwasu 【食わす】',
                               'wo食wasaseru': 'wo kuwasaseru 【食わさせる】',
                               'no気': 'no ki 【気】',
                               'no気wo養u': 'no ki 【気】 wo yashinau 【養う】',
                               'no生活': 'no seikatsu 【生活】',
                               'wo持/ジ/su': 'wo jisu 【持す】',
                               'suruwo': 'suru wo',
                               '転geru': 'korogeru 【転げる】',
                               'ano人ha': 'ano hito 【人】 wa',
                               'tofutotteiru': 'to futotte iru',
                               '狐ga': 'kitsune 【狐】 ga',
                               'to咳wosuru': 'to seki 【咳】 wo suru',
                               '雪ga': 'yuki 【雪】 ga',
                               'to降ru': 'to furu 【降る】',
                               'to流reru': 'to nagareru 【流れる】',
                               'toshite尽kinai': 'to shite tsukinai 【尽きない】',
                               'to通ru': 'to tooru 【通る】',
                               'toiu彼no音': 'to iu kare 【彼】 no oto 【音】',
                               'toiu音': 'to iu oto 【音】',
                               'no阿蒙/アモウ/dearu': 'no amou 【阿蒙】 dearu',
                               '酔iwo': 'yoi 【酔い】 wo',
                               '迷i(迷夢)wo': 'mayoi 【迷い】 (meimu 【迷夢】) wo',
                               '[目ga]': '[me 【目】 ga]',
                               '目no': 'me 【目】 no',
                               'youna美人': 'you na bijin 【美人】',
                               '酔iga': 'yoi 【酔い】 ga',
                               '砂wo': 'suna 【砂】 wo',
                               '踏mu': 'fumu 【踏む】',
                               '降rino雨': 'furi 【降り】 no ame 【雨】',
                               'no花': 'no hana 【花】',
                               'no眺megaaru': 'no nagame 【眺め】 ga aru',
                               '痛mu': 'itamu 【痛む】',
                               'no旅': 'no tabi 【旅】',
                               'no旅ni上ru': 'no tabi 【旅】 ni noboru 【上る】',
                               'no山': 'no yama 【山】',
                               'no人qi': 'no hitobito 【人々】',
                               '体wo': 'karada 【体】 wo',
                               '一致shite': 'icchishite 【一致して】',
                               'ni掌握suru': 'ni shouakusuru 【掌握する】',
                               'wo脱suru': 'wo dassuru 【脱する】',
                               '神mo': 'kami 【神】 mo',
                               'are': 'are',
                               'ni濡reru': 'ni nureru 【濡れる】',
                               '雨ga': 'ame 【雨】 ga',
                               '降ru': 'furu 【降る】',
                               'shita目': 'shita me 【目】',
                               'shita髯': 'shita hige 【髯】',
                               'no見ru所': 'no miru 【見る】 tokoro 【所】',
                               'soreha': 'sore wa',
                               'no一失': 'no isshitsu 【一失】',
                               'no弁/ベン/': 'no ben 【弁】',
                               'no弁wo振u': 'no ben 【弁】 wo furuu 【振う】',
                               'taru大海': 'taru taikai 【大海】',
                               'to暮rete行ku': 'to kurete 【暮れて】 iku 【行く】',
                               'ni載seru': 'ni noseru 【載せる】',
                               '耳wo': 'mimi 【耳】 wo',
                               'wo食tte殺sareru': 'wo kutte 【食って】 korosareru 【殺される】',
                               'abunai': 'abunai',
                               '誰': 'dare 【誰】',
                               '来tanoka？': 'kita 【来た】 no ka?',
                               'ni出su': 'ni dasu 【出す】',
                               'de売ru': 'de uru 【売る】',
                               'ni近i': 'ni chikai 【近い】',
                               'wo得/エ/ru': 'wo eru 【得る】',
                               'jiisantobaasan': 'jii-san to baa-san',
                               'de寝ru': 'de neru 【寝る】',
                               '世間知razuno': 'seken 【世間】 shirazu 【知らず】 no',
                               'wo憚ru': 'wo habakaru 【憚る】',
                               'wo許sanai': 'wo yurusanai 【許さない】',
                               'no一生wo送ru': 'no issei 【一生】 wo okuru 【送る】',
                               'no志士': 'no shishi 【志士】',
                               'no悪i戸': 'no warui 【悪い】 to 【戸】',
                               '上kara水ga': 'ue 【上】 kara mizu 【水】 ga',
                               '汗wo': 'ase 【汗】 wo',
                               '流shiteiru': 'nagashite 【流して】 iru',
                               'wo吹ku': 'wo fuku 【吹く】',
                               'wo吹ku人': 'wo fuku 【吹く】 hito 【人】',
                               'no交ri(契ri)': 'no majiri 【交り】 (chigiri 【契り】)',
                               'to刺su': 'to sasu 【刺す】',
                               'to痛mu': 'to itamu 【痛む】',
                               'to啼ku': 'to naku 【啼く】',
                               '無用no': 'muyou 【無用】 no',
                               '視suru': 'shisuru 【視する】',
                               'hasami切ru': 'hasami kiru 【切る】',
                               '切ru': 'kiru 【切る】',
                               'to立tsu': 'to tatsu 【立つ】',
                               '犬ga': 'inu 【犬】 ga',
                               '今月ha': 'kongetsu 【今月】 wa',
                               '嘘wo': 'uso 【嘘】 wo',
                               '喋beru': 'shaberu 【喋べる】',
                               '返答suru': 'hentousuru 【返答する】',
                               '云wazuni': 'iwazu 【云わず】 ni',
                               '言una': 'iu 【言う】 na',
                               '茶no': 'cha 【茶】 no',
                               '繭no': 'mayu 【繭】 no',
                               '一番ninaru': 'ichiban 【一番】 ni naru',
                               '一着': 'icchaku 【一着】',
                               'ni暮reru': 'ni kureru 【暮れる】',
                               'ni暮resaseru': 'ni kuresaseru 【暮れさせる】',
                               'monaiuso': 'mo nai uso',
                               '[no友]': '[no tomo 【友】]',
                               '言wazuni': 'iwazu 【言わず】 ni',
                               '言tte承諾shinai': 'itte 【言って】 shoudakushinai 【承諾しない】',
                               '腰wo下rosu': 'koshi 【腰】 wo orosu 【下ろす】',
                               '相場ga': 'souba 【相場】 ga',
                               '下gatta': 'sagatta 【下がった】',
                               '胸wo': 'mune 【胸】 wo',
                               '衝ku': 'tsuku 【衝く】',
                               'wo抜kareru': 'wo nukareru 【抜かれる】',
                               'no魚wo漏rasu': 'no sakana 【魚】 wo morasu 【漏らす】',
                               '顔woshite': 'kao 【顔】 wo shite',
                               '顔de': 'kao 【顔】 de',
                               'ni漂u': 'ni tadayou 【漂う】',
                               'wokaikuguru': 'wo kaikuguru',
                               '行kanai': 'ikanai 【行かない】',
                               '動kenai': 'ugokenai 【動けない】',
                               'ga継genai': 'ga tsugenai 【継げない】',
                               'wo継genakusuru': 'wo tsugenaku 【継げなく】 suru',
                               'kokoha': 'koko wa',
                               'noyoi床': 'no yoi yuka 【床】',
                               'no良i靴': 'no ii 【良い】 kutsu 【靴】',
                               'no一': 'no ichi 【一】',
                               '打込mu刀wo': 'uchikomu 【打込む】 yaiba 【刀】 wo',
                               '受ke止meru': 'uketomeru 【受け止める】',
                               '切ritsukeru': 'kiritsukeru 【切りつける】',
                               'desuka': 'desu ka',
                               '失礼desuga': 'shitsurei 【失礼】 desu ga',
                               'doumo': 'doumo',
                               'desuga': 'desu ga',
                               'wo憚ru事gara': 'wo habakaru 【憚る】 kotogara 【事がら】',
                               '空ni': 'sora 【空】 ni',
                               '[no笛]': '[no fue 【笛】]',
                               'no策': 'no saku 【策】',
                               '敵軍ni': 'tekigun 【敵軍】 ni',
                               'wo放tsu': 'wo hanatsu 【放つ】',
                               '苦肉no計': 'kuniku 【苦肉】 no kei 【計】',
                               '息wokirashite': 'iki 【息】 wo kirashite',
                               'no縄': 'no nawa 【縄】',
                               'nitsuku': 'ni tsuku',
                               'ikura言ttemo': 'ikura itte 【言って】 mo',
                               'ni聞ki流su': 'ni kikinagasu 【聞き流す】',
                               '[to]ドアwo締meru': '[to] DOA 【ドア】 wo shimeru 【締める】',
                               '水ni': 'mizu 【水】 ni',
                               '[to]落chiru': '[to] ochiru 【落ちる】',
                               'wo合waseru': 'wo awaseru 【合わせる】',
                               '撒ku': 'maku 【撒く】',
                               '仕事wo': 'shigoto 【仕事】 wo',
                               'ni通jiteiru人': 'ni tsuujite 【通じて】 iru hito 【人】',
                               'no気焔wo上geru': 'no kien 【気焔】 wo ageru 【上げる】',
                               '波瀾': 'haran 【波瀾】',
                               'datta時代': 'datta jidai 【時代】',
                               'no位ni上garu': 'no i 【位】 ni agaru 【上がる】',
                               'no君/キミ/': 'no kimi 【君】',
                               '食beru': 'taberu 【食べる】',
                               '吹kasu': 'fukasu 【吹かす】',
                               'to食beru': 'to taberu 【食べる】',
                               'to開ite': 'to aite 【開いて】',
                               '帆ga風ni': 'ho 【帆】 ga kaze 【風】 ni',
                               '翼wo': 'tsubasa 【翼】 wo',
                               '雨no': 'ame 【雨】 no',
                               'iu音gasuru': 'iu oto 【音】 ga suru',
                               'hokoriwo': 'hokori wo',
                               '払u': 'harau 【払う】',
                               '打tsu': 'utsu 【打つ】',
                               '指de': 'yubi 【指】 de',
                               'hajiku': 'hajiku',
                               '小銃no': 'shoujuu 【小銃】 no',
                               '手wo': 'te 【手】 wo',
                               '叩ku': 'tataku 【叩く】',
                               '嚙mitsuku': 'kamitsuku 【嚙みつく】',
                               'akeru': 'akeru',
                               '刀wo': 'yaiba 【刀】 wo',
                               'sayani納meru': 'saya ni osameru 【納める】',
                               '雨': 'ame 【雨】',
                               '木no葉ga': 'konoha 【木の葉】 ga',
                               '散tta': 'chitta 【散った】',
                               'kono部屋ha': 'kono heya 【部屋】 wa',
                               'no好i': 'no ii 【好い】',
                               'wo取tta女': 'wo totta 【取】 onna 【女】',
                               'wo命zuru': 'wo meizuru 【命ずる】',
                               'wo願u': 'wo negau 【願う】',
                               'ga好i': 'ga ii 【好い】',
                               '本wo閉jiru': 'hon 【本】 wo tojiru 【閉じる】',
                               '汗wokaku': 'ase 【汗】 wo kaku',
                               'シャツga汗de': 'SHATSU 【シャツ】 ga ase 【汗】 de',
                               '裂ku': 'saku 【裂く】',
                               '縫目wo': 'nuime 【縫目】 wo',
                               '解ku': 'toku 【解く】',
                               'to震動suru': 'to shindousuru 【震動する】',
                               '往復': 'oufuku 【往復】',
                               '動ku': 'ugoku 【動く】',
                               '石wo打tsu': 'ishi 【石】 wo utsu 【打つ】',
                               'muchide打tsu': 'muchi de utsu 【打つ】',
                               'no仇/アダ/': 'no ada 【仇】',
                               'no恨miwoidaku': 'no urami 【恨み】 wo idaku',
                               'no態度wo持suru(執ru)': 'no taido 【態度】 wo jisuru 【持する】 (toru 【執る】)',
                               'no間柄ninaru': 'no aidagara 【間柄】 ni naru',
                               '思i切ru': 'omoikiru 【思い切る】',
                               'to手wo切ru': 'to te 【手】 wo kiru 【切る】',
                               '柔kakute': 'yawarakakute 【柔かくて】',
                               'ha頭ga': 'wa atama 【頭】 ga',
                               '立chi上garu': 'tachiagaru 【立ち上がる】',
                               'ha足ga': 'wa ashi 【足】 ga',
                               'no雨': 'no ame 【雨】',
                               'ni降ru': 'ni furu 【降る】',
                               '香気': 'kouki',
                               '臭気': 'shuuki 【臭気】',
                               '鼻wotsuku': 'hana 【鼻】 wo tsuku',
                               'tsutanaku死nu': 'tsutanaku shinu 【死ぬ】',
                               '云u': 'iu 【云う】',
                               '云u人': 'iu 【云う】 hito 【人】',
                               '煮eru': 'nieru 【煮える】',
                               'no間ni': 'no ma 【間】 ni',
                               'no間ni馳駆suru': 'no ma 【間】 ni chikusuru 【馳駆する】',
                               'taru小才子': 'taru kozaishi 【小才子】',
                               'toshite風ni翻ru': 'to shite kaze 【風】 ni hirugaeru 【翻る】',
                               'to坐ru': 'to suwaru 【坐る】',
                               'totsuku': 'to tsuku',
                               'o腹ga': 'onaka 【お腹】 ga',
                               'o辞儀wosuru': 'ojigi 【お辞儀】 wo suru',
                               'to凹mu': 'to hekomu 【凹む】',
                               '坐ru': 'suwaru 【坐る】',
                               '切手wo': 'kitte 【切手】 wo',
                               '張ru': ' haru 【張る】',
                               'tonameru': 'to nameru',
                               'to舌wo出su': 'to shita 【舌】 wo dasu 【出す】',
                               'to平rageru': 'to tairageru 【平らげる】',
                               'shitai': 'shitai',
                               '言itai': 'iitai 【言いたい】',
                               'nokotowo言u': 'no koto wo iu 【言う】',
                               '食i': 'kui 【食い】',
                               'ni食u': 'ni kuu 【食う】',
                               'no体/テイ/de': 'no tei 【体】 de',
                               'makkani顔wo': 'makka ni kao 【顔】 wo',
                               'sete': 'sete',
                               '体中ga': 'karadajuu 【体中】 ga',
                               'ni陣取ru': 'ni jindoru 【陣取る】',
                               'wo下garu': 'wo sagaru 【下がる】',
                               'no勇': 'no yuu 【勇】',
                               '湧ki出ru': 'wakideru 【湧き出る】',
                               '噛ru': 'kajiru 【噛る】',
                               '掻ku': 'kaku 【掻く】',
                               '本wo傍ni': 'hon 【本】 wo soba 【傍】 ni',
                               '投geru': 'nageru 【投げる】',
                               'tonaguru': 'to naguru',
                               'to明ku': 'to aku 【明く】',
                               '指no節wo': 'yubi 【指】 no fushi 【節】 wo',
                               '滴ru': 'shitataru 【滴る】',
                               '汽車ga': 'kisha 【汽車】 ga',
                               '煙wo吐kinagara停車場wo出ta':
                                   'kemuri 【煙】 wo hakinagara 【吐きながら】 teishajou 【停車場】 wo deta 【出た】',
                               '湯気no出teiruホット・ケ:キ': 'yuge 【湯気】 no dete 【出て】 iru HOTTO KEEKI 【ホット・ケーキ】',
                               'to一粒雨gaatatta': 'to hitotsubu 【一粒】 ame 【雨】 ga atatta',
                               'to星ga一tsu残tteiru': 'to hoshi 【星】 ga hitotsu 【一つ】 nokotte 【残って】 iru',
                               '爪de': 'tsume 【爪】 de',
                               'wohikkaku': 'wo hikkaku',
                               '涙wo': 'namida 【涙】 wo',
                               'kobosu': 'kobosu',
                               '砕keru': 'kudakeru 【砕ける】',
                               'toコルクwo抜ku': 'to KORUKU 【コルク】 wo nuku 【抜く】',
                               'to肩wo叩ku': 'to kata 【肩】 wo tataku 【叩く】',
                               'to投geteyaru': 'to nagete 【投げて】 yaru',
                               'to断ru': 'to kotowaru 【断る】',
                               '花火ga上gatta': 'hanabi 【花火】 ga agatta 【上がった】',
                               '手wo鳴rasu': 'te 【手】 wo narasu 【鳴らす】',
                               '怒ru': 'okoru 【怒る】',
                               'ga差shita': 'ga sashita 【差した】',
                               'no海': 'no umi 【海】',
                               'no手': 'no te 【手】',
                               'woyokeru': 'wo yokeru',
                               'youna高sa': 'you na takasa 【高さ】',
                               'renai': 'renai',
                               '肥eteiru': 'koete 【肥えて】 iru',
                               '事dehanai': 'koto 【事】 de wa nai',
                               '何ka': 'nani 【何】 ka',
                               '事gaaruka': 'koto 【事】 ga aru ka',
                               '弥陀/ミダ/no': 'mida 【弥陀】 no',
                               '六字no': 'rokuji 【六字】 no',
                               '起ki上garu': 'okiagaru 【起き上がる】',
                               '肥eta': 'koeta 【肥えた】',
                               'shiyoutoiu気ga': 'shiyou to iu ki 【気】 ga',
                               '[to]起ru': '[to] okoru 【起る】',
                               '病人no熱気de': 'byounin 【病人】 no netsuke 【熱気】 de',
                               '[気ga]': '[ki 【気】 ga]',
                               'wo見ru': 'wo miru 【見る】',
                               'ni捧geru': 'ni sasageru 【捧げる】',
                               '音gashite倒reru': 'oto 【音】 ga shite taoreru 【倒れる】',
                               '[to]云u': '[to] iu 【云う】',
                               '[to]動ku': '[to] ugoku 【動く】',
                               'wotsukeru': 'wo tsukeru',
                               'wotsukerutameni言u': 'wo tsukeru tame ni iu 【言う】',
                               '付ite以来': 'tsuite 【付いて】 irai 【以来】',
                               'notsukanai内ni': 'no tsukanai uchi 【内】 ni',
                               'suruto': 'suru to',
                               'no縁/エン/': 'no en 【縁】',
                               'sonnakotohanai': 'sonna koto wa nai',
                               '頼mu': 'tanomu 【頼む】',
                               'wo入reru': 'wo ireru 【入れる】',
                               'ga出ta': 'ga deta 【出た】',
                               'no利ku': 'no kiku 【利く】',
                               'dehakkiritoshinai': 'de hakkiri to shinai',
                               'wo得ru': 'wo eru 【得る】',
                               'ni入ri易/イリヤス/i': 'ni iriyasui 【入り易い】',
                               'ni入rinikui': 'ni irinikui 【入りにくい】',
                               'mo違wanai': 'mo chigawanai 【違がわない】',
                               '張tta目': 'hatta 【張った】 me 【目】',
                               'shita態度': 'shita taido 【態度】',
                               '寒暖計': 'kaidankei 【寒暖計】',
                               'no松/マツ/': 'no matsu 【松】',
                               'no契/チギ/ri': 'no chigiri 【契り】',
                               '流reru': 'nagareru 【流れる】',
                               '引掻kku': 'hikkaku 【引っ掻く】',
                               'sonnani': 'sonna ni',
                               'no物': 'no mono 【物】',
                               '事件wo劇ni': 'jiken 【事件】 wo geki 【劇】 ni',
                               '後wo': 'ato 【後】 wo',
                               '辞書wo': 'jisho 【辞書】 wo',
                               '堪eru': 'taeru 【堪える】',
                               '我慢suru': 'gamansuru 【我慢する】',
                               '見詰meru': 'mitsumeru 【見詰める】',
                               '木': 'ki 【木】',
                               'to腰wo下rosu': 'to koshi 【腰】 wo orosu 【下ろす】',
                               '物価ga': 'bukka 【物価】 ga',
                               '上gatteiru': 'agatte 【上がって】 iru',
                               '売reru': 'ureru 【売れる】',
                               '質問suru': 'shitsumonsuru 【質問する】',
                               '湯wo': 'yu 【湯】 wo',
                               '沸kasu': 'wakasu 【沸かす】',
                               '美shii': 'utsukushii 【美しい】',
                               '君、soudarou': 'kimi 【君】, sou darou',
                               'ga如shi(如kudearu)': 'ga gotoshi 【如し】 (gotoku 【如く】 de aru)',
                               'youni': 'you ni',
                               'naranu真実': 'naranu shinjitsu 【真実】',
                               'gadekinai': 'ga dekinai',
                               '梅雨ni': 'tsuyu 【梅雨】 ni',
                               '今月ni': 'kongetsu 【今月】 ni',
                               '店wo': 'mise 【店】 wo',
                               '世帯wo': 'setai 【世帯】 wo',
                               'ga問題da': 'ga mondai 【問題】 da',
                               '祖先no霊wo': 'sosen 【祖先】 no rei 【霊】 wo',
                               '祖先wo': 'sosen 【祖先】 wo',
                               'mohaya五時wo': 'mohaya goji 【五時】 wo',
                               'ni合u': 'ni au 【合う】',
                               'woyokusuru': 'wo yoku suru',
                               '槍(太刀)wo': 'yari 【槍】 (tachi 【太刀】) wo',
                               'shigoku': 'shigoku',
                               'ni刈ru': 'ni karu 【刈る】',
                               'ni接suru': 'ni sessuru 【接する】',
                               '耳ga': 'mimi 【耳】 ga',
                               'iru': 'iru',
                               'タバコwo吹kasu': 'TABAKO 【タバコ】 wo fukasu 【吹かす】',
                               '版図, 戸籍': 'hanto 【版図】, koseki 【戸籍】',
                               '変dato': 'hen 【変】 da to',
                               '＋': '~…'}
    _max_eid: WarodaiEid
    _highlighting = ('《', '》')

    def enable_transliteration(self, mode: bool):
        self._transliterate_collocations = mode

    def set_highlighting(self, left: str, right: str):
        self._highlighting = (left, right)

    def _normalize_kana(self, string: str) -> str:
        if self._transliterate_collocations:
            return self._normalizer[_hiragana_to_latin(string)]
        return string.replace('～', '~').replace('…', '-').replace('＝', '').replace('｜', '|').replace('＋', '~…')

    def rescan(self, fname: str = "../dictionaries/source/warodai_22.03.2020.txt",
               show_progress: bool = True) -> WarodaiDictionary:
        self._entries, temp_entries = self._load_db(fname, show_progress)

        self._max_eid = max([WarodaiEid(e.eid) for e in self._entries if e.eid.startswith('009')])

        self._entries.append(WarodaiEntry(eid=self._max_eid.inc(), reading=['つく'], lexeme=['尽く'], translation={},
                                          references={'1': [
                                              WarodaiReference(meaning_number=['-1'], eid='008-04-11',
                                                               mode='письменная форма гл.', usable=True)]}))
        self._entries[self._get_entry_index_by_eid('003-19-23')].references['1'][0].eid = str(self._max_eid)
        self._entries.append(WarodaiEntry(eid=self._max_eid.inc(), reading=['ざいく'], lexeme=['細工'],
                                          translation={
                                              '1': [
                                                  f'{self._highlighting[0]}как 2-й компонент сложн. сл.{self._highlighting[1]} [мелкие] изделия, поделки']},
                                          references={}))

        self._extend_database(temp_entries)

        self._resolve_references(show_progress)

        return WarodaiDictionary(self._entries)

    def _extend_database(self, temp_entries: [WarodaiEntry], show_progress: bool = True):
        entries_to_add = []
        custom_reading = {'尼さん': ['あまさん'],
                          'うっちゃって置く': ['うっちゃっておく'],
                          'いみじき': ['いみじき'],
                          'いざという時は親知らず子知らず': ['いざというときはおやしらずこしらず'],
                          '辞世の歌': ['じせいのうた'],
                          '次席の人': ['じせきのひと'],
                          '自尊の念': ['じそんのねん'],
                          '其限': ['それきり', 'それっきり'],
                          '取りつ置きつ': ['とりつおきつ'],
                          'なおりあい': ['なおりあい'],
                          '無くする': ['なくする'],
                          '失くする': ['なくする'],
                          '亡くする': ['なくする'],
                          'ねんねえ': ['ねんねえ'],
                          '不惑の年': ['ふわくのとし'],
                          '放っておく': ['ほうっておく'],
                          '骨を折る': ['ほねをおる'],
                          '持たせてやる': ['もたせてやる'],
                          'エデンの国': ['えでんのくに']}
        for entry in temp_entries:
            likely_duplicate = [e for e in entries_to_add if sorted(e.lexeme) == sorted(entry.lexeme) and
                                sorted(e.reading) == sorted(entry.reading)]
            if likely_duplicate:
                dup_id = entries_to_add.index(likely_duplicate[0])
                for tr_number in entry.translation:
                    entries_to_add[dup_id].translation.setdefault(tr_number, []).extend(entry.translation[tr_number])
                for ref_number in entry.references:
                    entries_to_add[dup_id].references.setdefault(ref_number, []).extend(entry.references[ref_number])
            else:
                entries_to_add.append(entry)
                entries_to_add[-1].eid = self._max_eid.inc()
                if not entries_to_add[-1].reading:
                    entries_to_add[-1].reading = custom_reading[entries_to_add[-1].lexeme[0]]

        self._entries.extend(entries_to_add)

    def _resolve_references(self, show_progress: bool = True):
        ids_to_delete = []

        for i, entry in tqdm(list(enumerate(self._entries)),
                             desc="[Warodai] Validating references".ljust(34),
                             disable=not show_progress):
            for ref_id in entry.references:
                for j, ref in enumerate(entry.references[ref_id]):
                    targ_entry = self._get_entry_by_eid(ref.eid)
                    if self._entries[i].references[ref_id][j].mode:
                        self._entries[i].references[ref_id][j].mode = \
                            f'{self._highlighting[0]}{self._entries[i].references[ref_id][j].mode}{self._highlighting[1]}'
                    if targ_entry is None:
                        self._entries[i].references[ref_id][j].usable = False
                    elif ref.meaning_number == ['-1']:
                        if not list(targ_entry.translation.keys()) + list(
                                self._get_entry_by_eid(ref.eid).references.keys()):
                            self._entries[i].references[ref_id][j].usable = False
                    else:
                        for m_n in ref.meaning_number:
                            if m_n not in list(targ_entry.translation.keys()) + list(targ_entry.references.keys()):
                                self._entries[i].references[ref_id][j].meaning_number.remove(m_n)
                    if not ref.meaning_number:
                        self._entries[i].references[ref_id][j].usable = False
                    if ref.prefix:
                        if not any(ref.prefix in tr for tr in
                                   '; '.join(sum(list(targ_entry.translation.values()), [])).split('; ')):
                            self._entries[i].references[ref_id][j].usable = False

            if all(not ref.usable for ref in sum(list(self._entries[i].references.values()), [])) and \
                    not self._entries[i].translation:
                ids_to_delete.append(i)

        for i in sorted(ids_to_delete, reverse=True):
            print(f'No translations and no usable references: {self._entries[i].eid}')
            self._entries.pop(i)

    def _convert_to_entry(self, string: str) -> (List[WarodaiEntry], List[WarodaiEntry]):
        def _resolve_lexemes_and_readings(string: str) -> (List[str], List[str]):
            s_string = string.split('\n')
            rl = re.search(r'^(?P<reading>.*?)…?\s?(?:【(?P<lexeme>.*?)】)?\(', s_string[0])

            if rl is None:
                return [string], [], ''

            lexemes = []

            if rl.group('lexeme') is not None:
                lex = rl.group('lexeme').replace('…', '')
                lexeme_number_temp = [val for val in re.findall(r'([IV]*)', lex) if val]
                if lexeme_number_temp:
                    lex = re.sub(r'([IV]*)', '', lex)
                lexemes.extend([lx for lx in re.split(r'[･ ,]', lex) if lx])
            else:
                rd = rl.group('reading').replace('…', '')
                lexeme_number_temp = [val for val in re.findall(r'([IV]*)', rd) if val]
                if lexeme_number_temp:
                    rd = re.sub(r'([IV]*)', '', rd)
                lexemes.extend(rd.split(', '))

            readings = [x.strip() for x in
                        re.split(r'[,･]', re.sub(r'[a-zA-Z]*?', '',
                                                 JTran.transliterate_from_kana_to_hira(
                                                     re.sub(r'[…IV]*', '', rl.group('reading')))).replace('・',
                                                                                                          '').replace(
                            '！', ''))]

            return lexemes, readings

        def _extract_references(translations: List[str]) -> ({str: List[WarodaiReference]}, List[str]):
            res = {}

            if not any('a href' in tr for tr in translations):
                return res, translations

            translations = [t.replace('《сокр. см.》', '《сокр.》') for t in translations]

            for i, _ in enumerate(translations):
                if 'a href' in translations[i]:
                    mode = ''
                    body = ''
                    while True:
                        decomposed_tr = re.search(
                            r'(?P<left_part>〔\d+〕.*?)(?P<body>[^;]*)(?P<mode>《[^《]+?》)(?P<to_remove> <a href=\"#('
                            r'?P<eid>.+?)\">.+?(?P<lexeme_number>[iv]+)?\s?(?P<meaning_number>[0-9,\s]+)?(?:\(('
                            r'?P<prefix>〈.+[〉＿])\))?<\/a>,?)(?P<right_part>.+)?',
                            translations[i])

                        if decomposed_tr is None:
                            translations[i] = re.sub(r'; $', '', translations[i].replace(f'{body}{mode}', ''))
                            if re.search(r'^〔\d+〕$', translations[i]) is not None:
                                translations[i] = ''
                            break

                        if 'сокр.' in decomposed_tr.group('mode') or any(
                                fr in decomposed_tr.group('mode') for fr in ['от ', ' от']):
                            break

                        if not mode:
                            mode = decomposed_tr.group('mode')
                        if not body:
                            body = decomposed_tr.group('body')

                        ref = WarodaiReference(eid=decomposed_tr.group('eid'),
                                               mode=re.sub(r',?\s?см.', '', decomposed_tr.group('mode')[1:-1]),
                                               body=body.strip(),
                                               usable=True)
                        if decomposed_tr.group('meaning_number') is not None:
                            ref.meaning_number.extend(decomposed_tr.group('meaning_number').strip().split(', '))
                        else:
                            ref.meaning_number.append('-1')
                        if decomposed_tr.group('prefix') is not None:
                            ref.prefix = decomposed_tr.group('prefix')

                        res.setdefault(re.search(r'^〔(\d+)〕', decomposed_tr.group('left_part')).group(1), []).append(
                            ref)

                        translations[i] = translations[i].replace(decomposed_tr.group('to_remove'), '')

            translations = [tr.strip() for tr in translations]
            return res, [tr for tr in translations if tr]

        def _normalize_translations(translations: List[str]) -> {int: List[str]}:
            def _change_highlighting(translation: str) -> str:
                if self._highlighting != ('《', '》'):
                    return translation.replace('《', self._highlighting[0]).replace('》', self._highlighting[1])
                return translation

            if not translations:
                return {}
            res = {}
            keys = [re.search(r'〔(\d+)〕', item).group(1) for item in translations]
            values = [_change_highlighting(re.search(r'〔\d+〕(.+)', item).group(1)) for item in translations]
            for item in zip(keys, values):
                res.setdefault(item[0], []).append(item[1])
            return res

        def _extract_translations(translations: List[str]) -> List[str]:
            cleaned_translations = []
            for tr in translations:
                tr_temp = tr * 1
                tr_temp = tr_temp.replace('тж.', '@') \
                    .replace('тк.', '@') \
                    .replace('редко', '@') \
                    .replace('чаще', '@') \
                    .replace('обычно', '@') \
                    .replace('сокр.', '@') \
                    .replace(' 永;', '') \
                    .replace('井)', '') \
                    .replace('(<i>сокр. погов.</i> いざという時は親知らず子知らず)', '') \
                    .replace('(<i>тж.</i> ～になる)', '') \
                    .replace('(<i>как написание</i> 角 <i>и</i> 甪)', '') \
                    .replace('五畿内', '') \
                    .replace('(<i>о своем тк.</i> 具える)', '') \
                    .replace('(<i>о себе тк.</i> 具わる)', '') \
                    .replace('(<i>как много в Китае людей по фамилии Чжан</i> 張 <i>и Ли</i> 李)', '') \
                    .replace('(<i>слово-каламбур, основанное на том, что</i> ななつ 七つ <i>синонимично слову</i> しち 七, '
                             '<i>а</i> しち 七 <i>омонимично компоненту</i> 質 <i>в слове</i> しちや 質屋)', '') \
                    .replace('<i>ист.</i> три царства (<i>а) Япония, Китай, Индия; б) Вэй</i> 魏, <i>У</i> 呉, '
                             '<i>Шу</i> 蜀)', '') \
                    .replace('(<i>искаженное</i> 取りつ置きつ)', '') \
                    .replace('(<i>сокр.</i> ななくさのせっく【七草の節句】)', '') \
                    .replace('(<i>из-за созвучия слов</i> 氷菓子 <i>и</i> 高利貸し)', '') \
                    .replace('(<i>обычно</i> 尼さん)', '') \
                    .replace('<i>сокр.</i> китигаи 気違い', '') \
                    .replace('(36 кв. сяку 尺)', '') \
                    .replace('(貫)', '') \
                    .replace(' 仙 ', '') \
                    .replace('愛郷の念', '') \
                    .replace('的', '') \
                    .replace('百尺竿頭に一歩を進める', '') \
                    .replace('頤振り三年', '') \
                    .replace('合わせ物は離れ物', '') \
                    .replace('一物あらば一累を添う', '') \
                    .replace('鷸蚌の争いは漁夫の利', '') \
                    .replace('陰陽五行説', '') \
                    .replace('瘡気と自惚れのないものはない', '') \
                    .replace('百年河清を待つ', '') \
                    .replace('鶏口となるも牛後(牛尾)となるなかれ', '') \
                    .replace('10 сяку (勺)', '') \
                    .replace('里腹三日', '') \
                    .replace('蜀犬日に吠える', '') \
                    .replace('自縄自縛に陥る', '') \
                    .replace('短気は損気', '') \
                    .replace('酒は憂いの玉ぼうき', '') \
                    .replace('太郎と花子', '') \
                    .replace('茶腹も一時', '') \
                    .replace('渇しても盗泉の水を飲まず', '') \
                    .replace('人には能不能がある', '') \
                    .replace(' 八)', '') \
                    .replace('<i>(звук</i> ん <i>в яп. языке)</i>', '') \
                    .replace('百聞一見に如かず', '') \
                    .replace('遠水をもって近火を救うべからず', '') \
                    .replace('一桃腐りて百桃損ず', '') \
                    .replace('(<i>букв.</i> как знак 一)', '') \
                    .replace('又もとの杢阿弥だ', '') \
                    .replace('«や»', '') \
                    .replace('細工は流々仕上げを御覧じろ', '') \
                    .replace('<i>сокр.</i> 気違い', '') \
                    .replace('ミイラ取りがミイラになる', '') \
                    .replace('(<i>в виде знака</i> 井)', '') \
                    .replace('盗人に追い銭', '') \
                    .replace('九仞の功を一簣に虧く', '') \
                    .replace('焼け跡の釘拾い', '') \
                    .replace('初めの勝ちは糞勝ち', '') \
                    .replace('巧遅は拙速に如かず', '') \
                    .replace('猿の尻笑い', '') \
                    .replace('<i>@</i> 気違い', '') \
                    .replace('(<i>от</i> ごねる <i>и</i> 得)', '') \
                    .replace('(<i>от яп.</i> とても <i>и нем.</i> schön)', '') \
                    .replace('(<i>яп.</i> はっぴ <i>и англ.</i> coat)', '') \
                    .replace('(<i>яп.</i> 丸 <i>и англ.</i> copyright)', '') \
                    .replace('(<i>кит.</i> 没法子 [мэй фацзы])', '') \
                    .replace('<i>от</i> ベース <i>и англ.</i> up', '') \
                    .replace('omachidousama 【お待ち遠様】 [deshita]', '') \
                    .replace('駟も舌に及ばず <i>посл.', '') \
                    .replace('衆口は金を熔かす', '') \
                    .replace('屁をひって尻すぼめ', '') \
                    .replace('知者にも千慮の一失', '') \
                    .replace('憎まれっ子世に憚る', '') \
                    .replace('万芸に通じて一芸を成さない', '') \
                    .replace('夜目遠目傘の中', '') \
                    .replace('李下に冠を正さず, 李下の冠', '')
                tr_temp = re.sub(r'(<a href=.+?<\/a>)', '', tr_temp)
                tr_temp = re.sub(r'\(<i>@.+?\s[^～]+?\)', '', tr_temp)
                tr_temp = re.sub(r'{.+}', '', tr_temp)

                tr_temp = re.sub(r'(～)([^\s]+?\s)', r'\1 ', tr_temp + ' ')
                tr_temp = re.sub(r'(＝[^\s]+)\s', ' ', tr_temp)
                tr_temp = re.sub(r'(＋[^\s]+)\s', ' ', tr_temp)
                tr_temp = re.sub(r'…([^～]+)～', '', tr_temp)
                tr_temp = re.sub(r'｜([^～]+)～', '', tr_temp)
                tr_temp = re.sub(r'(…?～ )', '', tr_temp)

                tr_temp = re.sub(r'(?P<to_remove>\(<i>(?P<mode>[^)]+?)</i> (?P<lexeme>[^<a-zа-я0-9(]+?)\) )', '',
                                 tr_temp)
                tr_temp = re.sub(r'\(\<i\>[а-я\s]+\.?\<\/i\> [^)]+\) \(\<i\>[а-я\s]+\.?\<\/i\> [^)]+\)', '', tr_temp)

                if not tr_temp or any(_is_kanji(char) for char in tr_temp) and any(
                        _is_kanji(char) for char in re.sub(r'(【.+?】)', '', tr_temp)) or \
                        'на своем месте по алфавиту' in tr_temp:
                    continue
                if not any(_is_hira_or_kata(char) for char in tr_temp):
                    cleaned_translations.append(tr)

            not_to_confuse_expressions = [r"(\(<i>не смешивать с.+?\)\s?)",
                                          r"(<i>не смешивать .+?</i> <a href=.+?>.+?</a>)",
                                          r"(\d\) <i>не смешивать с.+)"]
            for exp in not_to_confuse_expressions:
                cleaned_translations = [re.sub(exp, '', ct) for ct in cleaned_translations]

            cleaned_translations = [
                tr.replace(': ～', ' ～').replace(': …', ' …').replace('◇', '').replace('.:', '.').replace(
                    '<a href="#004-81-10">二</a>', '二').strip() for tr in
                cleaned_translations]
            cleaned_translations = [re.sub(r'^(\d+\)):', r'\1', tr) for tr in cleaned_translations]

            cleaned_translations = [
                tr.replace(' <i>и</i>', ',').replace(' <i>а тж.</i>', '').replace('<i>и ср.</i>', '<i>ср.</i>').replace(
                    '</a> и <a', '</a>, <a').replace(' <i>(зима)</i>', '').replace(' <i>(весна)</i>', '').replace(
                    ' <i>(лето)</i>', '').replace(' <i>(осень)</i>', '') for tr
                in
                cleaned_translations if tr]
            cleaned_translations = [re.sub(r'(^\d+\.):(\s.+)', r'\1\2', ct) for ct in cleaned_translations]

            removal_expressions = [r'\((?P<mode><i>с[мр]\.[\sа-я\.]*)</i>\s(?P<to_remove><a.+?>.+?</a>,?\s?)',
                                   r'>?(?P<mode>(?:[;,] )?<i>ср\.[\sа-я\.]*)</i>\s(?P<to_remove><a.+?>.+?</a>[,;]?\s?)',
                                   r'(?P<mode>[;,] с[мр]\.[\sа-я\.]*)</i>\s(?P<to_remove><a.+?>.+?</a>[,;]?\s?)',
                                   r'>?(?P<mode>[;,] <i>с[мр]\.[\sа-я\.]*)</i>\s(?P<to_remove><a.+?>.+?</a>[,;]?\s?)',
                                   r'(?P<mode>(?:; )?<i>ант.)</i>\s(?P<to_remove><a.+?>.+?</a>)',
                                   r'(?P<mode><i>гл. обр.)</i>\s(?P<to_remove><a.+?>.+?</a>[,;]?\s?)',
                                   r'(?P<mode>)(?P<to_remove>\s\(<a href.+?>.+?<\/a>\))']
            # extraction_expressions = [ r'<i>(?P<mode>[а-я\.\s]*см\.)</i>\s<a href=\"#(?P<eid>.+)\".+?(
            # ?P<lex_num>I*)】?(?P<tr_num>[0-9]*)</a>', r'～[^\s]+\s<i>(?P<mode>[а-я\.]*см\.)</i>\s<a href=\"#(
            # ?P<eid>.+)\".+?(?P<lex_num>I*)】?(?P<tr_num>[0-9]*)</a>']
            for i, _ in enumerate(cleaned_translations):
                if 'a href' not in cleaned_translations[i]:
                    continue
                for exp in removal_expressions:
                    mode = ''
                    while True:
                        to_remove = re.search(exp, cleaned_translations[i])
                        if to_remove is not None:
                            if not mode:
                                mode = to_remove.group('mode')
                            cleaned_translations[i] = cleaned_translations[i].replace(to_remove.group('to_remove'), '')
                        else:
                            if mode:
                                cleaned_translations[i] = cleaned_translations[i].replace(f'{mode}</i> <i>и т. п.</i>',
                                                                                          '')
                                if '<i>' in mode:
                                    cleaned_translations[i] = cleaned_translations[i].replace(f'{mode}</i> ', '')
                                else:
                                    cleaned_translations[i] = cleaned_translations[i].replace(f'{mode}', '')
                                cleaned_translations[i] = cleaned_translations[i].replace(' ()', '').replace('()', '')
                            break

            cleaned_translations = [re.sub(r'^[.;]$', '', ct) for ct in cleaned_translations]
            cleaned_translations = [re.sub(r'^\d+\)\s?.$', '', ct) for ct in cleaned_translations]
            cleaned_translations = [ct.replace(' )', ')') for ct in cleaned_translations if ct]

            pointed_numbers = any(re.search(r'^\d\.', ct) is not None for ct in cleaned_translations)
            bracketed_numbers = any(re.search(r'^\d\)', ct) is not None for ct in cleaned_translations)
            b_num_expression = r'(^\d+\))(.*)'
            p_num_expression = r'(^\d+\.)(.*)'
            if bracketed_numbers:
                num_expression = r'(^\d+\))(.*)'
            elif pointed_numbers:
                num_expression = r'(^\d+\.)(.*)'

            if len(cleaned_translations) > 1 and re.search(r'^\d[.)]', cleaned_translations[0]) is None and \
                    (bracketed_numbers or pointed_numbers):
                for i, _ in enumerate(cleaned_translations[1:]):
                    num = re.search(num_expression, cleaned_translations[i + 1])
                    if num is not None:
                        cleaned_translations[i + 1] = f'{num.group(1)} {cleaned_translations[0]}{num.group(2)}'
                cleaned_translations = cleaned_translations[1:]

            if bracketed_numbers or pointed_numbers:
                cur_id = -1
                for i, _ in enumerate(cleaned_translations):
                    num = re.search(p_num_expression, cleaned_translations[i])
                    if num is None and bracketed_numbers:
                        num = re.search(b_num_expression, cleaned_translations[i])
                    if num is not None:
                        cur_id = i
                    elif cleaned_translations[i][0] in ['～', '｜', '＝', '…', '[', '＋'] or _is_kanji(
                            cleaned_translations[i][0]) \
                            or _is_hira_or_kata(cleaned_translations[i][0]) or cleaned_translations[i].startswith('<i>') \
                            or re.search(r'^[а-ж]\)\s', cleaned_translations[i]) is not None or re.search(r'^[а-я]+',
                                                                                                          cleaned_translations[
                                                                                                              i]) is not None:
                        cleaned_translations[
                            cur_id] = f'{cleaned_translations[cur_id]}{" " if cleaned_translations[cur_id][-1] in [";", ")", "."] else "; "}{cleaned_translations[i]}'
                        cleaned_translations[i] = ''
                cleaned_translations = [ct for ct in cleaned_translations if ct]

            if pointed_numbers and bracketed_numbers:
                aff_id = -1
                cur_aff = ''
                add_meaning = ''
                for i, _ in enumerate(cleaned_translations):
                    if re.search(r'^\d\.$', cleaned_translations[i]) is not None:
                        cur_aff = cleaned_translations[i]
                        aff_id = i
                        add_meaning = ''
                    elif re.search(r'^\d\..+$', cleaned_translations[i]) is not None:
                        cur_aff = re.search(p_num_expression, cleaned_translations[i]).group(1)
                        add_meaning = re.search(p_num_expression, cleaned_translations[i]).group(2)[1:]
                        aff_id = i
                    elif re.search(b_num_expression, cleaned_translations[i]) is not None:
                        if not add_meaning:
                            cleaned_translations[
                                i] = f'{cur_aff}{"" if cleaned_translations[i][0].isnumeric() else " "}{cleaned_translations[i]} '
                        else:
                            tmp1 = re.search(b_num_expression, cleaned_translations[i]).group(1)
                            tmp2 = re.search(b_num_expression, cleaned_translations[i]).group(2)
                            cleaned_translations[
                                i] = f'{cur_aff}{"" if cleaned_translations[i][0].isnumeric() else " "}{tmp1} {add_meaning}{tmp2} '
                        cleaned_translations[aff_id] = ''
                cleaned_translations = [ct for ct in cleaned_translations if ct]

            if not pointed_numbers and not bracketed_numbers and len(cleaned_translations) > 1:
                if re.search(r'^<i>[^</i>]+</i>$', cleaned_translations[0]) is not None:
                    cleaned_translations = [f'{cleaned_translations[0]} {tr}' for tr in cleaned_translations[1:]]

            cleaned_translations = [re.sub(r'(\s[а-ж\d]+\)\s)', ' ', tr) for tr in cleaned_translations if tr]

            if pointed_numbers or bracketed_numbers:
                cur_val = -1
                for i, _ in enumerate(cleaned_translations):
                    tr_idx = re.search(r'^(?P<number>\d+)[\d).]+\s(?P<value>.*)', cleaned_translations[i])
                    if tr_idx is not None:
                        cleaned_translations[i] = f'〔{tr_idx.group("number")}〕{tr_idx.group("value")}'
                        cur_val = i
                    elif i > 0:
                        if cleaned_translations[i].startswith('～'):
                            cleaned_translations[cur_val] += ' ' + cleaned_translations[i]
                            cleaned_translations[i] = ''
                        else:
                            alpha_pref = re.search(r'^(?P<a_p>[а-ж]\))(?P<rest>\s.+)', cleaned_translations[i])
                            if alpha_pref is not None:
                                cleaned_translations[cur_val] += alpha_pref.group('rest')
                                cleaned_translations[i] = ''
            else:
                cleaned_translations = [f'〔1〕{ct}' for ct in cleaned_translations]

            cleaned_translations = [re.sub(r'(?:<i>)(?P<op_pr>\(?)(?P<body>.+?)(?P<punct>[:;]?)(?P<cl_pr>\)?)(?:<\/i>)',
                                           lambda m: f'《{m.group("op_pr")}{m.group("body")}{m.group("cl_pr")}》', tr) for
                                    tr in cleaned_translations]

            if any('》' in t for t in cleaned_translations):
                quotes = [(re.search(r'《([^》]+)》$', t), cleaned_translations.index(t)) for t in cleaned_translations]
                quotes = [i for q, i in quotes if q is not None and ' ' in q.group(1)
                          and re.search(r'[a-zA-Z]+', q.group(1)) is None
                          and q.group(1).endswith('.')
                          and not any(q.group(1).endswith(end) for end
                                      in [' вр.', '-л.', 'знач.', 'гл.', ' гг.', 'т. п.', 'и др.', 'т. д.', 'накл.'])
                          and q.group(1) != 'С. У.']

                for i in quotes:
                    cleaned_translations[i] = re.sub(r'\.》$', '》', cleaned_translations[i])

            cleaned_translations = [re.sub(r'(\(《[^\(\)]+》\))', lambda m: m.group(1)[1:-1], ct) for ct in
                                    cleaned_translations]
            cleaned_translations = [re.sub(r'(《\([^\(\)]+\)》)', lambda m: f'《{m.group(1)[2:-2]}》', ct) for ct in
                                    cleaned_translations]
            cleaned_translations = [ct.replace(', ср.》', '》') for ct in cleaned_translations]
            cleaned_translations = [re.sub(r'《«([^《«»》]+)»》', r'《\1》', ct) for ct in cleaned_translations]
            cleaned_translations = [ct.replace('.》 《', '. ') for ct in cleaned_translations]
            cleaned_translations = [ct.replace('》; 《', '; ') for ct in cleaned_translations]

            for i, _ in enumerate(cleaned_translations):
                old_kp = list({x[:-1] if re.search(r'^[^(]+\)', x) else x for x in
                               re.findall(r'[^а-я>【]([…～＝｜＋][^\s<,;a-zа-я».]+)', ' ' + cleaned_translations[i])})
                for j, kana_part in enumerate(sorted([re.split(r'([…～＝｜＋])', k) for k in old_kp], reverse=True)):
                    kana_part = [part for part in kana_part[1:] if part]
                    kana_part_latin = [(self._normalize_kana(part), part) for part in kana_part]
                    for kp1, kp2 in list(zip([x for x in kana_part_latin if kana_part_latin.index(x) % 2 == 0],
                                             [x for x in kana_part_latin if kana_part_latin.index(x) % 2 == 1])):
                        cleaned_translations[i] = cleaned_translations[i].replace(f'{kp1[1]}{kp2[1]}',
                                                                                  f'〈{kp1[0]}{kp2[0]}〉')
                if old_kp:
                    cleaned_translations[i] = cleaned_translations[i].replace('〈-', '…〈-')
                    cleaned_translations[i] = re.sub(r'〈\|([^〈]+?)〉～?', r'＿≪-\1≫', cleaned_translations[i])
                    cleaned_translations[i] = cleaned_translations[i].replace('〉〈~', '〉＿〈~')
                    cleaned_translations[i] = cleaned_translations[i].replace('〉～', '〉＿')

            cleaned_translations = [re.sub(r'〈~\[([a-z\s]+)]〉', r'〈[~\1]〉', ct) for ct in cleaned_translations]

            cleaned_translations = [ct.replace('》 《', ', ') for ct in cleaned_translations]
            for i, _ in enumerate(cleaned_translations):
                if 'т. п.》; 《' not in cleaned_translations[i]:
                    cleaned_translations[i] = cleaned_translations[i].replace('》; 《', '; ')
            cleaned_translations = [re.sub(r'([.;])$', '', ct.lower()).replace('́', '').strip() for ct in
                                    cleaned_translations if ct]
            return cleaned_translations

        def _resolve_ref_reading(src_readings: List[str], new_lexemes: str):
            temp_lexeme = re.sub(r'/.+?/', '', new_lexemes).split(', ')
            for t_l in temp_lexeme:
                kana = [char for char in t_l if _is_hiragana(char)]
                for s_r in src_readings:
                    if any(char not in s_r for char in kana) and kana:
                        return []
            return src_readings

        res = []
        temp_res = []
        s_string = string.split('\n')

        eid = re.search(r'〔(.*)〕', s_string[0]).group(1)

        if eid in ['006-98-10', '005-88-27',  # формы
                   # ссылки без статей
                   '004-93-94', '003-76-16', '003-28-65', '002-08-39', '003-56-77', '002-95-49',
                   '004-46-25', '000-29-73', '002-84-88', '002-50-71', '002-78-60', '003-64-62',
                   '006-88-95', '000-08-43',
                   # статья-дубликат
                   '009-19-87']:
            return [], []

        lexemes, readings = _resolve_lexemes_and_readings(string)
        references = []

        global_quasi_reference = re.search(r'^<i>(?P<mode>[^)(]+?)<\/i> (?P<lexeme>[^<a-zA-Zа-яА-Я0-9(～]+?)$',
                                           string.split('\n')[1])
        if global_quasi_reference is not None:
            mode = ''
            if global_quasi_reference.group('mode') in ['уст.', 'неправ.', 'кн.', 'редко', 'тж.', 'чаще']:
                mode = re.sub(r'\s?тж\.\s?', '', global_quasi_reference.group('mode'))
                mode = f'《{mode}》 ' if mode else ''
            else:
                mode = 'ignore'
            translations = _extract_translations(string.split('\n')[2:])
        else:
            translations = _extract_translations(string.split('\n')[1:])

        if translations:
            references, translations = _extract_references(translations)

            if global_quasi_reference is not None and mode and mode != 'ignore':
                if translations:
                    temp_translations = [re.sub(r'〔\d+〕', '', tr) for tr in translations]
                    temp_res.append(
                        WarodaiEntry(lexeme=re.sub(r'/.+?/', '', global_quasi_reference.group('lexeme')).split(', '),
                                     eid='||',
                                     reading=_resolve_ref_reading(readings, global_quasi_reference.group('lexeme')),
                                     translation={'1': [f'{mode}{tr}' for tr in temp_translations]},
                                     references={}))
                elif references:
                    temp_res.append(
                        WarodaiEntry(lexeme=re.sub(r'/.+?/', '', global_quasi_reference.group('lexeme')).split(', '),
                                     eid='||',
                                     reading=_resolve_ref_reading(readings, global_quasi_reference.group('lexeme')),
                                     translation={},
                                     references={'1': [WarodaiReference(eid=eid, usable=True,
                                                                        mode=mode[1:-2] if mode else '',
                                                                        meaning_number=list(references.keys()))]}))

            double_quasi_references = [re.search(r'\((《[а-я\s\.]+?》 [^)a-zа-я]+\) \(《[а-я\s\.]+》 [^)a-zа-я]+)\)', tr)
                                       for tr in translations]
            referenced_translations = list(zip(double_quasi_references, translations))
            referenced_translations = [(r, t) for r, t in referenced_translations if r is not None]
            if referenced_translations:
                for r, t in referenced_translations:
                    parts = [re.search(r'(?P<to_remove>《(?P<mode>[^》]+)》 (?P<lexeme>.+))', p) for p in
                             r.group(1).split(') (')]
                    parts = [p for p in parts if p.group('mode') != 'о себе тк.']
                    temp_tr = re.sub(r'〔\d+〕', '', t.replace(f'({r.group(1)}) ', ""))
                    for part in parts:
                        if part.group('mode') in ['тк.', 'тж.']:
                            temp_res.append(WarodaiEntry(lexeme=[part.group('lexeme')],
                                                         eid='|||',
                                                         reading=readings,
                                                         translation={'1': [temp_tr]},
                                                         references={}))
                        else:
                            temp_res.append(WarodaiEntry(lexeme=[part.group('lexeme')],
                                                         eid='|||',
                                                         reading=readings,
                                                         translation={'1': [
                                                             f'《{part.group("mode").replace("тк.", "").strip()}》 ' + temp_tr]},
                                                         references={}))
                    if 'тк.' in [p.group('mode') for p in parts]:
                        translations.remove(t)
                    else:
                        t_id = translations.index(t)
                        for part in parts:
                            translations[t_id] = translations[t_id].replace(f"({part.group('to_remove')}) ", '')

            quasi_references = [re.search(r'(?P<to_remove>\(《(?P<mode>[^)》]+?)》 (?P<lexeme>[^<a-zа-я0-9(]+?)\) )', tr)
                                for tr in translations]
            referenced_translations = list(zip(quasi_references, translations))
            referenced_translations = [(r, t) for r, t in referenced_translations if r is not None]
            if referenced_translations:
                for r, t in referenced_translations:
                    if all(new_lex in lexemes for new_lex in r.group('lexeme').split(', ')) and 'тк.' not in r.group(
                            'mode') or r.group('mode') in ['о своем тк.', 'о себе тк.', 'сокр. от', 'от сокр.',
                                                           'сокр. яп.', 'производное от']:
                        if r.group('mode') in ['сокр. от', 'от сокр.', 'сокр. яп.', 'производное от']:
                            translations[translations.index(t)] = t.replace(r.group('lexeme'),
                                                                            self._normalize_kana(r.group('lexeme')))
                        continue
                    if r.group('mode') not in ['вм.', 'сокр.', 'от', 'от первой буквы слова'] and 'форм' not in r.group(
                            'mode'):
                        mode = re.sub(r"\s?т[жк]\.\s?", "", r.group("mode"))
                        if mode == 'правильнее':
                            replace_with = 'неправ.'
                            mode = ''
                        elif mode == 'обычно':
                            replace_with = 'реже'
                            mode = ''
                        elif mode in ['часто', 'чаще']:
                            replace_with = ''
                            mode = ''
                        elif mode == 'сокр. погов.':
                            replace_with = r.group('to_remove')
                            mode = ''
                        elif mode == 'неправ. вм.':
                            replace_with = 'неправ.'
                            mode = ''
                        elif mode == 'искаж.':
                            replace_with = 'искаж.'
                            mode = ''
                        else:
                            replace_with = ''
                        mode = f'《{mode}》 ' if mode else ''
                        replace_with = f'《{replace_with}》 ' if replace_with else ''
                        special_reading = re.search(r'/([^/]+?)/', r.group('lexeme'))
                        if special_reading is not None:
                            ref_reading = [JTran.transliterate_from_kana_to_hira(special_reading.group(1))]
                        else:
                            ref_reading = _resolve_ref_reading(readings, r.group('lexeme'))
                        temp_res.append(WarodaiEntry(lexeme=re.sub(r'/.+?/', '', r.group('lexeme')).split(', '),
                                                     eid='|', reading=ref_reading,
                                                     translation={'1': [
                                                         re.sub(r'〔\d+〕', '', t.replace(r.group('to_remove'), mode))]},
                                                     references={}))
                        if 'тк.' in r.group('mode'):
                            translations.remove(t)
                        else:
                            translations[translations.index(t)] = t.replace(r.group('to_remove'), replace_with)
                    else:
                        t_id = translations.index(t)
                        translations[t_id] = re.sub(r'(/[^\s]+?/)', '', translations[t_id])
                        translations[t_id] = t.replace(r.group('lexeme'),
                                                       ' + '.join([self._normalize_kana(lx) for lx in
                                                                   r.group('lexeme').split(', ')]))

        if references or translations:
            res.append(WarodaiEntry(eid=eid, reading=readings, lexeme=lexemes,
                                    translation=_normalize_translations(translations), references=references))
        elif not temp_res:
            print(f'Failed to parse entry: {eid}')
        return res, temp_res

    def _load_db(self, fname: str, show_progress: bool = True) -> (List[WarodaiEntry], List[WarodaiEntry]):
        with open(fname, encoding='utf-16le') as f:
            contents = f.read()

            to_replace = [('<i>искажённое</i>', '<i>искаж.</i>'),
                          ('[…', '…['),
                          ('…[を, …に]', '…[を(に)]'),
                          ('(<i>сопровождается в конце предложения частицами</i> [に於て]をや)', ''),
                          ('(<i>вежл.</i> お早うございます)', ''),
                          ('(<i>т. н.</i> じょうげん【上元】, ちゅうげん【中元】<i>и</i> かげん【下元】)', ''),
                          ('(<i>сокр.</i> じばん【地盤】, かんばん【看板】, かばん【鞄】)',
                           '(<i>сокр.</i> 地盤, 看板, 鞄)'),
                          ('(<i>ант.</i> てがるい【手軽い】)', ''),
                          (
                              '<i>ист.</i> три царства (<i>а) Япония, Китай, Индия; б) Вэй</i> 魏, <i>У</i> 呉, '
                              '<i>Шу</i> 蜀)',
                              '<i>ист.</i> три царства (Япония, Китай, Индия; Вэй, У, Шу)'),
                          ('продавец насекомых (<i>напр.</i> <a href="#000-22-17">まつむし【松虫】</a>, '
                           '<a href="#001-53-38">すずむし【鈴虫】</a>, <i>которые ценятся за издаваемые ими '
                           'звуки</i>).',
                           'продавец насекомых (<i>которые ценятся за издаваемые ими звуки</i>).'),
                          ('(<i>напр.</i> 5 — <i>ср.</i> <a href="#008-31-45">ごあく【五悪】</a>, '
                           '<a href="#006-52-79">ごかい【五戒】</a>, <a href="#009-59-31">ごきょう【五経】</a> <i>и т. '
                           'п.</i>)', ''),
                          (' (<i>напр.</i> 瓜 <i>и</i> 爪)', ''),
                          (' (<i>напр.</i> ぴかぴか)', ''),
                          (' (<i>напр.</i> 上, 三, 凸)', ''),
                          ('(<i>напр.</i> 梅の花)', ''),
                          ('(<i>при отвлечённом счёте, ср.</i> ひ【一】, ふ【二】, み【三】)',
                           '(<i>при отвлечённом счёте</i>'),
                          ('(<i>из-за созвучия слов</i> <a href="#006-38-91">こおりがし【氷菓子】</a> <i>и</i> <a '
                           'href="#007-72-05">こうりがし【高利貸し】</a>)',
                           '(<i>из-за созвучия слов</i> 氷菓子 <i>и</i> 高利貸し)'),
                          ('(<i>обычно</i> あまさん【尼さん】)', '(<i>обычно</i> 尼さん)'),
                          (
                              '(<i>ср.</i> <a href="#006-83-43">にのぜん</a>, <a href="#000-74-07">にのいと</a> <i>и др. слова, '
                              'начинающиеся с</i> にの…【二の…】)', ''),
                          (' (<i>см.</i> しょうぎょうほうそう【商業放送】)', ''),
                          ('(<i>англ.</i> baby <i>и яп.</i> たんす【簞笥】)',
                           '(<i>англ.</i> baby <i>и яп.</i> 簞笥)'),
                          ('好い気味だ', '～だ'),
                          ('…[の]模様だ', '…[の]～だ'),
                          ('宜なるかな', '～なるかな'),
                          ('胸糞が悪い', '～が悪い'),
                          ('糞味噌にけなす', '～にけなす'),
                          ('</i> いわざる, きかざる, みざる <i>—', ''),
                          ('ぽきんと折れる', '～折れる'),
                          ('ぺちゃくちゃしゃべる', '～しゃべる'),
                          ('べちゃべちゃしゃべる', '～しゃべる'),
                          ('へとへとに疲れる', '～に疲れる'),
                          ('ぶらぶら下がる', '～下がる'),
                          ('[足の]踏み所もない', '…[足の]～もない'),
                          ('…の顰に傚う', '…の～に傚う'),
                          (
                              '(<i>8 февраля, в Киото 8 декабря, день подношения храму Авасима сломанных иголок, '
                              'накопившихся за год; в этот день отдыхают от шитья; Храм Авасима был избран потому, '
                              'что иначе назывался</i> Хариса́йнё 婆利才女)', ''),
                          ('腹鼓を打つ', '～を打つ'),
                          ('腹ごなしに散歩する', '～に散歩する'),
                          (' 明州', ''),
                          ('にゃおと鳴く', '～と鳴く'),
                          ('にたりと笑う', '～と笑う'),
                          ('どたりと落ちる', '～と落ちる'),
                          ('頬杖を突く', '～を突く'),
                          ('付きが良い', '～が良い'),
                          ('手枕をして寝る', '～をして寝る'),
                          ('外輪(鰐)の足', '～の足'),
                          ('金離れの悪い', '～の悪い'),
                          ('(<i>от кит.</i> сянь 仙)', '(<i>от кит.</i> 仙 [сянь])'),
                          ('…は世間知らずだ', '…は～だ'),
                          ('ずでんどうと倒れる', '～と倒れる'),
                          ('じろじろ見る', '～見る'),
                          ('しんにゅうをかけて言う', '～をかけて言う'),
                          ('さめざめと泣く', '～と泣く'),
                          ('ごろりと寝転ぶ', '～と寝転ぶ'),
                          ('(<i>санскр.</i> Pañca Skandhāh: 色 <i>плоть</i>, 受 <i>ощущения и чувства</i>, '
                           '想 <i>воображение</i>, 行 <i>духовная деятельность</i>, 識 <i>познание</i>)', ''),
                          ('こっくりこっくりやる(居眠りをする)', '～やる, ～居眠りをする'),
                          ('ぐりはまに行く', '～に行く'),
                          ('(<i>сокр.</i> 多伽羅, <i>санскр.</i> tagara)',
                           '(<i>сокр.</i> 多伽羅) (<i>санскр.</i> tagara)'),
                          ('きゃっと叫ぶ', '～叫ぶ'),
                          ('お聞きに入れる', '～に入れる'),
                          ('陰(蔭)口をいう(利く)', '～をいう, ～を利く'),
                          ('追い立てを食う', '～を食う'),
                          ('遺腹の子', '～の子'),
                          ('居心地のよい', '～のよい'),
                          ('主顔に振る舞う', '～に振る舞う'),
                          ('からからと笑う', '～と笑う'),
                          ('相々傘で行く', '～で行く'),
                          ('愛想尽かしを言う', '～を言う'),
                          ('相槌を打つ', '～を打つ'),
                          ('飽くことを知らぬ', '～ことを知らぬ'),
                          ('あけっぴろげの笑いを浮べる', '～の笑いを浮べる'),
                          ('明け残る空', '～空'),
                          ('足拵えを厳重にする', '～を厳重にする'),
                          ('足触りがよい', '～がよい'),
                          ('足なれた人', '～人'),
                          ('足湯を使う', '～を使う'),
                          (
                              'атэдзи, искусственно (неправильно) подобранные иероглифы <i>(1) иероглифы, которыми '
                              'пишутся некоторые слова японского и реже китайского происхождения по смыслу слова в '
                              'целом, а не по обычному чтению каждого иероглифа в отдельности:</i> 平時 <i>для '
                              'слова</i> ицумо <i>«всегда, обычно»;</i> 真実 <i>для слова</i> хонто: <i>«правда, '
                              'истина»; 2) иероглифы, употреблённые чисто фонетически:</i> 目出度く мэдэтаку '
                              '<i>«успешно»)</i>',
                              'атэдзи\n1) <i>иероглифы, которыми '
                              'пишутся некоторые слова японского и реже китайского происхождения по смыслу слова в целом, '
                              'а не по обычному чтению каждого иероглифа в отдельности</i>;\n2) <i>иероглифы, употреблённые чисто '
                              'фонетически</i>'),
                          ('跡白浪と逃げる(消えてしまう)', '～と逃げる, ～消えてしまう'),
                          ('あぶあぶいう, あぶあぶやる', '～いう, ～やる'),
                          ('雨まじりの雪', '～の雪'),
                          ('荒胆をひしぐ(抜く)', '～をひしぐ, ～を抜く)'),
                          ('<i>см.</i> <a href="#003-07-11">ぎしん【疑心】</a>.',
                           '<i>посл.</i> у страха глаза велики (<i>букв.</i> страх порождает чёрных чертей).'),
                          ('…は偉とするに足る', '…は～とするに足る'),
                          ('言い出しっ屁だから君からやりたまえ', '～だから君からやりたまえ'),
                          ('生き恥をさらす', '～をさらす'),
                          ('一議に及ばず', '～に及ばず'),
                          ('一物あらば一累/ルイ/を添う', '一物あらば一累を添う'),
                          ('一眸の中/ウチ/に収める', '～の中/ウチ/に収める'),
                          ('一騎当千のつわもの', '～のつわもの'),
                          ('一炊の夢/ユメ/', '～の夢/ユメ/'),
                          ('一寸逃れを云う', '～を云う'),
                          ('いっちょういきましょう', '～いきましょう'),
                          ('一敗地に塗れる', '～地に塗れる'),
                          ('一風変わった男', '～変わった男'),
                          ('逸を以て労を待つ', '～を以て労を待つ'),
                          ('田舎っぽく見える', '～見える'),
                          ('居抜きのままで買う', '～のままで買う'),
                          ('命からがら逃げる', '～逃げる'),
                          ('命勝負と申す', '～と申す'),
                          ('意表に出る', '～に出る'),
                          ('居待の月', '～の月'),
                          ('今迄の所', '～の所'),
                          ('今までずっと', '～ずっと'),
                          ('色気抜きで飲む', '～で飲む'),
                          ('色よい返事', '～返事'),
                          ('曰く付きの女', '～の女'),
                          ('…と言わず', '…と～'),
                          ('と言わぬばかりに', '…と～ばかりに'),
                          ('浮かぬ顔をする', '～顔をする'),
                          ('浮河竹に身を沈める', '～に身を沈める'),
                          ('烏合の衆', '～の衆'),
                          ('…に後手に縛る', '…に～に縛る'),
                          ('後指を差す', '～を差す'),
                          ('嘘八百を並べ立てる', '～を並べ立てる'),
                          ('梲が上がらない', '～が上がらない'),
                          ('内八文字を切る', '～を切る'),
                          ('腕っぷしの強い男', '～の強い男'),
                          ('兔の毛でついたほどのすきもない', '～でついたほどのすきもない'),
                          ('有髪の尼', '～の尼'),
                          ('怨みっこのないように, 怨みっこなしに', '～のないように, ～なしに'),
                          ('рыбы а́ю (鮎)', 'рыбы а́ю'),
                          ('嬉し泣きに泣く', '～に泣く'),
                          ('上の空', '～の空'),
                          ('得も言われぬ', '～も言われぬ'),
                          ('依估の沙汰/サタ/', '～の沙汰/サタ/'),
                          ('…は嘔吐く', '…は～'),
                          ('得体の知れない', '～の知れない'),
                          ('笑壷に入/イ/る', '～に入/イ/る'),
                          ('燕趙悲歌の士', '～の士'),
                          ('猿臂を伸ばす', '～を伸ばす'),
                          ('おいおい泣く', '～泣く'),
                          ('老恥をかく', '～をかく'),
                          ('大胡坐をかく', '～をかく'),
                          ('大腐りに腐って', '～に腐って'),
                          ('大雀の鉄砲', '～の鉄砲'),
                          ('大味噌を付ける', '～を付ける'),
                          ('…を大目に見る', '…を～に見る'),
                          ('大揉めに揉める', '～に揉める'),
                          ('お蚕ぐるみでいる', '～でいる'),
                          ('<i>то же, что</i> おかえりなさい, <i>см.</i> <a href="#003-89-20">かえる【帰る】</a>.',
                           '～なさい добро пожаловать! <i>(приветствие члену семьи, вернувшемуся домой со службы, из школы '
                           'и т. п.)</i>;'),
                          ('(= おかえりなさい) <i>см.</i> <a href="#003-89-20">かえる【帰る】</a>',
                           '<i>см.</i> <a href="000-10-29">おかえり【お帰り】 2 (～なさい)</a>'),
                          ('奥さん孝行の夫', '～の夫'),
                          ('恐れ気もなく', '～もなく'),
                          ('お手の筋だ', '～だ'),
                          ('男泣きに泣く', '～に泣く'),
                          ('お見外れ申しました', '～申しました'),
                          ('親分風を吹かす', '～を吹かす'),
                          ('…は女癖が悪い', '…は～が悪い'),
                          ('お乳母日傘で育てる', '～で育てる'),
                          (', <i>ср.</i> <a href="#005-96-67">ごぎょう【五行】</a>', ''),
                          ('華を去り実/ジツ/に就く', '～を去り実/ジツ/に就く'),
                          (
                              '<i>(напр. образование из</i> 山, 上, 下 <i>иероглифа</i> 峠<i>)</i>; <i>ср.</i> <a '
                              'href="#008-24-88">りくしょ【六書】</a>', ''),
                          ('会員外の者', '～の者'),
                          ('快哉を叫ぶ', '～を叫ぶ'),
                          (', напр.</i> 河, 峰<i>', ''),
                          ('掻い撫での作者', '～の作者'),
                          ('顔向けが出来ない, 顔向けならぬ', '～が出来ない, ～ならぬ'),
                          ('掛かりつけの医者', '～の医者'),
                          ('加冠の式', '～の式'),
                          ('書き具合がいい', '～がいい'),
                          ('掛心地のよい', '～のよい'),
                          ('瘡気と自惚/ウヌボ/れのないものはない', '瘡気と自惚れのないものはない'),
                          ('華胥の国に遊ぶ', '～の国に遊ぶ'),
                          ('華燭の典 ', '～の典 '),
                          ('拍手を打つ', '～を打つ'),
                          ('加除式のノート', '～のノート'),
                          ('(黄河, <i>букв.</i> жёлтой реки) станут голубыми (清)',
                           '(<i>букв.</i> жёлтой реки) станут голубыми'),
                          ('片息をつく', '～をつく'),
                          ('肩車に乗せる', '～に乗せる'),
                          ('かたりと音を立てて落ちる', '～と音を立てて落ちる'),
                          ('片ガラスの時計', '～の時計'),
                          ('隔靴掻痒の感がある', '～の感がある'),
                          ('齧り付き主義を取る', '～を取る'),
                          ('竈持ちのよい女', '～のよい女'),
                          ('からころ音を立てる', '～音を立てる'),
                          ('かるが故に', '～が故に'),
                          ('驩を交じえる', '～を交じえる'),
                          ('侃々諤々の論', '～の論'),
                          ('汗顔の至りである', '～の至りである'),
                          ('神ながらの道', '～の道'),
                          ('神嘗の祭/マツリ/', '～の祭/マツリ/'),
                          ('幹部級の人', '～の人'),
                          ('完膚なきまで', '～なきまで'),
                          ('管鮑の交わり', '～の交わり'),
                          ('官僚畑の政治家', '～の政治家'),
                          ('画餅に帰する', '～に帰する'),
                          ('がみがみ言う', '～言う'),
                          ('(<i>сокр.</i> 僧伽藍摩, <i>санскр.</i> Samgharama)',
                           '(<i>сокр.</i> 僧伽藍摩) (<i>санскр.</i> Samgharama)'),
                          ('側[の者]', '～[の者]'),
                          ('雁字搦み(め)に縛る', '～に縛る'),
                          ('頑丈作りの男', '～の男'),
                          ('旗印の下に', '～の下に'),
                          ('古事記 ', ''),
                          (' 日本書紀', ''),
                          ('きしきし鳴る', '～鳴る'),
                          ('帰心矢のごとし', '～矢のごとし'),
                          ('喜字の齢/ヨワイ/', '～の齢/ヨワイ/'),
                          ('驥足を伸ばす', '～を伸ばす'),
                          ('後朝の別れ', '～の別れ'),
                          ('九牛の一毛/イチモウ/', '～の一毛/イチモウ/'),
                          ('九五の位/クライ/', '～の位/クライ/'),
                          ('九仞の功/コウ/を一簣/イッキ/に虧/カ/く', '九仞の功を一簣に虧く'),
                          ('虚々実々の戦術', '～の戦術'),
                          ('曲学阿世の徒', '～の徒'),
                          ('遽然として退場する', '～として退場する'),
                          ('居然として動かない', '～として動かない'),
                          ('巨歩を進める', '～を進める'),
                          ('きょろきょろ眺める, 目をきょろきょろさせる', '～眺める, ＝目を～させる'),
                          ('きりょう自慢の娘', '～の娘'),
                          ('槿花一日の栄', '～の栄'),
                          ('槿花一朝の夢', '～の夢'),
                          ('金枝玉葉[の身]', '～[の身]'),
                          (' (琴)', ''),
                          ('琴瑟相和/アイワ/す[る]', '～相和/アイワ/す[る]'),
                          ('ぎすばった顔付', '～顔付'),
                          ('ぎちぎち言う音', '～言う音'),
                          ('牛歩で進む', '～で進む'),
                          ('魚腹に葬られる, 魚腹を肥/コ/やす', '～に葬られる, ～を肥/コ/やす'),
                          ('ぎらりと光る', '～と光る'),
                          ('ぎりぎり歯を鳴らす', '～歯を鳴らす'),
                          ('ぎーぎーいう, ぎーぎー音がする', '～いう, ～音がする'),
                          ('苦学力行の士', '～の士'),
                          ('口利き役を買って出る', '～を買って出る'),
                          ('口ぎれいな事を言う', '～な事を言う'),
                          ('口拍子を取る', '～を取る'),
                          ('くっくと笑う', '～と笑う'),
                          ('苦肉の計(策)', '～の計(策)'),
                          ('首斬り反対を叫ぶ', '～を叫ぶ'),
                          ('隈ない月光', '～月光'),
                          ('倉作りの家', '～の家'),
                          ('呉れ得の貰い損', '～の貰い損'),
                          ('黒山の様な人だかり', '～の様な人だかり'),
                          ('(знаки, <i>напр.</i> 一, 二, 三, 上, 中, 下, レ, <i>указывающие',
                           '(<i>знаки, указывающие'),
                          ('具眼の士', '～の士'),
                          ('ぐしゃぐしゃにこわれる', '～にこわれる'),
                          ('ぐっすり眠る', '～眠る'),
                          ('ぐつぐつ煮る', '～煮る'),
                          ('ぐてんぐてんに酔っぱらう', '～に酔っぱらう'),
                          ('ぐびりぐびり飲む', '～飲む'),
                          ('傾国の美人', '～の美人'),
                          ('決河の勢で', '～の勢で'),
                          ('闕下に伏奏す', '～に伏奏す'),
                          ('逆鱗に触れる', '～に触れる'),
                          ('下司口を利く', '～を利く'),
                          ('げっそり痩せる', '～痩せる'),
                          ('原級留置の生徒', '～の生徒'),
                          ('…目に遭わせる', '…目に～'),
                          ('げーげーする(言う)', '～する, ～言う'),
                          ('小当りに当って見る', '～に当って見る'),
                          ('好奇の念', '～の念'),
                          ('公聞に達する', '～に達する'),
                          ('ここと鳴く', '～と鳴く'),
                          ('心行くばかり', '～ばかり'),
                          ('小腰を屈める', '～を屈める'),
                          ('こつんと頭を打ちつける', '～と頭を打ちつける'),
                          ('ことりと音がした', '～と音がした'),
                          ('この段御通知申し上げます', '～御通知申し上げます'),
                          ('鼓腹撃壌の民/タミ/', '～の民/タミ/'),
                          ('小股に歩く', '～に歩く'),
                          ('◇小股をすくう', '～をすくう'),
                          ('小耳に挟む', '～に挟む'),
                          ('こりこり音がする', '～音がする'),
                          ('こんがりと焼けている', '～と焼けている'),
                          ('豪の者', '～の者'),
                          ('傲慢不遜な', '～な'),
                          ('ごくりと飲む', '～と飲む'),
                          ('五節の舞/マイ/', '～の舞/マイ/'),
                          ('ごつんと打つ', '～と打つ'),
                          ('…の伍伴に入/ハイ/る', '…の～に入/ハイ/る'),
                          ('五倫[の道/ドウ/]', '～[の道/ドウ/]'),
                          ('ごーんと鳴る', '～と鳴る'),
                          ('逆帆を食う', '～を食う'),
                          ('酒虫が起る', '～が起る'),
                          ('避くべき事', '～事'),
                          ('探り足で行く', '～で行く'),
                          ('差しぐむ涙', '～涙'),
                          ('砂上の楼閣/ロウカク/', '～の楼閣/ロウカク/'),
                          ('札びらを切る', '～を切る'),
                          ('薩摩守を決めこむ', '～を決めこむ'),
                          ('里腹三日/サンニチ/', '里腹三日'),
                          ('Ма-Хан</i> 馬韓, <i>Пйон-Хан</i> 弁韓 <i>и Син-Хан</i> 辰韓; <i>',
                           'Ма-Хан, Пйон-Хан и Син-Хан; '),
                          ('三顧の礼をとる', '～の礼をとる'),
                          ('三舟の才', '～の才'),
                          ('三寸の舌/シタ/を振う', '～の舌/シタ/を振う'),
                          ('三船の才', '～の才'),
                          ('【三舟】(の才)', '【三舟】(の才)'),
                          ('酸鼻を極めた', '～を極めた'),
                          ('ざくりと包丁を入れる', '～と包丁を入れる'),
                          ('ざぶりと湯に入/ハイ/る', '～[と]湯に入/ハイ/る'),
                          ('ざぶんと飛び込む', '～[と]飛び込む'),
                          ('匹股を踏む', '～を踏む'),
                          ('獅子奮迅の勢いで', '～の勢で'),
                          ('七分の二', '～の二'),
                          ('七歩の才/サイ/', '～の才'),
                          ('しめこの兔/ウサギ/', '～の兔'),
                          ('(歩) ', ''),
                          ('社稷の臣', '～の臣'),
                          ('しゃっちょこ立ちしても及ばない', '～しても及ばない'),
                          ('…は宗旨違いだ', '…は～だ'),
                          ('首鼠両端を持/ジ/す', '～両端を持/ジ/す'),
                          ('出処進退を誤らず', '～を誤らず'),
                          ('出藍の誉/ホマレ/がある', '～の誉/ホマレ/がある'),
                          ('しらを切る', '～を切る'),
                          ('尻居に倒れる', '～に倒れる'),
                          ('尻馬に乗る', '～に乗る'),
                          ('尻餅をつく', '～をつく'),
                          ('心身にこたえる', '～にこたえる'),
                          ('自家薬籠中の物とする', '～の物とする'),
                          ('自成の人', '～の人'),
                          ('地団駄[を]踏む', '～[を]踏む'),
                          ('自知の明/メイ/がある', '～の明/メイ/がある'),
                          ('事務系統の職員(従業員)', '～の職員, ～の従業員'),
                          ('十中の八九まで', '～の八九まで'),
                          ('…の術中に陥る', '…の～に陥る'),
                          ('…のため寿盃を挙げる', '…のため～を挙げる'),
                          ('順風満帆のうち', '～のうち'),
                          ('上巳の節句/セック/', '～の節句/セック/'),
                          ('冗談事じゃない', '～じゃない'),
                          ('人後に落ちない', '～に落ちない'),
                          ('水魚の交わり', '～の交わり'),
                          ('すいすい[と]飛ぶ', '～[と]飛ぶ'),
                          ('陬遠の地', '～の地'),
                          ('趨否を決する', '～を決する'),
                          ('好き合って夫婦になる', '～夫婦になる'),
                          ('すってんころりんと転ぶ', '～転ぶ'),
                          ('すとんところぶ', '～ころぶ'),
                          ('住心地のよい', '～のよい'),
                          ('擦れ違いに行き過ぎる', '～に行き過ぎる'),
                          ('座り心地がいい', '～がいい'),
                          ('寸々に切る', '～に切る'),
                          ('ずぶりと刺す', '～[と]刺す'),
                          ('ずぶ六に酔う(なる)', '～に酔う, ～になる'),
                          ('ずるりと落ちる', '～と落ちる'),
                          ('臍下丹田に力を入れる', '～に力を入れる'),
                          ('生殺与奪の権', '～の権'),
                          ('生死不明の人', '～の人'),
                          ('政治家肌の人', '～の人'),
                          ('聖聞に達する', '～に達する'),
                          ('世外に起然とする', '～に起然とする'),
                          ('尺寸の地 ', '～の地 '),
                          ('潜航艇式の遣り方', '～の遣り方'),
                          ('千仞の谷', '～の谷'),
                          ('先鞭を着ける', '～を着ける'),
                          ('; <i>напр.</i> 眼科専門医 офтальмолог', ''),
                          ('贅沢三昧に暮す', '～に暮す'),
                          ('漸を追(逐)って', '～追(逐)って'),
                          ('…に全人格を傾倒する', '…に～を傾倒する'),
                          ('禅門に入/ハイ/る', '～に入/ハイ/る'),
                          ('前略[御免下さい]', '～[御免下さい]'),
                          ('箏の琴/コト/', '～の琴/コト/'),
                          ('…が捜査線上に現われる(浮ぶ)', '…が～に現われる, …が～に浮ぶ'),
                          ('壮士風の男', '～の男'),
                          ('宋襄の仁/ジン/', '～の仁/ジン/'),
                          ('鏘然として鳴る', '～として鳴る'),
                          ('総ルビの本', '～の本'),
                          ('足労をかける', '～をかける'),
                          ('外八文字を切る', '～を切る'),
                          ('俗耳に入/イ/る', '～に入/イ/る'),
                          ('高あぐらをかく', '～をかく'),
                          ('…は高手小手に縛られた', '…は～に縛られた'),
                          ('高見の見物をする', '～の見物をする'),
                          ('田毎の月/ツキ/', '～の月'),
                          ('多士済々/セイセイ/', '～済々/セイセイ/'),
                          ('立ち竦みの状態である', '～の状態である'),
                          ('…の中/ウチ/に立ち交る', '…の中/ウチ/に～'),
                          ('酒は憂/ウレ/いの玉ぼうき', '酒は憂いの玉ぼうき'),
                          ('太郎と花子/ハナコ/', '太郎と花子'),
                          ('大尽風を吹かす', '～を吹かす'),
                          (' 大阪毎日[新聞]', ''),
                          ('…に卵を抱かす', '…に卵を～'),
                          ('出しっ放しにして置く', '～にして置く'),
                          ('脱兔の如く', '～の如く'),
                          ('だらしがない', '～がない'),
                          (' (<i>не смешивать с формой связки</i> だ)', ''),
                          ('弾丸黒子の地', '～の地'),
                          ('断腸の思いがする', '～の思いがする'),
                          ('千鳥形に進む', '～に進む'),
                          ('ちゃきちゃきはさむ', '～はさむ'),
                          ('ちゃっかりしている', '～している'),
                          ('茶屋酒の味を覚える', '～の味を覚える'),
                          ('超然主義を取る', '～を取る'),
                          ('頂門の一針', '～の一針'),
                          ('直情径行型の人', '～の人'),
                          ('ちょこんと木に止まる', '～木に止まる'),
                          ('ちょっかいを出す', '～を出す'),
                          ('…の枕席に侍/ジ/す', '…の～に侍/ジ/す'),
                          (
                              '<i>(слово-каламбур, основанное на том, что иероглиф</i> 百 <i>без верхней черты, т. е. 100 '
                              'минус 1, имеет вид</i> 白 <i>«белый»)</i>', ''),
                          ('(妻 <i>здесь заменяет</i> 爪)', ''),
                          ('強腰に出る', '～に出る'),
                          ('手置きが悪い', '～が悪い'),
                          ('敵影を認めず, 敵影無し', '～を認めず, ～無し'),
                          ('てくで歩く', '～で歩く'),
                          ('てくてく歩く', '～歩く'),
                          ('手ぐすねを引く, 手ぐすねを引いて待っている', '～を引く, ～を引いて待っている'),
                          ('手玉に取る', '～に取る'),
                          ('鉄火肌の女', '～の女'),
                          ('轍鮒の急', '～の急'),
                          ('手鼻をかむ', '～をかむ'),
                          ('天金の本', '～の本'),
                          ('椽大の筆を揮う', '～の筆を揮う'),
                          (
                              '<i>(напр.</i> お <i>иероглифа</i> 悪 <i>в знач. «ненависть» при более частом</i> あく <i>в '
                              'знач. «зло»)</i>', ''),
                          ('天聴に達する', '～に達する'),
                          ('陶然と酔う', '～と酔う'),
                          ('刀筆の吏', '～の吏'),
                          ('遠耳が利く', '～が利く'),
                          ('年甲斐も無い', '～も無い'),
                          ('とっぽいやつ', '～やつ'),
                          ('飛び切りをやる', '～をやる'),
                          ('同学の友', '～の友'),
                          ('同姓同名の人', '～の人'),
                          ('同体に落ちる', '～に落ちる'),
                          ('どうと倒れる', '～倒れる'),
                          ('道徳堅固の人', '～の人'),
                          ('どかんと落ちる', '～落ちる'),
                          ('度外して置く', '～して置く'),
                          ('独酌で飲む', '～で飲む'),
                          ('…の毒手にかかる', '…の～にかかる'),
                          ('…の毒刃に倒れる', '…の～に倒れる'),
                          ('どじを踏む', '～を踏む'),
                          ('どでん買越/カイコ/す(うりこす)', '～買越/カイコ/す, ～うりこす'),
                          ('怒髪天を突く', '～天を突く'),
                          ('どぶんと落ちる', '～落ちる'),
                          ('どろんを決める', '～を決める'),
                          ('内顧の憂い(煩い)', '～の憂い, ～の煩い'),
                          ('長葉の石持草/イシモチソー/', '～の石持草/イシモチソー/'),
                          ('亡き数に入/ハイ/る', '～に入/ハイ/る'),
                          ('泣き通しに泣く', '～に泣く'),
                          ('慰めようもなく, 慰めようもないほど', '～もなく, ～もないほど'),
                          ('名乗りをあげる', '～をあげる'),
                          ('生爪をはがす', '～をはがす'),
                          ('南山の寿', '～の寿'),
                          ('難中の難事', '～の難事'),
                          ('苦味走った顔付', '～顔付'),
                          ('…を憎からず思う', '…を～思う'),
                          ('二軒建の家', '～の家'),
                          ('二三分の所である', '～の所である'),
                          (': 両', ''),
                          ('似たり寄ったりだ', '～だ'),
                          ('にっと笑う', '～笑う'),
                          ('二本差しの武士', '～の武士'),
                          ('二枚開きの戸', '～の戸'),
                          ('入御あらせられる', '～あらせられる'),
                          ('忍の一字', '～の一字'),
                          ('根上がりの松', '～の松'),
                          ('猫糞を極める', '～を極める'),
                          ('寝酒を飲む', '～を飲む'),
                          ('寝待ちの月', '～の月'),
                          ('寝耳に水', '～に水'),
                          ('のっぺりした顔', '～した顔'),
                          ('背汗の至りです', '～の至りです'),
                          ('背水の陣/ジン/を布く', '～の陣/ジン/を布く'),
                          ('白玉楼中の人となる', '～の人となる'),
                          (
                              ' (<i>напр. слоги</i> の <i>и</i> さ <i>в</i> ぼくのつくえ(樸の机) <i>будет</i> ぼノサくのつノサくえ)', ''),
                          ('果たせるかな', '～かな'),
                          ('八掛けで卸す', '～で卸す'),
                          ('八字の眉', '～の眉'),
                          (
                              '(<i>букв.</i> 8½ ри, <i>слово-каламбур: жареный батат по вкусу похож на каштан (по-японски '
                              '— кури), а слово «кури» можно фонетически написать знаками</i> 九里, <i>что значит «девять ри»; но так '
                              'как батат всё же не каштан, то он «8½ ри»</i>)', ''),
                          (
                              '(<i>первые четыре см.</i> <a href="#004-67-90">しく【四苦】</a>, <i>плюс муки разлуки любящих</i> '
                              '愛別離/アイベツリ/; <i>встречи с ненавистью</i> 怨憎会/オンゾウエ/, <i>неисполнения желаний</i> 求不得/クフトク/, '
                              '<i>расцвета духовных способностей</i> 五陰盛/ゴオンジョウ/)',
                              '(<i>первые четыре плюс муки разлуки любящих, встречи с ненавистью, неисполнения желаний, '
                              'расцвета духовных способностей</i>)'),
                          ('八色の姓/カバネ/', '～の姓/カバネ/'),
                          (' (真人/マヒト/, 朝臣/アソミ/, 宿爾/スクネ/, 忌寸/イミキ/, 道師/ミチノシ/, 臣/オミ/, 連/ムラジ/, 稲置/イナギ/)', ''),
                          ('(<i>китайской медицины, см.</i> <a href="#004-13-59">よもぎ</a>, '
                           '<a href="#007-20-64">くまつづら</a>, <a href="#005-63-47">おうばこ</a>, <a href="#007-73-25">おなもみ</a>, '
                           '<a href="#003-02-17">しょうぶ【菖蒲】</a>, <a href="#000-62-80">すいかずら</a>, <a href="#000-08-32">はこべ</a>, '
                           '<a href="#008-74-72">はす【蓮】</a>, <i>однако при перечислении этих трав</i> はす <i>пишется</i> 荷葉)',
                           '(<i>китайской медицины</i>)'),
                          ('はったりを行う', '～を行う'),
                          ('はっと思う ', '～思う '),
                          ('八方美人型の性格', '～の性格'),
                          ('…は初耳だ', '…は～だ'),
                          ('話し半分に聞く', '～に聞く'),
                          ('鼻持ちがならない', '～がならない'),
                          ('腹塞ぎに食べる', '～に食べる'),
                          ('; напр.</i> 弓二張/ユミフタハリ/ два лука', ''),
                          ('半堅気な女', '～な女'),
                          ('繁簡よろしきを得た', '～よろしきを得た'),
                          ('半眼で見る', '～で見る'),
                          (' (パ, ピ, プ, ぺ, ポ)', ''),
                          ('半身に構える', '～に構える'),
                          ('売文の徒', '～の徒'),
                          ('抜山蓋世の勇がある', '～の勇がある'),
                          ('荊棘がきに引っかく', '～に引っかく'),
                          ('万止むを得なければ', '～止むを得なければ'),
                          ('万斛の涙を注ぐ', '～の涙を注ぐ'),
                          ('万仭の谷', '～の谷'),
                          ('晩節を全うする', '～を全うする'),
                          ('万籟絶/タ/ゆ', '～絶/タ/ゆ'),
                          ('万緑叢中の紅一点', '～中の紅一点'),
                          ('贔屓目に見る', '～に見る'),
                          ('非業の死(最期)を遂げる', '～の死(最期)を遂げる'),
                          ('非業消滅のため', '～のため'),
                          ('直押しに押し寄せる', '～に押し寄せる'),
                          ('ひた走りに走る', '～に走る'),
                          ('左団扇で暮らす', '～で暮らす'),
                          ('必死必中の武器', '～の武器'),
                          ('筆誅を加える', '～を加える'),
                          ('筆陣を張る', '～を張る'),
                          ('一泡吹かせてやる', '～吹かせてやる'),
                          ('一芝居を打つ', '～を打つ'),
                          ('人好きのいい', '～のいい'),
                          ('人好きのわるい', '～のわるい'),
                          ('一堪りもなく', '～もなく'),
                          ('人っ子ひとり見えない(いない)', '～ひとり見えない(いない)'),
                          ('一照り照ると', '～照ると'),
                          ('一肌脱/ヌ/ぐ', '～脱/ヌ/ぐ'),
                          ('ひと降りほしいですね', '～ほしいですね'),
                          ('独り言を言う', '～を言う'),
                          ('独り佳居で暮らす', '～で暮らす'),
                          ('霏々として降る', '～として降る'),
                          ('百聞一見に如/シ/かず', '百聞一見に如かず'),
                          ('百万台に達する', '～に達する'),
                          ('百味の飲食', '～の飲食'),
                          ('飆々と鳴る', '～と鳴る'),
                          ('表裏反覆常/ツネ/がない', '～常/ツネ/がない'),
                          ('ひんひん啼く', '～啼く'),
                          ('びくともしない', '～ともしない'),
                          ('廟堂に立つ', '～に立つ'),
                          ('屛風倒しに倒れる', '～に倒れる'),
                          ('ぴょこんと頭を下げる', '～頭を下げる'),
                          ('ぴょんと跳ぶ', '～跳ぶ'),
                          ('ぴょんぴょん跳ぶ', '～跳ぶ'),
                          ('ぴよぴよ鳴く', '～鳴く'),
                          ('風声鶴嗅に驚く', '～に驚く'),
                          ('風前の灯(灯火)', '～の灯(灯火)'),
                          ('深爪を切る', '～を切る'),
                          ('不帰の客となる', '～の客となる'),
                          ('不許複製', '～複製'),
                          ('俯仰天地に愧じず(恥ずる所がない)', '～天地に愧じず, ～天地に恥ずる所がない'),
                          ('不拘留のまま検束される', '～のまま検束される'),
                          (', <i>напр.</i> 二千/フタセン/三百 две тысячи триста', ''),
                          ('二桁の数', '～の数'),
                          ('二手に別れる', '～に別れる'),
                          ('二目と見られない[ような]', '～と見られない[ような]'),
                          ('…に足を踏みかける', '…に足を～'),
                          ('不問に付する', '～に付する'),
                          ('…降りみ降らずみ雨が絶えず', '～雨が絶えず'),
                          ('踏ん切りがつかない', '～がつかない'),
                          ('不精髭を生やしている', '～を生やしている'),
                          ('物質不滅の法則', '～の法則'),
                          ('物質保存の原理', '～の原理'),
                          ('ぶつくさ言う', '～言う'),
                          ('炳として日月の如し', '～として日月の如し'),
                          (
                              ' へへ <i>брови</i>, のの <i>глаза</i>, も <i>нос</i>, へ <i>рот</i>, じ <i>овал лица)</i>',
                              '),'),
                          ('へへののもへじの生徒', '～の生徒'),
                          ('へべれけに酔う', '～に酔う'),
                          ('一臂の力を貸す', '～の力を貸す'),
                          ('…は未だしという所がある', '…は～という所がある'),
                          ('魚心あれば水心/ミズゴコロ/', '～あれば水心/ミズゴコロ/'),
                          ('打出の小槌', '～の小槌'),
                          ('売言葉に買い言葉', '～に買い言葉'),
                          ('叡聞に達する', '～に達する'),
                          ('鸚鵡返しに言う', '～に言う'),
                          ('哀々禁ぜず', '～禁ぜず'),
                          ('意気衝天の概がある', '～の概がある'),
                          ('いたずら盛りの年頃', '～の年頃'),
                          ('一寸試し五分試めしにする', '～五分試めしにする'),
                          ('一桃腐りて百桃/ヒャクトウ/損ず', '一桃腐りて百桃損ず'),
                          ('お手数ながら', '～ながら'),
                          ('親風を吹かせる', '～を吹かせる'),
                          ('会稽の恥をそそぐ', '～の恥をそそぐ'),
                          ('海陸両棲の動物', '～の動物'),
                          ('隔日発作のおこり', '～のおこり'),
                          ('籠抜け[詐欺]を働く', '～[詐欺]を働く'),
                          ('家庭型の女', '～の女'),
                          ('下風に立つ', '～に立つ'),
                          ('べそをかく', '～をかく'),
                          ('捧刀の礼', '～の礼'),
                          ('ほんこでめんこをする', '～でめんこをする'),
                          ('本腹の子', '～の子'),
                          ('忙中の閑 ', '～の閑 '),
                          ('; <i>ср.</i> にょうぼさつ', ''),
                          ('ぼたりと落ちた', '～と落ちた'),
                          ('ぼちゃんと飛び込む', '～と飛び込む'),
                          ('ぼってり太った', '～太った'),
                          ('凡慮の及ぶ所でない', '～の及ぶ所でない'),
                          ('前先が見えない', '～が見えない'),
                          ('目蔭をさす', '～をさす'),
                          ('紛う方なき', '～方/こと/なき'),
                          ('まじまじ[と]見る', '～[と]見る'),
                          ('末筆ながら', '～ながら'),
                          ('まんじりともしない', '～ともしない'),
                          ('見極めのつかぬ', '～のつかぬ'),
                          ('右枕に寝る', '～に寝る'),
                          ('身ぐるみはぎとられる', '～はぎとられる'),
                          ('三桁の数', '～の数'),
                          ('見だてがない', '～がない'),
                          ('三つ巴の争い(混戦)', '～の争い, ～の混戦'),
                          ('身分相応に暮らす(やって行く)', '～に暮らす, ～やって行く'),
                          ('無可有の郷/サト/', '～の郷/サト/'),
                          ('向かっ腹を立てる', '～を立てる'),
                          ('無原罪の御宿/オヤド/', '～の御宿/オヤド/'),
                          ('無駄飯を食う', '～を食う'),
                          ('胸三寸に納める', '～に納める'),
                          ('盲打ちに打つ', '～に打つ'),
                          ('盲撃ちに撃つ', '～に撃つ'),
                          ('盲判を押す', '～を押す'),
                          ('目覚め勝ちの一夜を過ごす', '～の一夜を過ごす'),
                          ('目潰しを食わせる, 目潰しに…を投げる', '～を食わせる, ～に…を投げる'),
                          ('目端がきく', '～がきく'),
                          ('めらめら燃え上がる', '～燃え上がる'),
                          ('もうと鳴く', '～と鳴く'),
                          ('…の話で持ち切りだった', '…の話で～だった'),
                          ('…に身を持ち崩す', '…に身を～'),
                          ('勿怪の幸', '～の幸'),
                          ('本立ちのよい', '～のよい'),
                          ('諸肌を脱ぐ(脱いでいる)', '～を脱ぐ(脱いでいる)'),
                          ('門前雀羅を張る', '～を張る'),
                          ('薬石効/コウ/なく', '～効/コウ/なく'),
                          ('役人風を吹かす', '～を吹かす'),
                          ('自暴酒をあふる', '～をあふる'),
                          ('八又の大蛇/オロチ/', '～の大蛇/オロチ/'),
                          ('湯気にあたる', '～にあたる'),
                          ('夢聊 <i>с отриц.</i>', '<i>с отриц.</i>'),
                          ('夢聊も忘れない', '～も忘れない'),
                          ('夢枕に立つ', '～に立つ'),
                          ('俑を作る', '～を作る'),
                          ('様です(だ)', '～です(だ)'),
                          ('よう[御座います]', '～[御座います]'),
                          ('よくせきの事で', '～の事で'),
                          ('横車を押す', '～を押す'),
                          ('横抱きにかかえる', '～にかかえる'),
                          ('横手を打つ', '～を打つ'),
                          ('横飛びに飛んでゆく', '～に飛んでゆく'),
                          ('横隣りの人', '～人'),
                          ('横薙ぎに払う', '～に払う'),
                          ('…由です', '…～です'),
                          ('由ある ', '～ある '),
                          ('由ある人', '～ある人'),
                          ('余憤を漏らす', '～を漏らす'),
                          ('寄辺なぎさの捨小舟/ステコブネ/', '～の捨小舟/ステコブネ/'),
                          ('埒外に出る', '～に出る'),
                          (
                              '<i>(в древнем Китае:</i> 礼 <i>нормы общественного поведения</i>, 楽 <i>музыка</i>, '
                              '射 <i>стрельба из лука</i>, 御/ギョ/ <i>управление колесницей</i>, 書 <i>письмо</i>, '
                              '数 <i>счёт)</i>',
                              '<i>(в древнем Китае: нормы общественного поведения, музыка, стрельба из лука, управление колесницей, письмо, счёт)'),
                          ('陸路[を通って]', '～[を通って]'),
                          ('立錐の余地もない', '～の余地もない'),
                          ('両袖付の机', '～の机'),
                          ('りーんと呼鈴を鳴らす', '～と呼鈴を鳴らす'),
                          ('老残の身', '～の身'),
                          ('論陣を張る', '～を張る'),
                          ('論歩を進める', '～を進める'),
                          (' (<i>ср.</i> あんか【案下】, <a href="#001-61-23">おんちゅう【御中】</a>)', ''),
                          ('和局を結ぶ', '～を結ぶ'),
                          ('わなわな震える', '～震える'),
                          ('インキ止の紙', '～の紙'),
                          ('カメラ顔のいい', '～のいい'),
                          ('(<i>букв.</i> с отметкой «ки», <i>сокр.</i> китигаи 気違い)',
                           '(<i>букв.</i> с отметкой «ки», <i>сокр.</i> 気違い)'),
                          ('ジェスチャーまじりの話しぶり', '～の話しぶり'),
                          ('(ストリキニン <i>англ.</i> strychnin, ストリキニーネ <i>голл.</i> strychnine)',
                           '(<i>от англ.</i> strychnin, <i>голл.</i> strychnine)'),
                          ('タイム・アップだ', '～だ'),
                          ('てくしーで行く', '～で行く'),
                          ('ヌーボー式の人', '～の人'),
                          ('バベルの塔', '～の塔'),
                          ('ビキニの灰/ハイ/', '～の灰/ハイ/'),
                          ('フォールに持ち込む', '～に持ち込む'),
                          ('ホーム・ルーム指導教師', '～指導教師'),
                          ('(<i>кит.</i> мэй фацзы 没法子)', '(<i>кит.</i> 没法子 [мэй фацзы])'),
                          (
                              ' (<i>слово-каламбур, основанное на графическом разложении иероглифа</i> 只 тада)', ''),
                          (
                              ' (<i>прост. слово-каламбур — прочтённый по частям иероглиф</i> 頗, <i>см.</i> <a href="#002-34-54">すこぶる</a>)',
                              ''),
                          ('があがあ鳴く', '～鳴く'),
                          ('がぶがぶ飲む', '～飲む'),
                          ('がらんがらんと鳴る', '～と鳴る'),
                          ('菊水の紋', '～の紋'),
                          (
                              ' <i>(поскольку иероглифы</i> 七十七 <i>в скорописном написании похожи на иероглиф</i> 喜, <i>написанный тоже скорописью)</i>',
                              ''),
                          ('鬼籍に入/ハイ/る', '～に入/ハイ/る'),
                          ('きゃんきゃん鳴く', '～鳴く'),
                          ('旧遊の地', '～の地'),
                          ('狭斜の巷', '～の巷'),
                          ('響板のエゾ', '～のエゾ'),
                          ('器量好みの人', '～の人'),
                          ('錦上更に花を添える', '～更に花を添える'),
                          ('義師を起す', '～を起す'),
                          ('ぎょっだね', '～だね'),
                          ('苦汁をなめる', '～をなめる'),
                          ('汲み立ての水', '～の水'),
                          ('ぐうの音 ', '～の音/ね/ '),
                          ('化粧くずれを直す', '～を直す'),
                          ('欠を補う', '～を補う'),
                          ('拳匪の乱', '～の乱'),
                          ('げらげら笑う', '～笑う'),
                          ('口角泡/アワ/を飛ばして', '～泡/アワ/を飛ばして'),
                          ('哄然と笑う', '～と笑う'),
                          ('巧遅は拙速に如/シ/かず', '巧遅は拙速に如かず'),
                          ('口辺に微笑をたたえて', '～に微笑をたたえて'),
                          ('小首をかしげる(傾ける, ひねる)', '～をかしげる, ～を傾ける, ～をひねる'),
                          ('心待ちに待つ', '～に待つ'),
                          ('滑稽交じりに話す', '～に話す'),
                          ('<i>форма</i> なさい <i>в женской речи, см.</i> <a href="#006-93-67">なさい</a>',
                           '<i>в женской речи</i> <i>см.</i> <a href="#006-93-67">なさい</a>'),
                          ('両極端は相会す', '＝両極端は～す'),
                          ('顔を赤らめる', '＝顔を～'),
                          ('夜の明け抜けに', '＝夜の～に'),
                          ('どうぞ悪しからず[思って下さい]', '＝どうぞ～[思って下さい]'),
                          ('二人は熱々だ', '＝二人は～だ'),
                          ('疑心暗鬼を生ず', '＝疑心～を生ず'),
                          ('故人遺愛の', '＝故人～の'),
                          ('夫婦は異身同体である', '＝夫婦は～である'),
                          ('[打って]一丸とする', '＝[打って]～とする'),
                          ('赤ん坊にいないいないばーをやって見る', '＝赤ん坊に～をやって見る'),
                          (' 得意の鼻をうごめかす', ' ＝得意の鼻を～'),
                          ('一点の打処もない', '＝一点の～もない'),
                          ('顔を俯向ける', '＝顔を～'),
                          ('時を得顔に振舞う', '＝時を～に振舞う'),
                          ('そんなことがあるかしら！あるとも、あるとも大ありだ', '＝そんなことがあるかしら！あるとも、あるとも～'),
                          ('子供のお仕置き', '＝子供の～'),
                          ('これでおつもりだよ', '＝これで～だよ'),
                          ('人口に膾炙する', '＝人口に～する'),
                          ('意に介する', '＝意に～'),
                          ('涙に掻き暮れる', '＝涙に～'),
                          ('御加餐を祈る', '＝御～を祈る'),
                          ('鼻をかむ', '＝鼻を～'),
                          ('声を嗄らす', '＝声を～'),
                          ('世は刈菰と乱れた', '＝世は～と乱れた'),
                          ('声が嗄れる ', '＝声が～ '),
                          ('犬の川端歩き', '＝犬の～'),
                          ('酒の燗をする(付ける)', '＝酒の～をする(付ける)'),
                          ('事務を簡捷にする', '＝事務を～にする'),
                          ('生死の関頭に立つ', '＝生死の～に立つ'),
                          ('もう堪忍袋の緒が切れた', '＝もう～の緒/お/が切れた'),
                          ('それが聞き納めだった', '＝それが～だった'),
                          ('妙な気先が動いて', '＝妙な～が動いて'),
                          ('私は着た切り雀だ', '＝私は～だ'),
                          ('万事休す', '＝万事～'),
                          ('半/ナカバ/香落ちで指す', '＝半/ナカバ/～で指す'),
                          ('毛の癖直しをする', '＝毛の～をする'),
                          ('彼は糞落着きに落着いている', '＝彼は～に落着いている'),
                          ('日が暮れなずむ', '＝日が～'),
                          ('鹿児島くんだりへ行く', '＝鹿児島～へ行く'),
                          ('獅子の子落とし', '＝獅子の～'),
                          ('風邪を拗らす', '＝風邪を～'),
                          ('家/イエ/の子郎党', '＝家/イエ/の～'),
                          ('小刀を逆手に持つ(取る)', '＝小刀を～に持つ(取る)'),
                          ('言うも更なり', '＝言うも～'),
                          ('日/ヒ/既に三竿', '＝日/ヒ/既に～'),
                          ('舌にざらつく', '＝舌に～'),
                          ('顔を顰める', '＝顔を～'),
                          ('身に泌み感じる', '＝身に～'),
                          ('夜は深々と更けて行く', '＝夜は～と更けて行く'),
                          ('その言は今もなお耳底にある', '＝その言は今もなお～にある'),
                          ('うんともすんとも言わない не сказать', '＝うんとも～とも言わない не сказать'),
                          ('うんともすんとも言わない не ответить', '～ともすんとも言わない не ответить'),
                          ('面/ツラ/の皮が千枚張りだ', '＝面/ツラ/の皮が～だ'),
                          ('模様を染め出す', '＝模様を～'),
                          ('風がそよそよ吹く', '＝風が～吹く'),
                          ('小便をたれる', '＝小便を～'),
                          ('そのたんびに', '＝その～に'),
                          ('不信任の弾劾案', '＝不信任の～'),
                          ('お使い立てをしまして申訳ありません', '＝お～をしまして申訳ありません'),
                          ('口を噤む', '＝口を～'),
                          ('五日付け[の]', '＝五日～[の]'),
                          ('権威づけの為に', '＝権威～の為に'),
                          ('目をつぶる', '＝目を～'),
                          ('酒を手酌で飲む', '＝酒を～で飲む'),
                          ('『済みません』てな事を言う', '＝『済みません』～事を言う'),
                          ('両刀を手挟む', '＝両刀を～'),
                          ('写真を摂る', '＝写真を～'),
                          ('最後の一周は彼の独走の観があった', '＝最後の一周は彼の～の観があった'),
                          ('二十円ドタを割る', '＝二十円～を割る'),
                          ('目を泣き潰す', '＝目を～'),
                          ('事も無げ', '＝事も～'),
                          ('思案の投げ首', '＝思案の～'),
                          ('髪をなで付けにする', '＝髪を～にする'),
                          ('怒ったのなんのって', '＝怒ったの～'),
                          ('犬の逃げ吠え', '＝犬の～'),
                          ('たけのこのようににょきにょき出ている', '＝たけのこのように～出ている'),
                          ('烏の濡羽色', '＝烏の～'),
                          ('一人乗りの', '＝一人～の'),
                          ('鯛の浜焼', '＝鯛の～'),
                          ('涙をふるって馬謖を斬る', '＝涙をふるって～を斬る'),
                          ('眼をぱちくりさせる', '＝眼/め/を～させる'),
                          ('夜の引き明けに', '＝夜の～に'),
                          ('贔屓の引き倒し', '＝贔屓の～'),
                          ('お膝繰りを願います', '＝お～を願います'),
                          ('ざっと(急いで)一風呂浴びる', '＝急いで(ざっと)～浴びる'),
                          ('屁を放る', '＝屁を～'),
                          ('水の中をびちゃびちゃ歩く', '＝水の中を～歩く'),
                          ('風がびゅうびゅう吹く', '＝風が～吹く'),
                          ('タバコを吹かす курить', '＝タバコを～ курить'),
                          ('夜/ヨル/を更かす', '＝夜を～'),
                          ('風の吹き回し', '＝風の～'),
                          ('裏面に伏在する', '＝裏面に～する'),
                          ('足を踏み入れる', '＝足を～'),
                          ('[足の]踏み所もない', '＝[足の]～もない'),
                          ('[足を]踏み外ずす', '＝[足を]～'),
                          ('気がふれる', '＝気が～'),
                          ('…の便をはかって', '…の～をはかって'),
                          ('用が便じる', '＝用が～'),
                          ('どうぞ御放念下さいますよう', '＝どうぞ御～下さいますよう'),
                          ('御母堂', '＝御～'),
                          ('帽子を目深に被る', '＝帽子を～に被る'),
                          ('鰯の味醂干し', '＝鰯の～'),
                          ('人にめいめい向き向きある', '＝人にめいめい～ある'),
                          ('あの店は儲け主義だ', '＝あの店は～だ'),
                          ('煙がもくもく[と]出る', '＝煙が～[と]出る'),
                          (' 気の持ちよう', ' ＝気の～'),
                          ('袴の股立ちを高くとる', '＝袴の～を高くとる'),
                          ('八幡の藪知らず', '＝八幡/やわた/の～'),
                          ('髪の結い方', '＝髪の～'),
                          ('あの男は何事も行き当たりばったり主義だ', '＝あの男は何事も～だ'),
                          ('この本は読み出がない', '＝この本は～がない'),
                          ('人間は万物の霊長', '＝人間は万物～'),
                          ('彼は笑い上戸だ', '＝彼は～だ'),
                          ('彼は悪口屋だ', '＝彼は～だ'),
                          ('彼は悪擦れしている', '＝彼は～している'),
                          ('(<i>отриц. форма гл.</i> <a href="#006-14-76">しく【如く】</a>) ', ''),
                          ('大所高所から物を見る', '～から物を見る'),
                          ('<i>связ. кн. отриц. форма гл.</i>', '<i>кн. отриц. форма гл.</i>'),
                          ('(<i>с 1949 г., ср.</i> <a href="#007-97-88">ていだい【帝大】</a>)',
                           '(<i>с 1949 г.</i>)'),
                          ('夜が明け離れた', '＝夜が｜た～'),
                          ('捏ね交ぜて一つにした', '｜て～一つにした'),
                          ('身を挺して', '＝身を｜して～'),
                          ('取って付けた様な', '｜た～様な'),
                          ('…は両眼を泣き腫らしていた', '…は両眼を｜して～いた'),
                          ('…は記憶から拭い去られた', '…は記憶から｜られた～'),
                          ('羽/ハネ/の生え揃った', '＝羽/ハネ/の｜った～'),
                          ('罷り間違えば, 罷り間違っても', '｜えば～, ｜って～も'),
                          ('目に物見せてやる', '＝目に｜て～やる'),
                          ('<i>от</i> <a href="#001-55-99">ベースI 1</a> <i>и англ.</i> up',
                           '<i>от</i> ベース <i>и англ.</i> up'),
                          ('(<i>от</i> ごねる <i>и</i> 得 току)', '(<i>от</i> ごねる <i>и</i> 得)'),
                          ('(<i>от сокр.</i> мосурин モスリン)', '(<i>от сокр.</i> モスリン)'),
                          ('<i>связ.:</i> お待ち遠様/オマチドウサマ/[でした]',
                           '<i>связ.:</i> omachidousama 【お待ち遠様】 [deshita]'),
                          (
                              '<i>побуд. залог гл.</i> つく【尽く】, <i>т. е. письменной формы гл.</i> <a href="#008-04-11">つきる</a>;',
                              '<i>побуд. залог гл.</i> <a href="#003-19-230">つく</a>'),
                          ('(<i>редко</i> 供わる) (<i>о себе тк.</i> 具わる)\n: …が～',
                           '(<i>о себе тк.</i> 具わる) (<i>редко</i> 供わる) …が～'),
                          ('…を記録に残す, …の記録をとる <i>см. выше</i> ～する;', 'を記録に残す, の記録をとる <i>см. выше</i> する'),
                          ('(<i>в конце предл. после</i> かも)', '…かも～ <i>в конце предл.</i>'),
                          ('(<i>как 2-й компонент сложн. сл.</i> ざいく) [мелкие] изделия, поделки;', 'ざいく'),
                          (' (<i>напр.</i> кударидзака <i>вм.</i> кударисака)', ''),
                          (
                              '2) как и следовало ожидать (<i>неправ. вм.</i> <a href="#000-83-01">かぜん【果然】</a>).',
                              '2) <i>неправ.</i> как и следовало ожидать.'),
                          ('(<i>вм.</i> 言わざる) ', ''),
                          ('(<i>вм.</i> 聞かざる) ', ''),
                          ('(<i>вм.</i> 見ざる) ', ''),
                          ('～の <i>см.</i> <a href="#003-79-98">かいせんじょう(～の)</a>.',
                           '～の <i>см.</i> <a href="#003-79-98">かいせんじょう(～[の])</a>.'),
                          ('～にする <i>см.</i> <a href="#003-80-49">こ【粉】(～にする)</a>.',
                           '～にする <i>см.</i> <a href="#003-80-49">こ【粉】(～にする)</a>.'),
                          ('粉にする(ひく) растирать в порошок; перемалывать на муку; толочь, молоть;',
                           '～にする, ～にひく растирать в порошок; перемалывать на муку; толочь, молоть;'),
                          ('<i>см.</i> <a href="#008-81-40">しりごみ (～する)</a>.',
                           '<i>см.</i> <a href="#008-81-40">しりごみ (～[を]する)</a>.'),
                          ('～[の] <i>см.</i> <a href="#003-36-07">じもと (～の)</a>.',
                           '～[の] <i>см.</i> <a href="#003-36-07">じもと (～[の])</a>.'),
                          ('ぶっ違いに置く', '～に置く'),
                          ('頰を火照らす <i>см.</i> <a href="#002-55-30">ほてる(頰が火照る)</a>;',
                           '＝頰を～ <i>см.</i> <a href="#002-55-30">ほてる(＝頰が～)</a>;'),
                          ('頰がほてる щёки пылают, лицо залила краска;',
                           '＝頰が～ щёки пылают, лицо залила краска;'),
                          ('<i>прост. что-л. рассчитанное на дешёвый эффект</i>;',
                           '<i>прост.</i> что-л. рассчитанное на дешёвый эффект;'),
                          ('<i>кн. заставлять быть таким:</i>',
                           '<i>кн.</i> заставлять быть таким:'),
                          ('<i>ономат. звук кипящего масла (при жаренье)</i>;',
                           '<i>ономат.</i> звук кипящего масла (при жаренье);'),
                          ('<i>ономат. отрывистый звук хлопушки и т. п.</i>;',
                           '<i>ономат.</i> отрывистый звук <i>хлопушки и т. п.</i>;'),
                          ('旅慣れた', '｜た～'),
                          ('旅慣れない', '｜ない～'),
                          ('1) <i>ономат. о выстреле, стуке или ударе обо что-л.</i>;',
                           '1) <i>ономат.</i> о выстреле, стуке или ударе <i>обо что-л.</i>;'),
                          ('<i>после сущ. подчёркивающая частица:</i>',
                           '<i>после сущ.</i> подчёркивающая частица:'),
                          ('待ち切れない', '｜れない～'),
                          ('足掻きがとれない', '～がとれない'),
                          ('胡座をかく', '～をかく'),
                          ('呆気に取られる', '～に取られる'),
                          ('口をあんぐりあいて', '＝口を～あいて'),
                          ('生き馬の目を抜く[ような事をする]', '～の目を抜く[ような事をする]'),
                          ('一縷の煙', '～の煙'),
                          ('一縷の命', '～の命'),
                          ('一縷の望みを抱く', '～の望みを抱く'),
                          ('一掬の水', '～の水'),
                          ('一掬の涙なきを得ない', '～のなきを得ない'),
                          ('一挙手一投足の労/ロウ/', '～の労/ロウ/'),
                          ('一挙手一投足に過ぎない', '～に過ぎない'),
                          ('一再ならず', '～ならず'),
                          ('いっそ[のこと]', '～[のこと]'),
                          ('命冥加が尽きた', '～が尽きた'),
                          ('命冥加な奴だ', '～な奴だ'),
                          ('うんうん言っている', '～言っている'),
                          ('江戸前の魚', '～の魚'),
                          ('江戸前の料理', '～の料理'),
                          ('王手飛車を食う', '～を食う'),
                          ('王手飛車に掛ける', '～に掛ける'),
                          ('臆面もない', '～もない'),
                          ('烏滸の沙汰/サタ/', '～の沙汰/サタ/'),
                          ('心を落ち着けて', '＝心を｜て～'),
                          ('お目にかかる', '～にかかる'),
                          ('お目にかける', '～にかける'),
                          ('…のお目に止まる', '…の～に止まる'),
                          ('思い半ばに過ぎぬ', '～に過ぎぬ'),
                          ('思う壷に嵌る', '～に嵌る'),
                          ('開巻第一に', '～第一に'),
                          ('回天の偉業(事業)', '～の偉業(事業)'),
                          ('回天の力', '～の力'),
                          ('固唾を飲む', '～を飲む'),
                          ('かちんと鳴る', '～と鳴る'),
                          ('かちんかちん', '～かちん'),
                          ('紙一重である', '～である'),
                          ('からっと晴れ上がる', '～晴れ上がる'),
                          ('からっと揚げる', '～揚げる'),
                          ('かんかん鳴る', '～鳴る'),
                          ('かんかん日が照っている', '～日が照っている'),
                          ('かんかんに怒る', '～に怒る'),
                          ('三学年生', '＝三～'),
                          ('がちゃがちゃいう音', '～いう音'),
                          ('がちゃがちゃさせる(音を立てる)', '～させる, ～音を立てる'),
                          ('がぶりと飲み込む', '～と飲み込む'),
                          ('がぶりと嚙み付く', '～と嚙み付く'),
                          ('がやがや騒ぐ', '～騒ぐ'),
                          ('がらがらいう音', '～いう音'),
                          ('がらがらと音がする', '～と音がする'),
                          ('がりがり音を立てる', '～音を立てる'),
                          ('がりがり嚙む', '～嚙む'),
                          ('がんがん鳴る(鳴らす)', '～鳴る, ～鳴らす'),
                          ('喜の字', '～の字'),
                          ('喜の祝', '～の祝'),
                          ('機影を現わす', '～を現わす'),
                          ('機影を没する', '～を没する'),
                          ('気位が高い', '～が高い'),
                          ('気随気儘な', '～気儘な'),
                          ('きちきちに詰まっている', '～に詰まっている'),
                          ('時間きちきちに', '＝時間～に'),
                          ('気骨が折れる', '～が折れる'),
                          ('気骨の折れる', '～の折れる'),
                          ('きーきーいう音', '～いう音'),
                          ('きーきー鳴く', '～鳴く'),
                          ('ぎゃふんと言わせる', '～言わせる'),
                          ('ぎゃふんと言う(参る)', '～言う, ～参る'),
                          ('ぎゅうぎゅう鳴る', '～鳴る'),
                          ('ぎゅうぎゅう鳴る音', '～鳴る音'),
                          ('ぎゅーと言う音', '～と言う音'),
                          ('ぎゅーと開く', '～と開く'),
                          ('ぎょろぎょろ見る', '～見る'),
                          ('くすくす笑う', '～笑う'),
                          ('くたくた疲れる', '～疲れる'),
                          ('汗でカラーがくたくたになった', '＝汗でカラーが～になった'),
                          ('くたくた煮る', '～煮る'),
                          ('口占で見る', '～で見る'),
                          ('口幅ったい事を言う', '～事を言う'),
                          ('くつくつ笑う', '～笑う'),
                          ('目がくらくらする', '＝目が～する'),
                          ('くらくら煮え立つ', '～煮え立つ'),
                          ('暮れ残る空の色', '～空の色'),
                          ('暮れ残る白百合/シラユリ/', '～白百合/シラユリ/'),
                          ('くんくんと泣く', '～と泣く'),
                          ('くんくん嗅ぐ', '～嗅ぐ'),
                          ('ぐいっと飲む', '～飲む'),
                          ('ぐいと引く', '～引く'),
                          ('ぐうぐう寝る', '～寝る'),
                          ('腹がぐうぐう鳴る', '＝腹が～鳴る'),
                          ('ぐうの音も出ない', '～も出ない'),
                          ('ぐるぐる回る', '～回る'),
                          ('ぐるぐる巻く(巻きにする)', '～巻く, ～巻きにする'),
                          ('けりが付く', '～が付く'),
                          ('鳧を付ける', '～を付ける'),
                          ('乾坤一擲の勝負をやる', '～の勝負をやる'),
                          ('玄関払いを食わす', '～を食わす'),
                          ('玄関払いを食わさせる', '～を食わさせる'),
                          ('好学の念', '～の念'),
                          ('好学の士', '～の士'),
                          ('浩然の気', '～の気'),
                          ('浩然の気を養う', '～の気を養う'),
                          ('孤高の生活', '～の生活'),
                          ('孤高を持/ジ/す', '～を持/ジ/す'),
                          ('…するを快しとする', '…するを～とする'),
                          ('…するを快しとしない', '…するを～としない'),
                          ('ころころ転げる', '～転げる'),
                          ('あの人はころころとふとっている', '＝あの人は～とふとっている'),
                          ('ころころと鳴く', '～と鳴く'),
                          ('狐がこんこんと鳴く', '＝狐が～と鳴く'),
                          ('こんこんと咳をする', '～と咳をする'),
                          ('こんこんと打つ', '～と打つ'),
                          ('雪がこんこんと降る', '＝雪が～と降る'),
                          ('滾々と流れる', '～と流れる'),
                          ('滾々として尽きない', '～として尽きない'),
                          ('ごうと通る', '～と通る'),
                          ('ごうという彼の音', '～という彼の音'),
                          ('轟々という音', '～という音'),
                          ('轟々と鳴る', '～と鳴る'),
                          ('呉下の阿蒙/アモウ/である', '～の阿蒙/アモウ/である'),
                          ('ごぼごぼいう音', '～いう音'),
                          ('ごぼごぼ音がする', '～音がする'),
                          ('目を覚ます', '＝目を～'),
                          ('酔いを醒ます', '＝酔いを～'),
                          ('迷い(迷夢)を醒ます', '＝迷い(迷夢)を～'),
                          ('[目が]覚める', '＝[目が]～'),
                          ('目のさめるような美人', '＝目の～ような美人'),
                          ('酔いが醒める', '＝酔いが～'),
                          ('ざくざく音がする', '～音がする'),
                          ('砂をざくざく踏む', '＝砂を～踏む'),
                          ('ざーざー降りの雨', '～降りの雨'),
                          ('ざーざー流れる', '～流れる'),
                          ('四季折々の花', '～の花'),
                          ('四季折々の眺めがある', '～の眺めがある'),
                          ('しくしく泣く', '～泣く'),
                          ('腹がしくしく痛む', '＝腹が～痛む'),
                          ('死出の旅', '～の旅'),
                          ('死出の山', '～の山'),
                          ('芝居道の人々', '～の人々'),
                          ('芝居道に入/ハイ/る', '～に入/ハイ/る'),
                          ('…を斜に構える', '…を～に構える'),
                          ('体を斜に構える', '＝体を～に構える'),
                          ('衆口一致して', '～一致して'),
                          ('手裏に掌握する', '～に掌握する'),
                          ('手裏を脱する', '～を脱する'),
                          ('しゅーと言う音', '～と言う音'),
                          ('しゅーしゅーいう', '～いう'),
                          ('性懲りもなく', '～もなく'),
                          ('神も照覧あれ', '＝神も～あれ'),
                          ('雨がしょぼしょぼ降る', '＝雨が～降る'),
                          ('しょぼしょぼに濡れる', '～に濡れる'),
                          ('しょぼしょぼした目', '～した目'),
                          ('目をしょぼしょぼさせる', '＝目を～させる'),
                          ('しょぼしょぼした髯', '～した髯'),
                          ('十目の見る所', '～の見る所'),
                          ('上聞に入れる', '～に入れる'),
                          ('上聞に達する', '～に達する'),
                          ('すかを食わす', '～を食わす'),
                          ('すかを食う', '～を食う'),
                          ('足を滑らす', '＝足を～'),
                          ('口をすべらした', '＝口を｜した～'),
                          ('専門外の人', '～の人'),
                          ('それは専門外だ', '＝それは～だ'),
                          ('千慮の一失', '～の一失'),
                          ('ぜいぜい言う', '～言う'),
                          ('蘇張の弁/ベン/', '～の弁/ベン/'),
                          ('蘇張の弁を振う', '～の弁を振う'),
                          ('蒼茫たる大海', '～たる大海'),
                          ('蒼茫と暮れて行く', '～と暮れて行く'),
                          ('俎上の魚', '～の魚'),
                          ('俎上に載せる', '～に載せる'),
                          ('俎上に置く(載せる)', '～に置く, ～に載せる'),
                          ('耳を欹てる', '＝耳を～'),
                          ('目を欹てる', '＝目を～'),
                          ('側杖を食う', '～を食う'),
                          ('側杖を食って殺される', '～を食って殺される'),
                          ('あぶないぞ！', '＝あぶない～！'),
                          ('誰ぞ来たのか？', '＝誰～来たのか？'),
                          ('ぞっきに出す', '～に出す'),
                          ('ぞっきで売る', '～で売る'),
                          ('大過なきに近い', '～に近い'),
                          ('大過無きを得/エ/る', '～を得/エ/る'),
                          ('高砂のじいさんとばあさん', '～じいさんとばあさん'),
                          ('高砂の松', '～の松'),
                          ('高枕で寝る', '～で寝る'),
                          ('世間知らずの高枕', '＝世間知らずの～'),
                          ('お互っこだ', '＝お～だ'),
                          ('他見を憚る', '～を憚る'),
                          ('他見を許さない', '～を許さない'),
                          ('多情多恨の一生を送る', '～の一生を送る'),
                          ('多情多恨の志士', '～の志士'),
                          ('立て付けが良い', '～が良い'),
                          ('立て付けの悪い戸', '～の悪い戸'),
                          ('他聞を憚る', '～を憚る'),
                          ('上から水がたらたら落ちる', '＝上から水が～落ちる'),
                          ('汗をたらたら流している', '＝汗を～流している'),
                          ('駄法螺を吹く', '～を吹く'),
                          ('断金の友', '～の友'),
                          ('断金の交り(契り)', '～の交り(契り)'),
                          ('ちくりと刺す', '～と刺す'),
                          ('ちくりと痛む', '～と痛む'),
                          ('ちちと啼く', '～と啼く'),
                          ('ちびりちびり飲む', '～飲む'),
                          ('ちゅーちゅー鳴く', '～鳴く'),
                          ('無用の長物', '＝無用の～'),
                          ('無用の長物視する', '＝無用の～視する'),
                          ('ちょきちょきはさみ切る', '～はさみ切る'),
                          ('ちょきん切る', '～切る'),
                          ('ちょきんと立つ', '～と立つ'),
                          ('犬がちんちんをする', '＝犬が～をする'),
                          ('月回りがいい', '～がいい'),
                          ('今月は月回りが悪い', '＝今月は～が悪い'),
                          ('息をつく', '＝息を～'),
                          ('嘘を吐く', '＝嘘を～'),
                          ('つべこべ喋べる', '～喋べる'),
                          ('つべこべ返答する', '～返答する'),
                          ('つべこべ云わずに', '～云わずに'),
                          ('つべこべ言うな', '～言うな'),
                          ('茶の出殻', '＝茶の～'),
                          ('繭の出殻', '＝繭の～'),
                          ('飛び抜けて一番になる', '～一番になる'),
                          ('飛び抜けて一着', '～一着'),
                          ('途方に暮れる', '～に暮れる'),
                          ('途方に暮れさせる', '～に暮れさせる'),
                          ('途方もない', '～もない'),
                          ('同窓[の友]', '～[の友]'),
                          ('どうのこうの言わずに', '～言わずに'),
                          ('どうのこうの言って承諾しない', '～言って承諾しない'),
                          ('どかっと腰を下ろす', '～腰を下ろす'),
                          ('相場がどかっと下がった', '＝相場が～下がった'),
                          ('胸をどきんと衝く', '＝胸を～衝く'),
                          ('…の度胆を抜く', '…の～を抜く'),
                          ('度胆を抜かれる', '～を抜かれる'),
                          ('呑舟の魚', '～の魚'),
                          ('呑舟の魚を漏らす', '～の魚を漏らす'),
                          ('何食わぬ顔で(顔をして)', '～顔で, ～顔をして'),
                          ('何食わぬ顔をする', '～顔をする'),
                          ('波間に漂う', '～に漂う'),
                          ('波間をかいくぐる', '～をかいくぐる'),
                          ('二進も三進も行かない', '～行かない'),
                          ('二進も三進も動けない', '～動けない'),
                          ('二の句が継げない', '～が継げない'),
                          ('二の句を継げなくする', '～を継げなくする'),
                          ('ここは寝心地がよい', '＝ここは～がよい'),
                          ('寝心地のよい床', '～のよい床'),
                          ('穿き心地の良い靴', '～の良い靴'),
                          ('八分の一', '～の一'),
                          ('はっしと切りつける', '～切りつける'),
                          ('打込む刀をはっしと受け止める', '＝打込む刀を～受け止める'),
                          ('お話中', '＝お～'),
                          ('お話中ですか', '＝お～ですか'),
                          ('お話中失礼ですが', '＝お～失礼ですが'),
                          ('どうも憚りさま', '＝どうも～'),
                          ('憚りさまですが', '～ですが'),
                          ('空に憚る', '＝空に～'),
                          ('大角[の笛]', '～[の笛]'),
                          ('大角を吹く', '～を吹く'),
                          ('反間の策', '～の策'),
                          ('反間苦肉の計', '～苦肉の計'),
                          ('敵軍に反間を放つ', '＝敵軍に～を放つ'),
                          ('はーはー言う', '～言う'),
                          ('息をきらしてはーはーいう', '＝息をきらして～いう'),
                          ('場数を踏む', '～を踏む'),
                          ('縛の縄', '～の縄'),
                          ('縛につく', '～につく'),
                          ('莫逆の友', '～の友'),
                          ('莫逆の交わり', '～の交わり'),
                          ('いくら言っても馬耳東風だ', '＝いくら言っても～だ'),
                          ('馬耳東風に聞き流す', '～に聞き流す'),
                          ('ばちゃんとドアを締める', '～[と]ドアを締める'),
                          ('水にばちゃんと落ちる', '＝水に～[と]落ちる'),
                          ('ばつを合わせる', '～を合わせる'),
                          ('ばつが悪い', '～が悪い'),
                          ('ばらばら撒く', '～撒く'),
                          ('ばらばら降る', '～降る'),
                          ('ばりばり引っ掻く', '～引っ掻く'),
                          ('ばりばり嚙む', '～嚙む'),
                          ('仕事をばりばりやる', '＝仕事を～やる'),
                          ('万芸に通じている人', '～に通じている人'),
                          ('万芸に通じて一芸を成/ナ/さない', '万芸に通じて一芸を成さない'),
                          ('万丈の気焰を上げる', '～の気焰を上げる'),
                          ('波瀾万丈だった時代', '＝波瀾～だった時代'),
                          ('万乗の位に上がる', '～の位に上がる'),
                          ('万乗の君/キミ/', '～の君/キミ/'),
                          ('口をぱくぱくさせる', '＝口を～させる'),
                          ('ぱくぱく食べる', '～食べる'),
                          ('タバコをぱくぱく吹かす', '＝タバコを～吹かす'),
                          ('口をぱくりと開いて', '＝口を～と開いて'),
                          ('ぱくりと食べる', '～と食べる'),
                          ('帆が風にぱたぱたする', '＝帆が風に～する'),
                          ('翼をぱたぱたする', '＝翼を～する'),
                          ('雨のぱたぱたいう音がする', '＝雨の～いう音がする'),
                          ('ほこりをぱたぱた払う', '＝ほこりを～払う'),
                          ('ぱちっと音がする', '～音がする'),
                          ('ぱちっと打つ', '～打つ'),
                          ('指でぱちっとはじく', '＝指で～はじく'),
                          ('小銃のぱちぱち言う音', '＝小銃の～言う音'),
                          ('手をぱちぱち叩く', '＝手を～叩く'),
                          ('目をぱちぱちさせる', '＝目を～させる'),
                          ('ぱっくり嚙みつく', '～嚙みつく'),
                          ('ぱっちりした目', '～した目'),
                          ('目をぱっちりあける', '＝目を～あける'),
                          ('刀をぱっちりさやに納める', '＝刀を～さやに納める'),
                          ('ぱっぱとタバコを吹かす', '～タバコを吹かす'),
                          ('ぱらぱら雨', '～雨'),
                          ('木の葉がぱらぱら散った', '＝木の葉が～散った'),
                          ('日当たりの好い', '～の好い'),
                          ('この部屋は日当たりが悪い', '＝この部屋は～が悪い'),
                          ('左褄を取る', '～を取る'),
                          ('左褄を取った女', '～を取った女'),
                          ('人払いをする', '～をする'),
                          ('人払いを命ずる', '～を命ずる'),
                          ('人払いを願う', '～を願う'),
                          ('火持ちが好い', '～が好い'),
                          ('火持ちが悪い', '～が悪い'),
                          ('ひゅーひゅー鳴る', '～鳴る'),
                          ('ひゅーひゅー鳴る音', '～鳴る音'),
                          ('びしゃりと打つ', '～打つ'),
                          ('びしゃりと本を閉じる', '～本を閉じる'),
                          ('雨がびしょびしょ降る', '＝雨が～降る'),
                          ('びしょびしょに濡れる', '～に濡れる'),
                          ('びっしょり汗をかく', '～汗をかく'),
                          ('シャツが汗でびっしょりだ', '＝シャツが汗で～だ'),
                          ('びりびり裂く', '～裂く'),
                          ('縫目をびりびり解く', '＝縫目を～解く'),
                          ('びりびりと震動する', '～と震動する'),
                          ('電撃をびりびりと感じる', '電撃を～と感じる'),
                          ('鬢太を張る', '～を張る'),
                          ('往復びんたを張る', '＝往復～を張る'),
                          ('ぴくぴく動く', '～動く'),
                          ('顔をぴくぴくさせる', '＝顔を～させる'),
                          ('ぴしっと石を打つ', '～石を打つ'),
                          ('ぴしっとむちで打つ', '～むちで打つ'),
                          ('不倶戴天の仇/アダ/', '～の仇/アダ/'),
                          ('不倶戴天の恨みをいだく', '～の恨みをいだく'),
                          ('不即不離の態度を持する(執る)', '～の態度を持する(執る)'),
                          ('不即不離の間柄になる', '～の間柄になる'),
                          ('ふっつり思い切る', '～思い切る'),
                          ('…とふっつりと手を切る', '…と～と手を切る'),
                          ('ふにゃふにゃ言う', '～言う'),
                          ('柔かくてふにゃふにゃしている', '＝柔かくて～している'),
                          ('…は頭がふらふらする', '…は頭が～する'),
                          ('…は足がふらふらする', '…は足が～する'),
                          ('ふらふら立ち上がる', '～立ち上がる'),
                          ('降り通しの雨', '～の雨'),
                          ('降り通しに降る', '～に降る'),
                          ('鼻をふんふん鳴らす', '＝鼻を～鳴らす'),
                          ('ふんふん嗅ぐ', '～嗅ぐ'),
                          ('香気芬芬たり', '＝香気～たり'),
                          ('臭気芬々鼻をつく', '＝臭気～鼻をつく'),
                          ('ぶうぶう鳴る', '～鳴る'),
                          ('ぶうぶう言う', '～言う'),
                          ('武運つたなく死ぬ', '～つたなく死ぬ'),
                          ('ぶつぶつ云う', '～云う'),
                          ('ぶつぶつ云う人', '～云う人'),
                          ('ぶつぶつ煮える', '～煮える'),
                          ('ぶんぶん言う', '～言う'),
                          ('兵馬倥傯の間に', '～の間に'),
                          ('兵馬倥傯の間に馳駆する', '～の間に馳駆する'),
                          ('翩々たる小才子', '～たる小才子'),
                          ('翩々として風に翻る', '～として風に翻る'),
                          ('べたりと坐る', '～と坐る'),
                          ('べたりとつく', '～とつく'),
                          ('お腹がぺこぺこだ', '＝お腹が～だ'),
                          ('ぺこんとお辞儀をする', '～お辞儀をする'),
                          ('ぺこんと凹む', '～と凹む'),
                          ('ぺたんと坐る', '～坐る'),
                          ('切手をぺたんと張る', '＝切手を～張る'),
                          ('ぺろりとなめる', '～となめる'),
                          ('ぺろりと舌を出す', '～と舌を出す'),
                          ('ぺろりと食べる(平らげる)', '～と食べる, ～と平らげる'),
                          ('奉公振りがよい', '～がよい'),
                          ('奉公振りが悪い', '～が悪い'),
                          ('したい放題に', '＝したい～に'),
                          ('食い放題に食う', '＝食い～に食う'),
                          ('言いたい放題のことを言う', '＝言いたい～のことを言う'),
                          ('這々の体/テイ/で', '～の体/テイ/で'),
                          # contents = contents.replace('這々の体で引退る', '～の体で引退る'),
                          ('頰を火照らす', '＝頰を～'),
                          ('顔をほてらして', '＝顔を｜して～'),
                          ('まっかに顔をほてらせて', '＝まっかに顔を｜せて～'),
                          ('頰が火照る', '＝頰が～'),
                          ('頰がほてる', '＝頰が～'),
                          ('体中がほてっていた', '＝体中が｜って～いた'),
                          ('ほやりと笑う', '～笑う'),
                          ('洞が峠に陣取る', '～に陣取る'),
                          ('洞が峠を下がる', '～を下がる'),
                          ('暴虎慿河の勇', '～の勇'),
                          ('暴虎慿河の徒', '～の徒'),
                          ('ぼこぼこ音がする', '～音がする'),
                          ('ぼこぼこ湧き出る', '～湧き出る'),
                          ('ぼりぼり噛る', '～噛る'),
                          ('ぼりぼり掻く', '～掻く'),
                          ('ぼろくそに言う', '～に言う'),
                          ('本を傍にぽいと投げる', '＝本を傍に～投げる'),
                          ('ぽいと跳ぶ', '～跳ぶ'),
                          ('ぽかりとなぐる', '～となぐる'),
                          ('ぽかりと明く', '～と明く'),
                          ('指の節をぽきぽき鳴らす', '＝指の節を～鳴らす'),
                          ('ぽたり落ちる', '～落ちる'),
                          ('ぽたり滴る', '～滴る'),
                          ('汽車がぽっぽと煙を吐きながら停車場を出た',
                           '＝汽車が～煙を吐きながら停車場を出た'),
                          ('ぽっぽと湯気の出ているホット', '～湯気の出ているホット'),
                          ('ぽつりと一粒雨があたった', '～と一粒雨があたった'),
                          ('ぽつりと星が一つ残っている', '～と星が一つ残っている'),
                          ('ぽりぽり食べる', '～食べる'),
                          ('ぽりぽり爪で…をひっかく', '～爪で…をひっかく'),
                          ('涙をぽろぽろこぼす', '＝涙を～こぼす'),
                          ('ぽろぽろ砕ける', '～砕ける'),
                          ('ぽんとコルクを抜く', '～とコルクを抜く'),
                          ('ぽんと肩を叩く', '～と肩を叩く'),
                          ('ぽんと投げてやる', '～と投げてやる'),
                          ('ぽんと断る', '～と断る'),
                          ('ぽんぽん花火が上がった', '～花火が上がった'),
                          ('ぽんぽん手を鳴らす', '～手を鳴らす'),
                          ('ぽんぽん怒る', '～怒る'),
                          ('ぽーぽー鳴く', '～鳴く'),
                          ('ぽーぽー鳴る', '～鳴る'),
                          ('魔が差した', '～が差した'),
                          ('魔の海', '～の海'),
                          ('魔の手', '～の手'),
                          ('魔をよける', '～をよける'),
                          ('目が眩う', '＝目が～'),
                          ('待ちぼけを食う', '～を食う'),
                          ('丸々と太った', '～太った'),
                          ('丸々と肥えている', '～肥えている'),
                          ('万分の一', '～の一'),
                          ('それは耳新らしい事ではない', '＝それは～事ではない'),
                          ('何か耳新らしい事があるか', '＝何か～事があるか'),
                          ('弥陀/ミダ/の名号', '＝弥陀/ミダ/の～'),
                          ('六字の名号', '＝六字の～'),
                          ('むっくり起き上がる', '～起き上がる'),
                          ('むっくり肥えた', '～肥えた'),
                          ('むにゃむにゃ言う', '～言う'),
                          ('…しようという気がむらむら[と]起る',
                           '…しようという気が～[と]起る'),
                          ('煙がむらむら立ち上がる', '＝煙が～立ち上がる'),
                          ('病人の熱気でむんむんする', '＝病人の熱気で～する'),
                          (
                              '[気が]めいる быть в подавленном настроении, прийти в уныние, пасть духом;\n気がめいって с тяжёлым сердцем;',
                              '[気が]めいる быть в подавленном настроении, прийти в уныние, пасть духом;\n＝気が｜って～ с тяжёлым сердцем;'),
                          ('[気が]めいる', '＝[気が]～'),
                          ('目八分に捧げる', '～に捧げる'),
                          ('目八分に…を見る', '～に…を見る'),
                          ('めりめりいう', '～いう'),
                          ('めりめり音がして倒れる', '～音がして倒れる'),
                          ('もぐもぐ[と]云う, 口をもぐもぐさせる',
                           '～[と]云う, ＝口を～させる'),
                          ('もぐもぐ嚙む', '～嚙む'),
                          ('もぐもぐ[と]動く', '～[と]動く'),
                          ('勿体をつける', '～をつける'),
                          ('物心が付く', '～が付く'),
                          ('物心付いて以来', '～付いて以来'),
                          ('物心のつかない内に', '～のつかない内に'),
                          ('門前払いを食わす', '～を食わす'),
                          ('門前払いを食う', '～を食う'),
                          ('動ともすれば, 動もすると', '～すれば, ～すると'),
                          ('動ともすれば…する', '～すれば…する'),
                          ('行きずりの人', '～の人'),
                          ('行きずりの縁/エン/', '～の縁/エン/'),
                          ('そんなことはないよ', '＝そんなことはない～'),
                          ('頼むよ', '＝頼む～'),
                          ('横槍を入れる', '～を入れる'),
                          ('横槍が出た', '～が出た'),
                          ('夜目の利く', '～の利く'),
                          ('夜目ではっきりとしない', '～ではっきりとしない'),
                          ('◇夜目遠目/トウメ/傘の中/ウチ/',
                           '夜目遠目傘の中'),
                          ('よろしきを得る', '～を得る'),
                          ('寛大よろしきを得る', '＝寛大～を得る'),
                          ('経営よろしきを得る', '＝経営～を得る'),
                          ('李下に冠/カンムリ/を正さず, 李下の冠',
                           '李下に冠を正さず, 李下の冠'),
                          ('俚耳に入り易/イリヤス/い', '～に入り易/イリヤス/い'),
                          ('俚耳に入りにくい', '～に入りにくい'),
                          ('槍を(太刀を)りゅうりゅうとしごく',
                           '＝槍(太刀)を～しごく'),
                          ('両手ききの人', '～の人'),
                          ('両天秤をかける', '～をかける'),
                          ('釐亳も', '～も'),
                          ('釐毫も違わない', '～も違わない'),
                          ('凜と張った目', '～張った目'),
                          ('凜とした態度', '～した態度'),
                          ('列氏寒暖計', '～寒暖計'),
                          ('連理の松/マツ/', '～の松/マツ/'),
                          ('連理の契/チギ/り', '～の契/チギ/り'),
                          ('六分の一', '～の一'),
                          ('カ氏寒暖計', '～寒暖計'),
                          ('セ氏寒暖計', '～寒暖計'),
                          ('<i>неправ., см.</i>', '<i>неправ.</i> <i>см.</i>'),
                          ('<i>сокр., см.</i>', '<i>сокр.</i> <i>см.</i>'),
                          ('\n<i>употребляется в двух формах:</i>', ''),
                          ('<i>уст. особая бумага, посылаемая в знак благодарности за небольшой подарок.</i>',
                           '<i>уст.</i> особая бумага, посылаемая в знак благодарности за небольшой подарок.'),
                          ('<a href="#005-67-03">おそるべき</a> <i>и др.</i>',
                           '<a href="#005-67-03">おそるべき</a>.'),
                          (
                              'название яп. способа исчисления возраста (до 1949 г.), состоявшего в том, что год '
                              'рождения считался за полный год; напр., считалось, что, в каком бы месяце 1903 г. ни '
                              'родился ребёнок, с 1 января 1904 г. ему пошёл второй год',
                              'название яп. способа исчисления возраста (до 1949 г.)'),
                          ('<i>приписка после фамилии при адресовании писем.</i>',
                           'приписка после фамилии при адресовании писем.'),
                          ('[気が]腐る', '＝[気が]～'),
                          ('そんなに腐るな', '＝そんなに～な'),
                          ('化生の物', '～の物'),
                          ('<i>буд. что-л., ставшее живым существом; что-л., принявшее облик живого существа</i>',
                           '<i>буд.</i> что-л., ставшее живым существом; что-л., принявшее облик живого существа'),
                          ('<a href="#000-65-10">これほど</a> <i>и др.</i>',
                           '<a href="#000-65-10">これほど</a>'),
                          ('<i>ист. один из титулов придворных дам.</i>',
                           '<i>ист.</i> один из титулов придворных дам.'),
                          ('事件を劇に仕組む', '＝事件を劇に～'),
                          ('下に出す', '～に出す'),
                          ('下に取る', '～に取る'),
                          ('後を慕う', '＝後を～'),
                          ('失礼ながら…, 失礼ですが…', '～ながら, ～ですが'),
                          ('では失礼', '＝では～'),
                          ('じっと見詰める', '～見詰める'),
                          ('じっと我慢する(堪える)', '～我慢する, ～堪える'),
                          ('\n<i>в наст. вр. обычно не переводится</i>;', ''),
                          (';\n4) <i>после вопр. мест. см. самые местоимения</i>', ''),
                          ('木という木', '＝木～木'),
                          ('所だった', '～だった'),
                          ('所です(だ)', '～です(だ)'),
                          ('どっこいしょと腰を下ろす', '～と腰を下ろす'),
                          ('物価がどんどん上がっている', '＝物価が～上がっている'),
                          ('どんどん売れる', '～売れる'),
                          ('湯をどんどん沸かす', '＝湯を～沸かす'),
                          ('どんどん質問する', '～質問する'),
                          ('美しいなあ', '＝美しい～'),
                          ('なあ君、そうだろう', '～君、そうだろう'),
                          ('猶…が如し(如くである)', '～…が如し(如くである)'),
                          ('…ようになる', '…ように～'),
                          ('…なくなる', '…なく～'),
                          ('抜き差しができない', '～ができない'),
                          ('抜き差しならぬ真実', '～ならぬ真実'),
                          ('今月に入ってから', '＝今月に｜って～から'),
                          ('梅雨に入る', '＝梅雨に～'),
                          ('店を張る', '＝店を～'),
                          ('世帯を張る', '＝世帯を～'),
                          ('ぱっぱとタバコを吹かす', '～タバコを吹かす'),
                          ('辞書を引く', '＝辞書を～'),
                          ('祖先を祀る', '＝祖先を～'),
                          ('祖先の霊を祀る', '＝祖先の霊を～'),
                          ('もはや五時を回った', '＝もはや五時を｜った～'),
                          ('…の向こうを張る', '…の～を張る'),
                          ('目に合う', '～に合う'),
                          ('持ちが良い', '～が良い'),
                          ('持ちをよくする', '～をよくする'),
                          ('そこが問題だ', '～が問題だ'),
                          ('～になる постричься [в монахи]; <i>обр.</i> (<i>тж.</i> 坊主に刈る) коротко '
                           'остричься, обрить голову;',
                           '～になる постричься [в монахи];\n～になる, ～に刈る <i>обр.</i> коротко остричься, '
                           'обрить голову;'),
                          ('2): ～の <i>поэтическое определение к таким словам, как</i> <a '
                           'href="#000-74-10">とし</a> <i>(год)</i>, <a href="#000-60-20">つき</a> <i>('
                           'месяц) и т. п.</i>;',
                           '2): ～の <i>поэтическое определение к таким словам, как год, месяц и т. п.</i>;'),
                          ('～のある, 勢いの好い сильный; энергичный; смелый; влиятельный;',
                           '～のある, ～の好い сильный; энергичный; смелый; влиятельный;'),
                          ('～をする(に接する) приглашать (принимать) гостей, устраивать приём;',
                           '～をする, ～に接する приглашать (принимать) гостей, устраивать приём;'),
                          ('2) (<i>тк.</i> 梶, <i>тж.</i> 楫) рулевое весло;',
                           '2) (<i>тк.</i> 梶) (<i>тж.</i> 楫) рулевое весло;'),
                          ('(<i>санскр.</i> namo, <i>сокр.</i> 南無阿弥陀仏/ナムアミダブツ/)',
                           '(<i>санскр.</i> namo) (<i>сокр.</i> 南無阿弥陀仏/ナムアミダブツ/)'),
                          ('耳が肥えている', '＝耳が｜て～いる'),
                          (
                              '<i>разг. замужняя женщина, которая по уговору с мужем или по его принуждению заводит любовника, после чего муж угрозами вымогает у того деньги.</i>',
                              '<i>разг.</i> замужняя женщина, которая по уговору с мужем или по его принуждению заводит любовника, после чего муж угрозами вымогает у того деньги.'),
                          ('\n～に <i>см. ниже</i> 2;', ''),
                          (
                              '2. действительно, на [самом] деле, фактически, реально; практически, на практике; <i>(в конце предл.)</i> уверяю вас!',
                              '2. ～[に] действительно, на [самом] деле, фактически, реально; практически, на практике; <i>(в конце предл.)</i> уверяю вас!'),
                          (
                              ' (<i>а) время от 2 до 4 часов дня и ночи; б) 3 часа дня и ночи; ср.</i> <a href="#005-63-22">とき【時】2</a> <i>и</i> <a href="#008-10-52">ここのつどき</a>).',
                              ': время от 2 до 4 часов дня и ночи; 3 часа дня и ночи.'),
                          (
                          '<i>(1) на билет при покупке у барышника; 2) на акцию сверх номинала, при повышенной котировке)</i>.',
                          '<i>(на билет при покупке у барышника; на акцию сверх номинала, при повышенной котировке)</i>.'),
                          ('1) <i>уст., вошёл в р-н Дайто;</i> 2) <i>парк в Токио.</i>',
                           '\n1) <i>уст., вошёл в р-н Дайто;</i>\n2) <i>парк в Токио.</i>'),
                          (
                          '<i>(1) высшая степень духовного совершенства; 2) подвижник, достигший этой степени совершенства)</i>.',
                          '\n1) <i>высшая степень духовного совершенства;</i>\n2) <i>подвижник, достигший этой '
                          'степени совершенства</i>.'),
                          ('<i>(1) о соперничестве между государствами в области науки; 2) о войне с применением '
                           'средств, созданных на основе новейших достижений науки и техники)</i>.',
                           '\n1) <i>о соперничестве между государствами в области науки;</i>\n2) <i>о войне с '
                           'применением средств, созданных на основе новейших достижений науки и техники</i>.'),
                          ('<i>(1) сорт яп. зелёного чая; 2) название одного из танцев театра Кабуки)</i>.',
                           '\n1) <i>сорт яп. зелёного чая;</i>\n2) <i>название одного из танцев театра Кабуки</i>.'),
                          ('<i>(1) приставка к посмертным буддийским именам верующих мирян; 2) обозначение тех, '
                           'кто выполняет обеты без пострижения в монахи)</i>.',
                           '\n1) <i>приставка к посмертным буддийским именам верующих мирян;</i>\n2) <i>обозначение '
                           'тех, кто выполняет обеты без пострижения в монахи</i>.'),
                          (
                          '<i>(1) вид песенного сказа в XV — XVI вв.; 2) вид представлений театра марионеток в XVII — XVIII вв.)</i>.',
                          '\n1) <i>вид песенного сказа в XV — XVI вв.;</i>\n2) <i>вид представлений театра марионеток в XVII — XVIII вв.</i>.'),
                          ('<i>(совр.</i> Чунгбо́, <i>центральная часть Вьетнама).</i>',
                           '(<i>совр.</i> Чунгбо́).'),
                          ('(<i>т. н.</i> <a href="#003-04-08">じょうげん【上元】</a>, <a href="#007-27-72">ちゅうげん【中元】</a> '
                           '<i>и</i> <a href="#000-47-62">かげん【下元】</a>)',
                           ''),
                          (' <i>ср.</i> <a href="#000-11-63">しょうブル</a>, <a href="#000-49-03">しょうきぼ</a> <i>и др.</i>;',
                           ''),
                          ('; ср.</i> <a href="#008-10-52">ここのつどき</a>, <a href="#008-35-40">やつどき</a>, '
                           '<a href="#007-11-93">ななつどき</a> <i>и т. д.</i>',
                           ''),
                          (': а) 春の七草<i> (весенние)</i>: <i>см.</i> <a href="#006-10-38">なずな</a>, '
                           '<a href="#002-81-23">すずな</a>, <a href="#005-70-70">すずしろそう</a>, '
                           '<a href="#009-22-62">せり【芹】</a>, <a href="#000-08-32">はこべ</a>, '
                           '<a href="#006-03-68">ははこぐさ</a>, <a href="#002-63-33">ほとけのざ</a>; б) 秋の七草 <i>(осенние)</i>: '
                           '<i>см.</i> <a href="#002-25-56">あさがお</a> <i>или</i> <a href="#006-48-72">ききょう【桔梗】</a>, '
                           '<a href="#000-13-03">くず【葛】</a>, <a href="#001-59-57">なでしこ</a>, '
                           '<a href="#007-37-62">おばな【尾花】</a>, <a href="#003-47-59">おみなえし</a>, '
                           '<a href="#003-40-34">ふじばかま</a>, <a href="#008-09-53">はぎ【萩】</a>',
                           ''),
                          ('(<i>а) время от 4 до 6 часов дня и ночи; б) 5 часов дня и ночи; ср.</i> <a '
                           'href="#005-63-22">とき【時】2</a> <i>и</i> <a href="#008-10-52">ここのつどき</a>)',
                           '(<i>время от 4 до 6 часов дня и ночи; 5 часов дня и ночи</i>)'),
                          ('1) <i>побуд. залог гл.</i> <a href="#007-86-07">はたらく</a> <i>в 1-м знач.</i>;',
                           '1) <i>побуд. залог гл.</i> <a href="#007-86-07">はたらく 1</a>;'),
                          ('<a href="#004-38-66">はんと【版図】</a> <i>и</i> <a href="#006-92-82">こせき【戸籍】</a>',
                           '版図 <i>и</i> 戸籍'),
                          ('; ср.</i> <a href="#001-18-01">ひばく【被爆】</a>, <a href="#002-00-07">ひがいしゃ</a> <i>и др',
                           ''),
                          ('(<i>из-за созвучия слов</i> <a href="#006-38-91">こおりがし【氷菓子】</a> <i>и</i> <a '
                           'href="#007-72-05">こうりがし【高利貸し】</a>)',
                           ''),
                          ('2) <i>побуд. форма гл.</i> <a href="#008-81-72">おこす【起こす】</a> <i>в других знач.</i>',
                           '2) <i>побуд. форма гл.</i> <a href="#008-81-72">おこす【起こす】</a>'),
                          (
                          '\n7) <i>форма потенциального залога гл.</i> <a href="#009-18-92"> とる【取る】</a> <i>во всех его знач., напр.:</i>',
                          ''),
                          (
                          '\n(<i>форма побуд. залога от гл.</i> <a href="#002-28-12">おもう</a> <i>в знач.</i> 1 <i>и</i> 2)',
                          ''),
                          ('2) <i>см.</i> <a href="#006-78-84">ちゅうきょう</a> <i>(географическое название)</i>',
                           '2) <i>(географическое название)</i> <i>см.</i> <a href="#006-78-84">ちゅうきょう</a>'),
                          ('1): 彼は人に彼一人でそれをやったのだと思わせた он создал впечатление, будто он это сделал один;',
                           '1) ＝変だと～ производить странное впечатление; наводить на мысль, что <i>что-л.</i> странно;'),
                          (', напр.:</i>', '</i>'),
                          ('～…', '＋'),
                          ('ё', 'е')]

            for src, trg in tqdm(to_replace, desc="[Warodai] Preprocessing raw source".ljust(34),
                                 disable=not show_progress):
                contents = contents.replace(src, trg)

            s_contents = contents.split('\n\n')[1:]

            res = []
            temp_res = []

            for c in tqdm(s_contents, desc="[Warodai] Building word database".ljust(34), disable=not show_progress):
                entries, temp_entries = self._convert_to_entry(c)
                if entries:
                    res.extend(entries)
                if temp_entries:
                    temp_res.extend(temp_entries)

            return res, temp_res

    def _get_entry_by_eid(self, eid) -> WarodaiEntry:
        return [e for e in self._entries if e.eid == eid][0]

    def _get_entry_index_by_eid(self, eid) -> int:
        return self._entries.index(self._get_entry_by_eid(eid))

    def load(self, fname: str = "default") -> WarodaiDictionary:
        if fname == 'default':
            fname = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.joinpath(
                'dictionaries/warodai.jtdb')
        return pickle.load(open(fname, "rb"))
