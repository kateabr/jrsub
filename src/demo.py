from src.warodai import WarodaiDictionary, WarodaiLoader
from src.yarxi import YarxiDictionary, YarxiLoader
import pickle

if __name__ == '__main__':
    # yd = YarxiLoader().rescan()         # reload database from the raw source
    # yd = YarxiLoader().load()           # load a pre-generated snapshot
    # while True:
    #     lookup_res = yd.lookup(input().strip(), input().strip())
    #     for res in lookup_res:
    #         print(' / '.join(res.lexeme), '|', '〖' + ' / '.join(res.reading) + '〗')
    #         for tr in res.translation:
    #             print('  -', tr)
    #         print('')

    # wd = WarodaiLoader().rescan()       # reload database from the raw source
    wd = WarodaiLoader().load()         # load a pre-generated snapshot
    while True:
        lookup_res = wd.lookup(input().strip(), input().strip())
        for res in lookup_res:
            print(' / '.join(res.lexeme), '|', '〖' + ' / '.join(res.reading) + '〗')
            for tr in res.translation:
                print('  -', tr)
            print('')
