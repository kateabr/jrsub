from src.warodai import WarodaiDictionary, WarodaiLoader
from src.yarxi import YarxiDictionary, YarxiLoader
import pickle

if __name__ == '__main__':
    # yd = YarxiLoader().load()         # full rescan
    # yd = YarxiLoader().from_bin()     # load a snapshot
    # while True:
    #     lookup_res = yd.lookup(input().strip(), input().strip())
    #     for res in lookup_res:
    #         print(' / '.join(res.lexeme), '|', '〖' + ' / '.join(res.reading) + '〗')
    #         for tr in res.translation:
    #             print('  -', tr)
    #         print('')

    # wd = WarodaiLoader().load()       # full rescan
    wd = WarodaiLoader().from_bin()     # load a snapshot
    while True:
        lookup_res = wd.lookup(input().strip(), input().strip())
        for res in lookup_res:
            print(' / '.join(res.lexeme), '|', '〖' + ' / '.join(res.reading) + '〗')
            for tr in res.translation:
                print('  -', tr)
            print('')
