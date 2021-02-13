from jrsub import WarodaiLoader
from jrsub import YarxiLoader

if __name__ == '__main__':
    print('Select a dictionary: [y]arxi or [w]arodai')
    dict_switcher = input().strip()
    if dict_switcher == 'y':
        print("- Yarxi -")
        # yd = YarxiLoader().rescan()         # reload database from the raw source
        yd = YarxiLoader().load()           # load a pre-generated snapshot
        while True:
            lookup_res = yd.lookup(input().strip(), input().strip())
            for res in lookup_res:
                print(' / '.join(res.lexeme), '|', '〖' + ' / '.join(res.reading) + '〗')
                for tr in res.translation:
                    print('  -', tr)
                print('')
    elif dict_switcher == 'w':
        print("- Warodai -")
        # wd = WarodaiLoader().rescan()       # reload database from the raw source
        wd = WarodaiLoader().load()         # load a pre-generated snapshot
        while True:
            lookup_res = wd.lookup(input().strip(), input().strip())
            for res in lookup_res:
                print(' / '.join(res.lexeme), '|', '〖' + ' / '.join(res.reading) + '〗')
                for tr in res.translation:
                    print('  -', tr)
                print('')