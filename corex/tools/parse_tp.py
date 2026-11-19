import json
import sys

d = json.load(open('/tmp/tp24.json'))
subs = {e['name']: e['sha'] for e in d if e.get('type') == 'submodule'}
if len(sys.argv) > 1:
    for n in sys.argv[1:]:
        print(subs.get(n, ''))
else:
    for k, v in subs.items():
        print(k, v)
