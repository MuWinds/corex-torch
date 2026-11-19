import collections
import json

d = json.load(open('/tmp/tp24.json'))
print('TYPE_COUNTS', collections.Counter(e.get('type') for e in d))
print('NAMES', [e.get('name') for e in d if e.get('name') in ('fmt', 'cutlass', 'pocketfft', 'nlohmann', 'flatbuffers')])
for e in d:
    if e.get('name') == 'fmt':
        print('FMT_ENTRY', e)
