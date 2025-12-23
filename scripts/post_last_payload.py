import json, requests
p=json.load(open('last_run_graph_payload.json','r',encoding='utf-8'))
print('Posting payload nodes:', [n.get('data',{}).get('label') for n in p.get('nodes',[])])
r=requests.post('http://127.0.0.1:5000/run_graph', json=p, timeout=600)
print('STATUS', r.status_code)
try:
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
except Exception:
    print(r.text)
