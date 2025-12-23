import requests, json

payload={
  "nodes":[
    {"id":"n1","type":"LoadData","data":{"label":"LoadData","params":{"samples":[{"id":"s1","question":"What is AI?","context":""}]}}},
    {"id":"n2","type":"Annotate","data":{"label":"Annotate","params":{"candidate_llms":["gpt2"],"llm_mode":"local","task_class":"tasks.qa.QATask"}}},
    {"id":"n3","type":"Output","data":{"label":"Output","params":{}}}
  ],
  "edges":[{"source":"n1","target":"n2"},{"source":"n2","target":"n3"}]
}

try:
    r = requests.post('http://127.0.0.1:5000/run_graph', json=payload, timeout=600)
    print('STATUS', r.status_code)
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(r.text)
except Exception as e:
    print('REQUEST FAILED', e)
