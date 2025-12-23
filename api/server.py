from flask import Flask, request, jsonify
try:
    # prefer using flask_cors if available
    from flask_cors import CORS
except Exception:
    CORS = None
import importlib
import traceback
from typing import Dict, Any, List
import os
import ast

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
if CORS is not None:
    # allow all origins for local development
    CORS(app, resources={r"/*": {"origins": "*"}})
else:
    # fallback: add header in after_request
    @app.after_request
    def _add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response

# Helper to find upstream outputs

def topological_sort(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    id_set = {n['id'] for n in nodes}
    adj = {n['id']: [] for n in nodes}
    indeg = {n['id']: 0 for n in nodes}
    for e in edges:
        s = e.get('source')
        t = e.get('target')
        if s in id_set and t in id_set:
            adj[s].append(t)
            indeg[t] += 1
    q = [nid for nid, d in indeg.items() if d == 0]
    order = []
    while q:
        n = q.pop(0)
        order.append(n)
        for nei in adj.get(n, []):
            indeg[nei] -= 1
            if indeg[nei] == 0:
                q.append(nei)
    if len(order) != len(nodes):
        # cycle or disconnected nodes — fallback to nodes order
        return [n['id'] for n in nodes]
    return order


# Review queue persistence helpers
REVIEW_QUEUE_PATH = os.path.join(ROOT_DIR, 'human_review_queue.json')

def read_review_queue():
    try:
        import json
        if os.path.exists(REVIEW_QUEUE_PATH):
            with open(REVIEW_QUEUE_PATH, 'r', encoding='utf-8') as fh:
                return json.load(fh) or []
    except Exception:
        return []
    return []

def append_review_queue(items: List[Dict[str, Any]]):
    if not items:
        return
    try:
        import json
        existing = read_review_queue() or []
        # tag each item with received timestamp
        import time
        ts = int(time.time())
        for it in items:
            # shallow copy
            entry = dict(it)
            entry['_received_ts'] = ts
            existing.append(entry)
        with open(REVIEW_QUEUE_PATH, 'w', encoding='utf-8') as fh:
            json.dump(existing, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def instantiate_class_from_module(fullname: str, *args, **kwargs):
    # fullname: 'filters.ActiveLearningFilter' or 'routers.KNNRouter'
    module_name, class_name = fullname.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(*args, **kwargs)


@app.route('/run_graph', methods=['POST'])
def run_graph():
    payload = request.get_json(force=True)
    # persist incoming payload for debugging / audit
    try:
        import json, time
        dump_path = os.path.join(ROOT_DIR, f"run_graph_payload_{int(time.time())}.json")
        with open(dump_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        # also write latest copy to a stable file
        latest_path = os.path.join(ROOT_DIR, 'last_run_graph_payload.json')
        with open(latest_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        app.logger.info('Saved /run_graph payload to %s', dump_path)
    except Exception:
        app.logger.exception('Failed to write run_graph payload for debugging')
    nodes = payload.get('nodes', [])
    edges = payload.get('edges', [])

    node_map = {n['id']: n for n in nodes}
    order = topological_sort(nodes, edges)

    context = {}  # node_id -> output
    server_review_items = []

    try:
        for nid in order:
            node = node_map[nid]
            ntype = node.get('data', {}).get('label') or node.get('type') or node.get('data', {}).get('type')
            params = node.get('data', {}).get('params', {}) or {}

            # gather inputs from predecessors
            inputs = []
            for e in edges:
                if e.get('target') == nid:
                    src = e.get('source')
                    if src in context:
                        inputs.append(context[src])
            # flatten inputs if lists
            flat_input = []
            for item in inputs:
                if isinstance(item, list):
                    flat_input.extend(item)
                else:
                    flat_input.append(item)

            # Node dispatch
            if ntype == 'LoadData':
                # params: either 'samples' list or 'dataset' name
                if 'samples' in params:
                    data = params['samples']
                else:
                    ds = params.get('dataset', 'squad')
                    max_samples = int(params.get('max_samples', 1000))
                    if ds == 'squad':
                        try:
                            from datasets import SquadDataset
                            data = SquadDataset.from_url(save_path='squad_tmp.json', max_samples=max_samples)
                        except Exception:
                            data = []
                    else:
                        # try load from file path
                        import json
                        try:
                            with open(ds, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                        except Exception:
                            data = []
                context[nid] = data

            elif ntype == 'Filter':
                # params: filter_class (e.g., 'filters.ActiveLearningFilter') and filter_params
                clsname = params.get('filter_class', 'filters.ActiveLearningFilter')
                fparams = params.get('filter_params', {})
                try:
                    filt = instantiate_class_from_module(clsname, **fparams)
                    # decide input data
                    input_data = flat_input[0] if flat_input else []
                    out = filt.filter(input_data)
                except Exception as e:
                    out = {'error': str(e), 'trace': traceback.format_exc()}
                context[nid] = out

            elif ntype == 'Router':
                clsname = params.get('router_class', 'routers.KNNRouter')
                rparams = params.get('router_params', {}) or {}
                candidate_llms = params.get('candidate_llms', [])
                try:
                    # import the class so we can inspect its __init__ signature
                    module_name, class_name = clsname.rsplit('.', 1)
                    mod = importlib.import_module(module_name)
                    cls = getattr(mod, class_name)
                    # Build an Annotator instance so routers that require it receive one
                    try:
                        from misc.llm_provider import LocalLLM, APILLM
                        from annotation import Annotator
                        # router-level llm params (fallbacks)
                        r_llm_mode = params.get('llm_mode', 'local')
                        r_api_config = params.get('api_config', {}) or {}
                        r_task_class = params.get('task_class', 'tasks.QATask')

                        # build llm_dict
                        llm_dict = {}
                        if r_llm_mode == 'local':
                            for name in candidate_llms:
                                llm_dict[name] = LocalLLM(name)
                        else:
                            for name in candidate_llms:
                                conf = r_api_config.get(name, {})
                                llm_dict[name] = APILLM(conf.get('api_url', ''), conf.get('api_key'), conf.get('extra_headers'))

                        # instantiate task class
                        task_mod, task_name = r_task_class.rsplit('.', 1)
                        task_cls = getattr(importlib.import_module(task_mod), task_name)
                        task = task_cls()

                        annotator = Annotator(candidate_llms, llm_dict, task=task)
                    except Exception:
                        annotator = None

                    # prepare kwargs for constructor
                    init_kwargs = dict(rparams)
                    init_kwargs['annotator'] = annotator
                    init_kwargs['candidate_llms'] = candidate_llms

                    router = cls(**init_kwargs)

                    input_data = flat_input[0] if flat_input else []
                    out = router.route(input_data)
                except Exception as e:
                    out = {'error': str(e), 'trace': traceback.format_exc()}
                context[nid] = out

            elif ntype == 'Annotate' or ntype == 'Annotation':
                # params: candidate_llms, llm_mode, api_config, task_class
                candidate_llms = params.get('candidate_llms', [])
                llm_mode = params.get('llm_mode', 'local')
                api_config = params.get('api_config', {})
                task_class = params.get('task_class', 'tasks.QATask')
                try:
                    from misc.llm_provider import LocalLLM, APILLM
                    # build llm_dict
                    llm_dict = {}
                    if llm_mode == 'local':
                        for name in candidate_llms:
                            llm_dict[name] = LocalLLM(name)
                    else:
                        for name in candidate_llms:
                            conf = api_config.get(name, {})
                            llm_dict[name] = APILLM(conf.get('api_url', ''), conf.get('api_key'), conf.get('extra_headers'))

                    # instantiate task
                    task_mod, task_name = task_class.rsplit('.', 1)
                    task_cls = getattr(importlib.import_module(task_mod), task_name)
                    task = task_cls()

                    from annotation import Annotator
                    annotator = Annotator(candidate_llms, llm_dict, task=task)
                    # pass the full list of samples to annotate_batch
                    input_data = flat_input if flat_input else []
                    out = annotator.annotate_batch(input_data)
                    # collect any low-confidence items queued for human review
                    try:
                        q = getattr(annotator, 'human_review_queue', None)
                        if q and hasattr(q, 'queue') and isinstance(q.queue, list):
                            server_review_items.extend(q.queue)
                    except Exception:
                        pass
                except Exception as e:
                    out = {'error': str(e), 'trace': traceback.format_exc()}
                context[nid] = out

            elif ntype == 'Output':
                # params: path optional
                input_data = flat_input[0] if flat_input else []
                path = params.get('path')
                if path:
                    try:
                        import json
                        # resolve relative paths under ROOT_DIR
                        full_path = path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)
                        parent = os.path.dirname(full_path)
                        if parent and not os.path.exists(parent):
                            os.makedirs(parent, exist_ok=True)
                        # ensure data is JSON-serializable
                        def _to_jsonable(o):
                            try:
                                json.dumps(o)
                                return o
                            except TypeError:
                                try:
                                    if hasattr(o, 'to_list'):
                                        return _to_jsonable(o.to_list())
                                except Exception:
                                    pass
                                try:
                                    if hasattr(o, 'to_dict'):
                                        return _to_jsonable(o.to_dict())
                                except Exception:
                                    pass
                                try:
                                    if hasattr(o, '__dict__'):
                                        return _to_jsonable({k: v for k, v in o.__dict__.items()})
                                except Exception:
                                    pass
                                return str(o)

                        jsonable = _to_jsonable(input_data)
                        with open(full_path, 'w', encoding='utf-8') as f:
                            json.dump(jsonable, f, ensure_ascii=False, indent=2)
                        out = {'saved_to': full_path}
                    except Exception as e:
                        out = {'error': str(e), 'trace': traceback.format_exc()}
                else:
                    out = input_data
                context[nid] = out

            elif ntype == 'Task':
                # set task configuration node — just pass through params
                context[nid] = params

            elif ntype == 'CandidateLLMs':
                context[nid] = params.get('candidate_llms', [])

            else:
                # Unknown node type — pass through
                context[nid] = params

        # find nodes with type Output to return their results, else return last node result
        outputs = {}
        for n in nodes:
            ntype = n.get('data', {}).get('label') or n.get('type')
            if ntype == 'Output':
                outputs[n['id']] = context.get(n['id'])
        # persist server_review_items to persistent review queue store
        try:
            append_review_queue(server_review_items)
        except Exception:
            pass

        # Ensure context/outputs are JSON serializable (datasets and custom objects may not be)
        def make_jsonable(obj):
            import json
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                # dict-like
                if isinstance(obj, dict):
                    return {k: make_jsonable(v) for k, v in obj.items()}
                # iterable (but not string)
                if isinstance(obj, (list, tuple, set)):
                    return [make_jsonable(x) for x in obj]
                # try common conversion methods
                try:
                    if hasattr(obj, 'to_dict'):
                        return make_jsonable(obj.to_dict())
                except Exception:
                    pass
                try:
                    if hasattr(obj, 'to_list'):
                        return make_jsonable(obj.to_list())
                except Exception:
                    pass
                # fallback to __dict__ if available
                try:
                    if hasattr(obj, '__dict__'):
                        return make_jsonable({k: v for k, v in obj.__dict__.items()})
                except Exception:
                    pass
                # last resort: string representation
                try:
                    return str(obj)
                except Exception:
                    return None

        serializable_context = {k: make_jsonable(v) for k, v in context.items()}
        serializable_outputs = {k: make_jsonable(v) for k, v in outputs.items()}

        if not serializable_outputs:
            return jsonify({'status': 'ok', 'context': serializable_context})
        return jsonify({'status': 'ok', 'outputs': serializable_outputs, 'context': serializable_context})
    except Exception as e:
        # save error details to disk for debugging
        try:
            import json, time
            err = {'ts': int(time.time()), 'error': str(e), 'trace': traceback.format_exc(), 'payload': payload}
            err_path = os.path.join(ROOT_DIR, f"run_graph_error_{int(time.time())}.json")
            with open(err_path, 'w', encoding='utf-8') as fh:
                json.dump(err, fh, ensure_ascii=False, indent=2)
            latest_err = os.path.join(ROOT_DIR, 'last_run_graph_error.json')
            with open(latest_err, 'w', encoding='utf-8') as fh:
                json.dump(err, fh, ensure_ascii=False, indent=2)
            app.logger.error('Saved run_graph error to %s', err_path)
        except Exception:
            app.logger.exception('Failed to persist run_graph error')
        return jsonify({'status': 'error', 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/list_classes', methods=['GET'])
def list_classes():
    """Scan `filters`, `routers`, `tasks` packages and return available module.ClassName strings.

    Returns JSON: { "filters": [...], "routers": [...], "tasks": [...] }
    """
    def scan_package(subdir):
        out = []
        base = os.path.join(ROOT_DIR, subdir)
        if not os.path.exists(base):
            return out
        for root, dirs, files in os.walk(base):
            for f in files:
                if not f.endswith('.py'):
                    continue
                path = os.path.join(root, f)
                rel = os.path.relpath(path, ROOT_DIR)
                module = rel.replace(os.sep, '.')[:-3]  # strip .py
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        src = fh.read()
                    tree = ast.parse(src)
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            out.append(f"{module}.{node.name}")
                except Exception:
                    continue
        return out

    res = {
        'filters': scan_package('filters'),
        'routers': scan_package('routers'),
        'tasks': scan_package('tasks')
    }
    return jsonify(res)


@app.route('/class_info', methods=['GET'])
def class_info():
    """Return docstring and basic signature/method docstrings for a given class name.

    Query param: `class` e.g. filters.al_filter.ActiveLearningFilter
    Response: {"class": fullname, "doc": str or None, "init_params": [...], "methods": [{name, doc}], "module": module}
    """
    fullname = request.args.get('class') or request.args.get('name')
    if not fullname:
        return jsonify({'error': 'missing class param, use ?class=module.ClassName'}), 400
    try:
        module_name, class_name = fullname.rsplit('.', 1)
    except Exception:
        return jsonify({'error': 'invalid class name format; expected module.ClassName'}), 400

    # Try to locate source file for module
    module_path = os.path.join(ROOT_DIR, *module_name.split('.'))
    py_path = module_path + '.py'
    init_path = os.path.join(module_path, '__init__.py')
    source_path = None
    if os.path.exists(py_path):
        source_path = py_path
    elif os.path.exists(init_path):
        source_path = init_path
    else:
        return jsonify({'error': f'module source not found for {module_name}'}), 404

    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                doc = ast.get_docstring(node)
                init_params = []
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        mdoc = ast.get_docstring(item)
                        if item.name == '__init__':
                            # collect arg names excluding self
                            args = [a.arg for a in item.args.args]
                            if args and args[0] == 'self':
                                args = args[1:]
                            init_params = args
                        else:
                            methods.append({'name': item.name, 'doc': mdoc})
                return jsonify({'class': fullname, 'module': module_name, 'doc': doc, 'init_params': init_params, 'methods': methods})
        return jsonify({'error': f'class {class_name} not found in module {module_name}'}), 404
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/submit_review', methods=['POST'])
def submit_review():
    """Accept a single reviewed sample and action. Payload: { sample: {...}, action: 'approve'|'reject'|'update' }
    If action=='approve', also append the sample to `knowledge_base.json` in the repo root.
    All submissions are appended to `human_review_submissions.json` for audit.
    """
    payload = request.get_json(force=True) or {}
    sample = payload.get('sample')
    action = payload.get('action', 'update')
    if not sample:
        return jsonify({'status': 'error', 'error': 'missing sample in payload'}), 400

    submissions_path = os.path.join(ROOT_DIR, 'human_review_submissions.json')
    try:
        existing = []
        if os.path.exists(submissions_path):
            import json
            with open(submissions_path, 'r', encoding='utf-8') as fh:
                existing = json.load(fh) or []
        import json, time
        entry = {'sample': sample, 'action': action, 'ts': int(time.time())}
        existing.append(entry)
        with open(submissions_path, 'w', encoding='utf-8') as fh:
            json.dump(existing, fh, ensure_ascii=False, indent=2)

        # If approved, also append to knowledge_base.json
        if action == 'approve':
            kb_path = os.path.join(ROOT_DIR, 'knowledge_base.json')
            kb = []
            if os.path.exists(kb_path):
                try:
                    with open(kb_path, 'r', encoding='utf-8') as kf:
                        kb = json.load(kf) or []
                except Exception:
                    kb = []
            kb.append(sample)
            with open(kb_path, 'w', encoding='utf-8') as kf:
                json.dump(kb, kf, ensure_ascii=False, indent=2)
        # remove the submitted sample from pending review queue if present
        try:
            rq = read_review_queue() or []
            def same(a, b):
                if not a or not b:
                    return False
                if isinstance(a, dict) and isinstance(b, dict):
                    # match by id/qid if available
                    if a.get('id') and b.get('id') and a.get('id') == b.get('id'):
                        return True
                    if a.get('qid') and b.get('qid') and a.get('qid') == b.get('qid'):
                        return True
                    # fallback to compare question/text fields
                    if a.get('question') and b.get('question') and a.get('question') == b.get('question'):
                        return True
                    if a.get('text') and b.get('text') and a.get('text') == b.get('text'):
                        return True
                return a == b

            new_rq = [it for it in rq if not same(it, sample)]
            with open(REVIEW_QUEUE_PATH, 'w', encoding='utf-8') as fh:
                import json
                json.dump(new_rq, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/review_submissions', methods=['GET'])
def review_submissions():
    """Return the current pending review queue stored server-side."""
    try:
        items = read_review_queue()
        return jsonify({'status': 'ok', 'items': items})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
