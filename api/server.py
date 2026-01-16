from flask import Flask, request, jsonify, Response, stream_with_context
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
from threading import Lock

# Progress store for runs: run_id -> { node_id: {current, total, info}, ... }
PROGRESS = {}
PROGRESS_LOCK = Lock()

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
    try:
        app.logger.info('instantiate_class_from_module: creating %s.%s args=%s kwargs=%s', module_name, class_name, args, kwargs)
    except Exception:
        try:
            import logging
            logging.getLogger(__name__).info('instantiate_class_from_module: creating %s.%s args=%s kwargs=%s', module_name, class_name, args, kwargs)
        except Exception:
            pass
    # instantiate
    obj = cls(*args, **kwargs)
    # If this object provides a `filter` method, wrap it with a proxy that logs entry/exit/timing
    try:
        if hasattr(obj, 'filter') and callable(getattr(obj, 'filter')):
            # attach app.logger handlers to the object's module logger so its info/debug logs appear
            try:
                import logging as _logging
                mod_name = obj.__class__.__module__ if hasattr(obj, '__class__') else None
                if mod_name:
                    tgt_logger = _logging.getLogger(mod_name)
                    try:
                        tgt_logger.handlers = list(app.logger.handlers)
                        tgt_logger.setLevel(app.logger.level or _logging.INFO)
                        tgt_logger.propagate = False
                        app.logger.info('Bound app.logger handlers to logger %s', mod_name)
                    except Exception:
                        pass
                    # also bind base_structure.active_learning so Selector/Embeddings logs appear
                    try:
                        aln = 'base_structure.active_learning'
                        al_logger = _logging.getLogger(aln)
                        al_logger.handlers = list(app.logger.handlers)
                        al_logger.setLevel(app.logger.level or _logging.INFO)
                        al_logger.propagate = False
                        app.logger.info('Bound app.logger handlers to logger %s', aln)
                    except Exception:
                        pass
            except Exception:
                pass
            class _FilterProxy:
                def __init__(self, target):
                    object.__setattr__(self, '_target', target)

                def filter(self, *f_args, **f_kwargs):
                    import time, threading
                    app.logger.info('Filter proxy: calling %s.filter args=%s kwargs=%s', module_name + '.' + class_name, f_args, f_kwargs)

                    result_container = {}

                    def _run():
                        try:
                            result_container['res'] = self._target.filter(*f_args, **f_kwargs)
                        except Exception as e:
                            result_container['err'] = e

                    th = threading.Thread(target=_run, daemon=True)
                    t0 = time.time()
                    th.start()

                    # heartbeat while thread is alive
                    while th.is_alive():
                        app.logger.info('Filter proxy: %s.filter still running (%.2fs elapsed)', module_name + '.' + class_name, time.time() - t0)
                        th.join(timeout=2.0)

                    if 'err' in result_container:
                        app.logger.exception('Filter proxy: %s.filter raised: %s', module_name + '.' + class_name, result_container['err'])
                        raise result_container['err']

                    res = result_container.get('res')
                    app.logger.info('Filter proxy: %s.filter returned type=%s elapsed=%.2fs', module_name + '.' + class_name, type(res).__name__ if res is not None else 'None', time.time() - t0)
                    return res

                def __getattr__(self, name):
                    return getattr(self._target, name)

                def __setattr__(self, name, value):
                    # forward attribute sets to the target unless setting internal attr
                    if name == '_target':
                        object.__setattr__(self, name, value)
                    else:
                        try:
                            setattr(self._target, name, value)
                        except Exception:
                            object.__setattr__(self, name, value)

            return _FilterProxy(obj)
    except Exception:
        pass
    return obj


def process_run_graph(payload: Dict[str, Any], run_id: str):
    """Worker to process a run_graph payload in background and update PROGRESS."""
    try:
        import json, time
        dump_path = os.path.join(ROOT_DIR, f"run_graph_payload_{int(time.time())}.json")
        with open(dump_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
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
    def make_cb(node_id):
        def cb(current, total, info=None):
            try:
                with PROGRESS_LOCK:
                    run = PROGRESS.get(run_id)
                    if run is None:
                        return
                    run_nodes = run.setdefault('nodes', {})
                    run_nodes.setdefault(node_id, {})
                    run_nodes[node_id].update({'current': int(current), 'total': int(total), 'info': info})
            except Exception:
                pass
        return cb

    try:
        for nid in order:
            # ensure PROGRESS knows the human-readable name for this node
            try:
                display_name = node_map.get(nid, {}).get('data', {}).get('label') or node_map.get(nid, {}).get('data', {}).get('name') or nid
                with PROGRESS_LOCK:
                    run = PROGRESS.get(run_id)
                    if run is not None:
                        run_nodes = run.setdefault('nodes', {})
                        run_nodes.setdefault(nid, {})
                        run_nodes[nid].setdefault('name', display_name)
            except Exception:
                pass
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
            # keep predecessor outputs as separate elements so datasets aren't flattened
            flat_input = inputs
            try:
                types = [type(it).__name__ for it in flat_input]
                lengths = [len(it) if hasattr(it, '__len__') else 'NA' for it in flat_input]
                app.logger.info('Node %s predecessors types=%s lengths=%s', nid, types, lengths)
                # If this is node n6, emit a full dump of predecessors for debugging
                if nid == 'n6':
                    try:
                        preds = []
                        for i, it in enumerate(flat_input):
                            entry = {'index': i, 'type': type(it).__name__}
                            try:
                                if isinstance(it, list):
                                    entry['len'] = len(it)
                                    previews = []
                                    for v in it[:3]:
                                        if isinstance(v, dict):
                                            previews.append({k: v.get(k) for k in list(v.keys())[:5]})
                                        else:
                                            previews.append(str(v)[:200])
                                    entry['preview'] = previews
                                elif isinstance(it, dict):
                                    entry['keys'] = list(it.keys())[:20]
                                    lists = {k: (len(v) if hasattr(v, '__len__') else 'NA') for k, v in it.items() if isinstance(v, list)}
                                    if lists:
                                        entry['list_fields'] = lists
                                    try:
                                        if 'error' in it:
                                            entry['error'] = it.get('error')
                                        if 'trace' in it:
                                            tr = it.get('trace')
                                            if isinstance(tr, str):
                                                entry['trace'] = tr[:4000]
                                    except Exception:
                                        pass
                                else:
                                    entry['repr'] = str(it)[:500]
                            except Exception as e:
                                entry['error'] = str(e)
                            preds.append(entry)
                        import json as _json
                        app.logger.info('Node %s full predecessors dump=%s', nid, _json.dumps(preds, ensure_ascii=False))
                    except Exception:
                        app.logger.exception('Failed to dump full predecessors for %s', nid)
            except Exception:
                pass

            # helper to pick a dataset-like input: prefer the first list-valued predecessor
            def _pick_dataset(inp_list):
                if not inp_list:
                    return []
                list_candidates = [it for it in inp_list if isinstance(it, list)]
                if list_candidates:
                    try:
                        dict_lists = [lst for lst in list_candidates if len(lst) > 0 and all(isinstance(x, dict) for x in lst)]
                        if dict_lists:
                            for lst in dict_lists:
                                if all('route' in x for x in lst):
                                    return lst
                            for lst in dict_lists:
                                if all((('text' in x) or ('question' in x)) for x in lst):
                                    return lst
                            return max(dict_lists, key=lambda x: len(x))

                        def is_primitive_list(lst):
                            return all(isinstance(x, (str, int, float, bool)) for x in lst)

                        non_primitive = [lst for lst in list_candidates if len(lst) > 0 and not is_primitive_list(lst)]
                        if non_primitive:
                            return max(non_primitive, key=lambda x: len(x))
                        pass
                    except Exception:
                        return list_candidates[0]
                for it in inp_list:
                    try:
                        if hasattr(it, 'to_list') and callable(getattr(it, 'to_list')):
                            return it.to_list()
                    except Exception:
                        pass
                for it in inp_list:
                    if hasattr(it, '__iter__') and not isinstance(it, (str, dict)):
                        try:
                            lst = list(it)
                            if isinstance(lst, list):
                                return lst
                        except Exception:
                            pass
                return inp_list[0]

            # Node dispatch
            if ntype == 'LoadData':
                if 'samples' in params:
                    data = params['samples']
                else:
                    ds = params.get('dataset', 'squad')
                    max_samples = int(params.get('max_samples', 200))
                    if ds == 'squad':
                        try:
                            from datasets import SquadDataset
                            tmp_path = os.path.join(ROOT_DIR, 'squad_tmp.json')
                            data = SquadDataset.from_url(save_path=tmp_path, max_samples=max_samples)
                        except Exception as e:
                            app.logger.exception('LoadData: failed to download SQuAD: %s', e)
                            tried = []
                            data = []
                            for p in ('squad_train.json', 'squad_tmp.json', 'squad.json', 'train-v1.1.json', 'train.json'):
                                fp = p if os.path.isabs(p) else os.path.join(ROOT_DIR, p)
                                tried.append(fp)
                                try:
                                    from datasets import SquadDataset
                                    if os.path.exists(fp):
                                        try:
                                            data = SquadDataset.from_file(fp, max_samples=max_samples)
                                            app.logger.info('LoadData: loaded SQuAD from %s', fp)
                                            break
                                        except Exception as e2:
                                            app.logger.exception('LoadData: failed to read SQuAD file %s: %s', fp, e2)
                                            data = []
                                except Exception as e3:
                                    app.logger.exception('LoadData: error while attempting to access %s: %s', fp, e3)
                                    data = []
                            if not data:
                                app.logger.warning('LoadData: failed to obtain SQuAD from URL and local files %s', tried)
                                data = []
                    else:
                        import json as _json
                        try:
                            with open(ds, 'r', encoding='utf-8') as f:
                                data = _json.load(f)
                        except Exception:
                            data = []
                try:
                    if hasattr(data, 'to_list') and callable(getattr(data, 'to_list')):
                        data = data.to_list()
                    elif hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
                        data = list(data)
                except Exception:
                    pass
                context[nid] = data

            elif ntype == 'Filter':
                clsname = params.get('filter_class', 'filters.ActiveLearningFilter')
                fparams = params.get('filter_params', {})
                try:
                    filt = instantiate_class_from_module(clsname, **fparams)
                    input_data = _pick_dataset(flat_input)
                    try:
                        app.logger.info('Filter node %s receiving input type=%s len=%s', nid, type(input_data), len(input_data) if hasattr(input_data, '__len__') else 'NA')
                    except Exception:
                        pass
                    if not isinstance(input_data, list):
                        input_data = [input_data]
                    # initialize progress for this node (ensure name present)
                    try:
                        with PROGRESS_LOCK:
                            run = PROGRESS.get(run_id)
                            if run is not None:
                                run_nodes = run.setdefault('nodes', {})
                                run_nodes.setdefault(nid, {})
                                run_nodes[nid].setdefault('name', node.get('data', {}).get('label') or node.get('data', {}).get('name') or nid)
                                run_nodes[nid].update({'current': 0, 'total': int(len(input_data))})
                    except Exception:
                        pass
                    try:
                        app.logger.info('Filter node %s calling filter() on %d samples', nid, len(input_data))
                        # set progress callback if filter supports it
                        try:
                            if hasattr(filt, 'progress_cb'):
                                filt.progress_cb = make_cb(nid)
                        except Exception:
                            pass
                        out = filt.filter(input_data)
                        # Normalize filter output: only allow a list of dicts (filtered samples)
                        try:
                            if isinstance(out, list):
                                cleaned = [o for o in out if isinstance(o, dict)]
                                if len(cleaned) != len(out):
                                    app.logger.info('Filter node %s: dropped %d non-dict items from filter output', nid, len(out) - len(cleaned))
                                out = cleaned
                            elif isinstance(out, dict) and 'items' in out and isinstance(out['items'], list):
                                cleaned = [o for o in out['items'] if isinstance(o, dict)]
                                out = cleaned
                                app.logger.info('Filter node %s: normalized dict->items list output to %d items', nid, len(out))
                            else:
                                # unexpected type -> coerce to empty list
                                app.logger.info('Filter node %s: unexpected filter output type %s; coercing to empty list', nid, type(out))
                                out = []
                        except Exception:
                            app.logger.exception('Filter node %s: error normalizing filter output', nid)
                            out = []
                        app.logger.info('Filter node %s filter() returned type=%s count=%s', nid, type(out), len(out) if hasattr(out, '__len__') else 'NA')
                        try:
                            with PROGRESS_LOCK:
                                run = PROGRESS.get(run_id)
                                if run is not None:
                                    run_nodes = run.setdefault('nodes', {})
                                    run_nodes.setdefault(nid, {})
                                    run_nodes[nid].update({'current': run_nodes[nid].get('total', len(input_data))})
                        except Exception:
                            pass
                    except Exception as e:
                        app.logger.exception('Filter node %s filter() raised: %s', nid, e)
                        raise
                except Exception as e:
                    out = {'error': str(e), 'trace': traceback.format_exc()}
                context[nid] = out

            elif ntype == 'Router':
                clsname = params.get('router_class', 'routers.KNNRouter')
                rparams = params.get('router_params', {}) or {}
                candidate_llms = params.get('candidate_llms', [])
                try:
                    module_name, class_name = clsname.rsplit('.', 1)
                    mod = importlib.import_module(module_name)
                    cls = getattr(mod, class_name)
                    try:
                        from misc.llm_provider import LocalLLM, APILLM
                        from annotation import Annotator
                        r_llm_mode = params.get('llm_mode', 'local')
                        r_api_config = params.get('api_config', {}) or {}
                        r_task_class = params.get('task_class', 'tasks.QATask')
                        llm_dict = {}
                        if r_llm_mode == 'local':
                            for name in candidate_llms:
                                llm_dict[name] = LocalLLM(name)
                        else:
                            for name in candidate_llms:
                                conf = r_api_config.get(name, {})
                                llm_dict[name] = APILLM(conf.get('api_url', ''), conf.get('api_key'), conf.get('extra_headers'))
                        task_mod, task_name = r_task_class.rsplit('.', 1)
                        task_cls = getattr(importlib.import_module(task_mod), task_name)
                        task = task_cls()
                        annotator = Annotator(candidate_llms, llm_dict, task=task)
                    except Exception:
                        annotator = None
                    init_kwargs = dict(rparams)
                    init_kwargs['annotator'] = annotator
                    init_kwargs['candidate_llms'] = candidate_llms
                    # attach annotator progress callback if available
                    try:
                        if annotator is not None:
                            annotator.progress_cb = make_cb(nid)
                    except Exception:
                        pass
                    router = cls(**init_kwargs)
                    # pick the most appropriate dataset from predecessors
                    try:
                        input_data = _pick_dataset(flat_input)
                    except Exception:
                        input_data = []
                    try:
                        previews = []
                        for i, it in enumerate(flat_input):
                            try:
                                item_preview = None
                                if isinstance(it, list):
                                    item_preview = [ (v.get('id') if isinstance(v, dict) else str(v)) for v in it[:3] ]
                                elif isinstance(it, dict):
                                    item_preview = {k: it[k] for k in list(it.keys())[:3]}
                                else:
                                    item_preview = str(it)[:200]
                            except Exception:
                                item_preview = 'unpreviewable'
                            previews.append({'index': i, 'type': type(it).__name__, 'len': (len(it) if hasattr(it, '__len__') else 'NA'), 'preview': item_preview})
                        app.logger.info('Router node %s selected input type=%s len=%s previews=%s', nid, type(input_data), (len(input_data) if hasattr(input_data, '__len__') else 'NA'), previews)
                    except Exception:
                        pass
                    if not isinstance(input_data, list):
                        input_data = [input_data]
                    # initialize progress for router node (ensure name present)
                    try:
                        with PROGRESS_LOCK:
                            run = PROGRESS.get(run_id)
                            if run is not None:
                                run_nodes = run.setdefault('nodes', {})
                                run_nodes.setdefault(nid, {})
                                run_nodes[nid].setdefault('name', node.get('data', {}).get('label') or node.get('data', {}).get('name') or nid)
                                run_nodes[nid].update({'current': 0, 'total': int(len(input_data))})
                    except Exception:
                        pass
                    try:
                        if hasattr(router, 'progress_cb'):
                            try:
                                router.progress_cb = make_cb(nid)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    out = router.route(input_data)
                    try:
                        with PROGRESS_LOCK:
                            run = PROGRESS.get(run_id)
                            if run is not None:
                                run_nodes = run.setdefault('nodes', {})
                                run_nodes.setdefault(nid, {})
                                run_nodes[nid].update({'current': run_nodes[nid].get('total', len(input_data))})
                    except Exception:
                        pass
                except Exception as e:
                    out = {'error': str(e), 'trace': traceback.format_exc()}
                context[nid] = out

            elif ntype == 'Annotate' or ntype == 'Annotation':
                candidate_llms = params.get('candidate_llms', [])
                llm_mode = params.get('llm_mode', 'local')
                api_config = params.get('api_config', {})
                task_class = params.get('task_class', 'tasks.QATask')
                try:
                    from misc.llm_provider import LocalLLM, APILLM
                    llm_dict = {}
                    if llm_mode == 'local':
                        for name in candidate_llms:
                            llm_dict[name] = LocalLLM(name)
                    else:
                        for name in candidate_llms:
                            conf = api_config.get(name, {})
                            llm_dict[name] = APILLM(conf.get('api_url', ''), conf.get('api_key'), conf.get('extra_headers'))
                    task_mod, task_name = task_class.rsplit('.', 1)
                    task_cls = getattr(importlib.import_module(task_mod), task_name)
                    task = task_cls()
                    from annotation import Annotator
                    # Determine confidence threshold to use for enqueuing human review.
                    # Priority: Annotate node params -> Annotator default
                    conf_thr = None
                    try:
                        if params.get('min_confidence') is not None:
                            conf_thr = float(params.get('min_confidence'))
                    except Exception:
                        conf_thr = None
                    if conf_thr is not None:
                        annotator = Annotator(candidate_llms, llm_dict, confidence_threshold=conf_thr, task=task)
                    else:
                        annotator = Annotator(candidate_llms, llm_dict, task=task)
                    if flat_input and all(isinstance(it, dict) for it in flat_input):
                        input_data = flat_input
                    else:
                        input_data = _pick_dataset(flat_input)
                    try:
                        app.logger.info('Annotate node %s receiving input type=%s len=%s', nid, type(input_data), len(input_data) if hasattr(input_data, '__len__') else 'NA')
                    except Exception:
                        pass
                    # initialize progress for annotate node
                    try:
                        total_samples = int(len(input_data)) if hasattr(input_data, '__len__') else 0
                        with PROGRESS_LOCK:
                            run = PROGRESS.get(run_id)
                            if run is not None:
                                run_nodes = run.setdefault('nodes', {})
                                run_nodes.setdefault(nid, {})
                                run_nodes[nid].setdefault('name', node.get('data', {}).get('label') or node.get('data', {}).get('name') or nid)
                                run_nodes[nid].update({'current': 0, 'total': total_samples})
                    except Exception:
                        pass
                    # attach annotator progress callback
                    try:
                        annotator.progress_cb = make_cb(nid)
                    except Exception:
                        pass
                    if isinstance(input_data, list) and all(isinstance(x, dict) for x in input_data):
                        try:
                            routed_only = all(('route' in x) and not (('text' in x) or ('question' in x)) for x in input_data)
                        except Exception:
                            routed_only = False
                        if routed_only:
                            text_lists = [it for it in flat_input if isinstance(it, list) and all(isinstance(x, dict) and (('text' in x) or ('question' in x)) for x in it)]
                            if text_lists:
                                text_list = text_lists[0]
                                if len(text_list) == len(input_data):
                                    merged = []
                                    for a, b in zip(text_list, input_data):
                                        merged.append({**a, **b})
                                    input_data = merged
                                    app.logger.info('Annotate node %s merged route info into %d samples', nid, len(input_data))
                    if isinstance(input_data, list) and (not input_data or not all(isinstance(x, dict) for x in input_data)):
                        try:
                            candidate_list = None
                            for it in flat_input:
                                if isinstance(it, list) and len(it) > 0 and all(isinstance(x, dict) for x in it):
                                    candidate_list = it
                                    break
                            if candidate_list is None:
                                for it in flat_input:
                                    if isinstance(it, dict):
                                        if 'samples' in it and isinstance(it['samples'], list) and len(it['samples'])>0:
                                            candidate_list = it['samples']
                                            break
                                        for v in it.values():
                                            if isinstance(v, list) and len(v)>0 and all(isinstance(x, dict) for x in v):
                                                candidate_list = v
                                                break
                                        if candidate_list is not None:
                                            break
                            if candidate_list is not None:
                                input_data = candidate_list
                                app.logger.info('Annotate node %s replaced non-dict list with predecessor list of %d samples', nid, len(input_data))
                            else:
                                app.logger.warning('Annotate node %s received non-dict list as input; no predecessor with samples found — converting items to text samples', nid)
                                input_data = [({'text': str(x)}) for x in input_data]
                        except Exception:
                            pass
                    try:
                        previews = []
                        if isinstance(input_data, list):
                            for i, it in enumerate(input_data[:3]):
                                if isinstance(it, dict):
                                    previews.append({k: it.get(k) for k in list(it.keys())[:5]})
                                else:
                                    previews.append(str(it)[:200])
                        else:
                            previews = str(input_data)[:200]
                        app.logger.info('Annotate node %s final input preview=%s', nid, previews)
                    except Exception:
                        pass
                    out = annotator.annotate_batch(input_data)
                    # ensure marked finished
                    try:
                        with PROGRESS_LOCK:
                            run = PROGRESS.get(run_id)
                            if run is not None:
                                run_nodes = run.setdefault('nodes', {})
                                run_nodes.setdefault(nid, {})
                                run_nodes[nid].update({'current': run_nodes[nid].get('total', (len(input_data) if hasattr(input_data, '__len__') else 0))})
                    except Exception:
                        pass
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
                input_data = _pick_dataset(flat_input)
                path = params.get('path')
                # Optionally filter by confidence threshold before saving.
                # Priority: use the Annotate node's `min_confidence` (if present),
                # otherwise fall back to any threshold provided in this Output node's params.
                try:
                    thr_val = None
                    try:
                        # search for Annotate node in the graph
                        for n in nodes:
                            try:
                                nlabel = n.get('data', {}).get('label') or n.get('type')
                                if nlabel in ('Annotate', 'Annotation'):
                                    ap = n.get('data', {}).get('params', {}) or {}
                                    if ap.get('min_confidence') is not None:
                                        thr_val = ap.get('min_confidence')
                                        break
                            except Exception:
                                continue
                    except Exception:
                        thr_val = None
                    # if not found on Annotate, check Output params (legacy)
                    if thr_val is None:
                        thr_val = params.get('min_confidence') if params.get('min_confidence') is not None else (params.get('confidence_threshold') if params.get('confidence_threshold') is not None else params.get('threshold'))

                    if thr_val is not None:
                        try:
                            thr = float(thr_val)
                        except Exception:
                            thr = None
                    else:
                        thr = None

                    def _is_num(v):
                        try:
                            float(v)
                            return True
                        except Exception:
                            return False

                    def _filter_items(data, thr):
                        if thr is None:
                            return data
                        # If top-level list of dicts, keep those with numeric confidence > thr
                        if isinstance(data, list) and len(data) > 0 and all(isinstance(x, dict) for x in data):
                            out_list = []
                            for x in data:
                                try:
                                    c = x.get('confidence')
                                    if c is None:
                                        continue
                                    if float(c) > thr:
                                        out_list.append(x)
                                except Exception:
                                    continue
                            return out_list
                        # If dict with 'items' list of dicts, filter that list
                        if isinstance(data, dict) and 'items' in data and isinstance(data['items'], list) and all(isinstance(x, dict) for x in data['items']):
                            new = dict(data)
                            new['items'] = [x for x in data['items'] if (x.get('confidence') is not None and _is_num(x.get('confidence')) and float(x.get('confidence')) > thr)]
                            return new
                        return data

                    input_data = _filter_items(input_data, thr)
                except Exception:
                    pass
                if path:
                    try:
                        import json as _json
                        full_path = path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)
                        parent = os.path.dirname(full_path)
                        if parent and not os.path.exists(parent):
                            os.makedirs(parent, exist_ok=True)
                        def _to_jsonable(o):
                            try:
                                _json.dumps(o)
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
                            _json.dump(jsonable, f, ensure_ascii=False, indent=2)
                        out = {'saved_to': full_path}
                    except Exception as e:
                        out = {'error': str(e), 'trace': traceback.format_exc()}
                else:
                    out = input_data
                context[nid] = out

            elif ntype == 'Task':
                context[nid] = params

            elif ntype == 'CandidateLLMs':
                context[nid] = params.get('candidate_llms', [])

            else:
                context[nid] = params

        outputs = {}
        for n in nodes:
            ntype = n.get('data', {}).get('label') or n.get('type')
            if ntype == 'Output':
                outputs[n['id']] = context.get(n['id'])
        try:
            append_review_queue(server_review_items)
        except Exception:
            pass

        def make_jsonable(obj):
            import json as _json
            try:
                _json.dumps(obj)
                return obj
            except TypeError:
                if isinstance(obj, dict):
                    return {k: make_jsonable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [make_jsonable(x) for x in obj]
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
                try:
                    if hasattr(obj, '__dict__'):
                        return make_jsonable({k: v for k, v in obj.__dict__.items()})
                except Exception:
                    pass
                try:
                    return str(obj)
                except Exception:
                    return None

        serializable_context = {k: make_jsonable(v) for k, v in context.items()}
        serializable_outputs = {k: make_jsonable(v) for k, v in outputs.items()}

        with PROGRESS_LOCK:
            PROGRESS[run_id]['status'] = 'finished'
            PROGRESS[run_id]['outputs'] = serializable_outputs
            PROGRESS[run_id]['context'] = serializable_context
    except Exception as e:
        try:
            with PROGRESS_LOCK:
                PROGRESS[run_id]['status'] = 'error'
                PROGRESS[run_id]['error'] = str(e)
                PROGRESS[run_id]['trace'] = traceback.format_exc()
        except Exception:
            pass
        app.logger.exception('Error processing run_graph %s', run_id)


@app.route('/run_graph', methods=['POST'])
def run_graph():
    payload = request.get_json(force=True)
    import time, uuid
    run_id = str(int(time.time() * 1000)) + '-' + uuid.uuid4().hex[:6]
    # initialize progress entry
    with PROGRESS_LOCK:
        PROGRESS[run_id] = {'status': 'running', 'nodes': {}}
    app.logger.info('Created run_id %s for run_graph', run_id)
    # process the graph asynchronously so frontend can poll progress
    from threading import Thread
    t = Thread(target=process_run_graph, args=(payload, run_id), daemon=True)
    t.start()
    return jsonify({'status': 'ok', 'run_id': run_id})
 


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


@app.route('/run_progress', methods=['GET'])
def run_progress():
    run_id = request.args.get('run_id')
    if not run_id:
        # return all runs
        with PROGRESS_LOCK:
            return jsonify({'status': 'ok', 'runs': PROGRESS})
    with PROGRESS_LOCK:
        run = PROGRESS.get(run_id)
        if run is None:
            return jsonify({'status': 'error', 'error': f'run_id {run_id} not found'}), 404
        return jsonify({'status': 'ok', 'run_id': run_id, 'progress': run})


@app.route('/read_output', methods=['GET'])
def read_output():
    """Read the JSON file saved by an Output node and return its parsed contents.

    Query params: `run_id` and `node_id` OR `path` (relative to project root).
    """
    run_id = request.args.get('run_id')
    node_id = request.args.get('node_id')
    path = request.args.get('path')
    import json as _json

    target = None
    if run_id and node_id:
        with PROGRESS_LOCK:
            run = PROGRESS.get(run_id)
            if not run:
                return jsonify({'status': 'error', 'error': f'run_id {run_id} not found'}), 404
            outputs = run.get('outputs') or {}
            node_out = outputs.get(node_id)
            if node_out and isinstance(node_out, dict) and node_out.get('saved_to'):
                target = node_out.get('saved_to')
    if not target and path:
        # allow relative paths under project root
        target = path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)

    if not target:
        return jsonify({'status': 'error', 'error': 'no saved output found for given run_id/node_id or path missing'}), 400

    # security: ensure target is inside project root
    try:
        full = os.path.abspath(target)
        if not full.startswith(os.path.abspath(ROOT_DIR)):
            return jsonify({'status': 'error', 'error': 'access denied to path'}), 403
        if not os.path.exists(full):
            return jsonify({'status': 'error', 'error': f'file not found: {full}'}), 404
        with open(full, 'r', encoding='utf-8') as fh:
            data = _json.load(fh)
        return jsonify({'status': 'ok', 'path': full, 'data': data})
    except Exception as e:
        app.logger.exception('Failed reading output file %s: %s', target, e)
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/run_progress_stream', methods=['GET'])
def run_progress_stream():
    """Server-Sent Events stream for live run progress updates. Query param `run_id` required."""
    run_id = request.args.get('run_id')
    if not run_id:
        return jsonify({'status': 'error', 'error': 'missing run_id param'}), 400

    def event_stream():
        import json, time
        last = None
        while True:
            with PROGRESS_LOCK:
                run = PROGRESS.get(run_id)
                data = json.dumps(run or {})
            if data != last:
                last = data
                yield f"data: {data}\n\n"
            if run is None:
                # if run was removed, end stream
                break
            try:
                parsed = json.loads(data)
                status = parsed.get('status')
                if status in ('finished', 'error'):
                    break
            except Exception:
                pass
            time.sleep(0.5)

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')


@app.route('/progress', methods=['GET'])
def progress_alias():
    # alias for backward-compatibility / convenience
    return run_progress()


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
