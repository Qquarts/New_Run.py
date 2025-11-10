#!/usr/bin/env python3
import json
from pathlib import Path
CHAIN = "pham_chain_new_run_quick.json"
data = json.loads(Path(CHAIN).read_text())
for b in data:
    print(f"Block {b['index']}: {b['data']['title']} by {b['data']['author']}")
    print(f"  Score: {b['data']['contribution']['score']} ({b['data']['contribution']['label']})")
    print(f"  CID: {b['data']['cid']}")
    print(f"  Hash: {b['hash'][:16]}...\n")
