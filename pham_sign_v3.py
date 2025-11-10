# =============================================================
#!/usr/bin/env python3
# pham_sign_v3.py
# =============================================================
# 📜 PHAM Sign v3 — 코드 기여도 서명 시스템 (기여 점수 + 스팸 필터 + 실행 결과 체인)
#
# 🎯 목적:
#   - 코드 파일을 블록체인 형태로 서명/기록하여 코드의 "기여도"를 정량화
#   - 코드의 실제 실행 결과(exec output)를 블록에 포함하여 신뢰성 검증
#   - 스팸 또는 의미 없는 변경을 자동 필터링 (signals 기반)
#   - 각 코드 실행 시 새로운 블록이 자동으로 연결됨 (previous_hash → hash)
#
# ⚙️ 사용 방법 (Usage):
#   python3 pham_sign_v3.py <파일이름> --author <작성자> --desc "<설명>" [--exec "<실행명령>"]
#
# 💡 예시:
#   python3 pham_sign_v3.py new_run_quick.py \
#       --author "GNJz" \
#       --desc "PTP 실험 통합판 (안정화 버전)" \
#       --exec "python3 {file}"
#
# 📂 결과물:
#   - 블록체인 로그 파일 생성: pham_chain_<filename>.json
#   - 각 블록에는 다음 정보 포함:
#       • title (파일명)
#       • author (작성자)
#       • description (설명)
#       • contribution score / label (기여도)
#       • CID / hash / timestamp
#       • 실행 결과(exec_output)
#
# ✅ 특징:
#   - 동일 해시(=동일 코드)는 서명 생략 (중복 차단)
#   - 파일이 변경되면 새로운 블록이 자동 생성되어 체인에 연결됨
#   - 안전 실행(sandbox) 및 스코어 기반 스팸 필터링 내장
#
# =============================================================

# =============================================================
#!/usr/bin/env python3
# pham_sign_v3.py
# PHAM Sign v3 — contribution scoring + spam filter + EXEC OUTPUT CHAINING
# Usage:
#   python3 pham_sign_v3.py
#  <file> --author GNJz --desc "message" [--exec "pytest -q"] 
# =============================================================

# Qquarts co Present # 지은이 : GNJz

import argparse
import hashlib
import json
import time
import subprocess
import shlex
import difflib
import ast
import tempfile
import os
import shutil
import sys
from pathlib import Path

# 체인 파일을 서명 대상 파일명 기준으로 분리 생성
if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    target_name = Path(sys.argv[1]).stem
    CHAIN_FILE = f"pham_chain_{target_name}.json"
else:
    CHAIN_FILE = "pham_chain_default.json"

# Config (튜닝 가능)
W_BYTE = 0.25
W_TEXT = 0.35
W_AST = 0.30
W_EXEC = 0.10

MIN_BYTE_CHANGE = 0.002   # 파일 바이트 변화 비율(0.2%) 미만 -> 의심
THRESHOLD_LOW = 0.08      # 최종 score < 0.08 -> LOW / 스팸 후보
ALLOWED_EXEC_BINS = ("python3", "pytest", "node", "bash")  # 실행 화이트리스트에 bash 추가

# ANSI colors
GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'; CYAN = '\033[96m'; ENDC = '\033[0m'

def sha256_bytes(b: bytes) -> str:
    """바이트 데이터의 SHA256 해시를 계산합니다."""
    return hashlib.sha256(b).hexdigest()

def safe_run(cmd_list, timeout=10, cwd=None):
    """지정된 커맨드를 안전하게 실행합니다. (rc, stdout, stderr) 반환"""
    try:
        # shell=False로 설정하여 쉘 인젝션 위험을 최소화합니다.
        p = subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout, cwd=cwd, shell=False)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"cmd not found: {cmd_list[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:
        return 1, "", str(e)

def load_chain():
    """체인 로그 파일을 로드합니다."""
    if Path(CHAIN_FILE).exists():
        try:
            return json.loads(Path(CHAIN_FILE).read_text(encoding="utf-8"))
        except Exception:
            return []
    else:
        return []

def save_chain(chain):
    """체인 로그 파일을 저장합니다."""
    Path(CHAIN_FILE).write_text(json.dumps(chain, indent=2, ensure_ascii=False), encoding="utf-8")

def find_latest_block_with_title(chain, title):
    """특정 파일 제목의 최신 블록을 찾습니다."""
    for b in reversed(chain):
        if isinstance(b.get("data"), dict) and b["data"].get("title") == title:
            return b
    return None

def compute_byte_ratio(old_bytes: bytes, new_bytes: bytes):
    """이전 바이트 대비 변경된 바이트 비율을 계산합니다."""
    if not old_bytes:
        return 1.0
    changed = sum(1 for i,(a,b) in enumerate(zip(old_bytes, new_bytes)) if a!=b)
    changed += abs(len(new_bytes)-len(old_bytes))
    denom = max(len(old_bytes), 1)
    return changed/denom

def text_similarity(old_text: str, new_text: str):
    """텍스트 유사도 (0.0: 완전 다름, 1.0: 동일)를 계산합니다."""
    if not old_text:
        return 0.0
    seq = difflib.SequenceMatcher(a=old_text, b=new_text)
    return seq.ratio()

def count_nodes_via_walk(tree):
    """ast.walk를 사용하여 AST 노드 개수를 계산합니다."""
    count = 0
    for _ in ast.walk(tree):
        count += 1
    return count

def ast_edit_distance(old_text: str, new_text: str):
    """AST 노드 개수 차이를 기반으로 편집 거리를 정규화하여 반환합니다 (0..1)."""
    try:
        old_ast = ast.parse(old_text)
        new_ast = ast.parse(new_text)
    except Exception:
        return 0.5
    
    try:
        oc = count_nodes_via_walk(old_ast)
        nc = count_nodes_via_walk(new_ast)
    except Exception:
        return 0.5
        
    if oc == 0:
        return 1.0
    return abs(nc-oc)/max(oc, nc)

def execute_and_score(exec_cmd_template: str, new_file_path: Path, old_text: str, previous_exec_output: str, safe_tmpdir: Path):
    """
    Exec 명령어를 실행하여 새로운 출력(new_output)을 얻고, 
    이전 출력(previous_exec_output)과 비교하여 점수를 계산합니다.
    """
    if not exec_cmd_template or "{file}" not in exec_cmd_template:
        return 0.0, None, "no-exec", ""
    
    parts = shlex.split(exec_cmd_template)
    if len(parts) == 0:
        return 0.0, None, "bad-cmd", ""
    base = parts[0]
    if not any(base.endswith(a) for a in ALLOWED_EXEC_BINS):
        return 0.0, None, f"bin-not-allowed: {base}", ""
    
    try:
        run_dir = new_file_path.parent
        safe_cmd = exec_cmd_template.format(file=shlex.quote(str(new_file_path)))

        # bash -c를 사용하여 명령 실행, 실행 경로는 원본 파일이 위치한 디렉터리로 설정
        rc, out, err = safe_run(["bash", "-c", safe_cmd], timeout=10, cwd=str(run_dir))
        
        new_output = out or ""
        
        if rc != 0:
            return 0.0, (rc,out,err), "exec-failed", new_output
        
        if not previous_exec_output:
            # 이전 출력이 없는 경우 -> 새로운 출력에 대한 작은 긍정 기여도 부여
            return 0.2, (rc,out,err), "exec-ok-newbase", new_output
        
        # 이전 실행 결과와 새로운 실행 결과의 유사도 측정
        sim = difflib.SequenceMatcher(a=previous_exec_output, b=new_output).ratio()
        
        # 유사도가 낮을수록 (즉, 변화가 클수록) 점수가 높음 (1.0 - sim)
        score = 1.0 - sim
        return score, (rc,out,err), "exec-ok", new_output
        
    except Exception as e:
        return 0.0, None, f"exec-exc:{e}", ""

def compute_contribution_score(old_bytes, old_text, new_bytes, new_text, exec_cmd, new_path, safe_tmpdir, previous_exec_output):
    """각 신호를 계산하고 가중 평균을 통해 최종 기여도 점수를 계산합니다."""
    
    # 1. Byte Signal
    byte_ratio = compute_byte_ratio(old_bytes, new_bytes)
    byte_signal = min(byte_ratio, 1.0) # 0..1로 클램프

    # 2. Text Signal (1 - 유사도)
    txt_sim = text_similarity(old_text or "", new_text or "")
    text_signal = 1.0 - txt_sim

    # 3. AST Signal
    ast_signal = ast_edit_distance(old_text or "", new_text or "")

    # 4. Exec Signal (Exec Output Chaining 적용)
    exec_signal = 0.0
    exec_info = None
    exec_meta = None
    new_exec_output = ""
    
    if exec_cmd:
        exec_signal, exec_info, exec_meta, new_exec_output = execute_and_score(
            exec_cmd, new_path, old_text or "", previous_exec_output or "", safe_tmpdir
        )

    # 가중치 합 계산
    total = (W_BYTE*byte_signal + W_TEXT*text_signal + W_AST*ast_signal + W_EXEC*exec_signal)
    
    # 가중치 합을 이용하여 정규화
    weight_sum = (W_BYTE + W_TEXT + W_AST + (W_EXEC if exec_cmd else 0))
    
    if weight_sum == 0:
        score = 0.0
    else:
        score = total/weight_sum
        
    score = max(0.0, min(1.0, score)) # 0..1로 클램프
    
    return {
        "score": score,
        "signals": {
            "byte_signal": byte_signal,
            "text_signal": text_signal,
            "ast_signal": ast_signal,
            "exec_signal": exec_signal
        },
        "exec_meta": exec_meta,
        "exec_info": exec_info,
        "new_exec_output": new_exec_output # 새롭게 계산된 실행 결과를 반환하여 블록에 저장
    }

def classify_label(score):
    """점수를 기반으로 기여도 레이블을 분류합니다."""
    if score >= 0.8:
        return "A_HIGH"
    if score >= 0.5:
        return "B_MEDIUM"
    if score >= THRESHOLD_LOW:
        return "C_LOW"
    return "SPAM_LOW"

def should_spam_flag(score, signals, new_bytes, old_bytes, old_text):
    """스팸 플래그를 결정하는 규칙을 적용합니다."""
    if old_bytes:
        byte_changed_frac = compute_byte_ratio(old_bytes, new_bytes)
        if byte_changed_frac < MIN_BYTE_CHANGE and signals["ast_signal"] < 0.01 and signals["text_signal"] < 0.02:
            return True, "tiny-byte-no-ast"
    if score < THRESHOLD_LOW:
        return True, "low-overall-score"
    return False, None

def ipfs_cat(cid):
    """IPFS에서 CID에 해당하는 내용을 가져옵니다."""
    try:
        # 타임아웃을 10초로 늘림
        rc, out, err = safe_run(["ipfs", "cat", cid], timeout=15)
        if rc == 0:
            return out
        return None
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", help="file to sign")
    p.add_argument("--author", default="unknown")
    p.add_argument("--desc", default="")
    p.add_argument("--exec", default=None, help="optional exec command template (must contain {file})")
    args = p.parse_args()

    target = Path(args.file)
    if not target.exists():
        print(f"{RED}[err]{ENDC} 파일 없음: {target}")
        return

    # 새 파일 읽기
    new_bytes = target.read_bytes()
    try:
        new_text = new_bytes.decode("utf-8")
    except Exception:
        new_text = ""

    new_hash = sha256_bytes(new_bytes)

    chain = load_chain()
    latest_same = find_latest_block_with_title(chain, target.name)

    old_bytes = b""
    old_text = ""
    old_cid = None
    previous_exec_output = "" # V5: 이전 블록의 실행 결과를 저장할 변수

    if latest_same:
        old_hash = latest_same["data"].get("hash")
        old_cid = latest_same["data"].get("cid")
        
        if old_hash == new_hash:
            print(f"{YELLOW}동일 해시 발견 — 파일 변경 없음. 서명 생략.{ENDC}")
            return
            
        # V5: 이전 실행 결과 로드 (없으면 빈 문자열)
        previous_exec_output = latest_same["data"].get("exec_output", "")
            
        # 2. IPFS에서 이전 파일 내용 로드 시도
        if old_cid and old_cid != "CID unavailable":
            out = ipfs_cat(old_cid)
            if out is not None:
                old_bytes = out.encode("utf-8")
                old_text = out
        else:
            old_bytes = b""
            old_text = ""

    # 안전한 임시 디렉터리 생성 및 사용
    tmpdir = Path(tempfile.mkdtemp(prefix="pham_sign_", dir="/tmp"))
    try:
        # 기여도 점수 계산
        # V5: previous_exec_output 인자 추가
        result = compute_contribution_score(
            old_bytes, old_text, new_bytes, new_text, args.exec, target, tmpdir, previous_exec_output
        )
        score = result["score"]
        label = classify_label(score)
        spam, spam_reason = should_spam_flag(score, result["signals"], new_bytes, old_bytes, old_text)
        
        # 블록 데이터 구성
        block_data = {
            "title": target.name,
            "author": args.author,
            "organization": "Qquarts Co",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hash": new_hash,
            "cid": "CID unavailable",
            "description": args.desc,
            "contribution": {
                "score": round(score, 4),
                "label": label,
                "signals": {k: round(v,4) for k,v in result["signals"].items()},
                "spam": bool(spam),
                "spam_reason": spam_reason
            },
            "exec_output": result["new_exec_output"] # V5: 새로운 실행 결과 저장
        }

        # IPFS add 시도 (논블로킹)
        try:
            rc, out, err = safe_run(["ipfs", "add", "-Q", str(target)], timeout=8)
            if rc == 0 and out.strip():
                block_data["cid"] = out.strip()
        except Exception:
            pass

        # 체인에 블록 추가
        chain = load_chain()
        if not chain:
            # Genesis 블록 생성
            chain = [{
                "index": 0,
                "timestamp": time.time(),
                "data": {"name": "PHAM Genesis", "author": "System", "exec_output": ""},
                "previous_hash": "0",
                "hash": "0"
            }]
        latest = chain[-1]
        
        new_block = {
            "index": len(chain),
            "timestamp": time.time(),
            "data": block_data,
            "previous_hash": latest.get("hash"),
        }
        
        # 블록 해시 계산 (deterministic)
        block_string = f"{new_block['index']}{new_block['timestamp']}{json.dumps(new_block['data'], sort_keys=True)}{new_block['previous_hash']}"
        new_block["hash"] = hashlib.sha256(block_string.encode()).hexdigest()
        
        chain.append(new_block)
        save_chain(chain)

        # 최종 출력 요약
        emoji = {"A_HIGH":"⭐","B_MEDIUM":"✅","C_LOW":"⚠️","SPAM_LOW":"🚫"}.get(label, "❓")
        color = GREEN if label=="A_HIGH" else (CYAN if label=="B_MEDIUM" else (YELLOW if label=="C_LOW" else RED))
        
        print(f"{color}{emoji} 기여도: {label} ({score:.4f}){ENDC}")
        print(f" signals: {result['signals']}")
        if result.get("exec_meta"):
            print(f" exec_meta: {result['exec_meta']}")
        if result.get("exec_info"):
            rc, out, err = result["exec_info"]
            out_preview = (out or "")[:100].replace('\n', ' ')
            if len(out or "") > 100:
                out_preview += "..."
            err_preview = (err or "")[:100].replace('\n', ' ')
            if len(err or "") > 100:
                err_preview += "..."
            print(f" exec_info: RC={rc}, Out='{out_preview}', Err='{err_preview}'")
        print(f" CID: {block_data['cid']}")
        print(f" Block {new_block['index']} added. Hash: {new_block['hash']}")
        print(f" 체인 파일: {CHAIN_FILE}")

        if spam:
            print(f"{RED}⚠️ 스팸 가능성 감지: {spam_reason}{ENDC}")
            
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()

# =============================================================
# PHAM-OPEN LICENSE v2.0 (Trust-Based Creative Ledger License)
# (C) 2025 Qquarts Co / GNJz
#
# 🪶 1. 기본 원칙 (Principles)
# 한국어 버전
# 이 라이선스는 법적 강제가 아닌, 신뢰·기록·기여를 바탕으로 한 새로운 오픈 코드 문화의 선언입니다.
# 모든 코드는 인간의 창의적 기여이며, 그 가치는 공개된 Ledger를 통해 투명하게 증명됩니다.
#
# English Version
# This license is a declaration of a new open-code culture founded on trust, record-keeping, and contribution,
# rather than legal compulsion. All code represents human creative contribution,
# and its value is transparently proven through a public Ledger.
#
# ... (이하 전체 PHAM-OPEN LICENSE v2.0 본문)
#
# “Trust as Law. Ledger as Proof. Code as Culture.”
# =============================================================