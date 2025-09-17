import subprocess, sys, yaml, os, re
cases = yaml.safe_load(open("eval/cases.yaml", "r", encoding="utf-8"))
ok = 0
for c in cases:
    env = os.environ.copy()
    env.update({"EMBED_BACKEND":"hf","LLM_BACKEND":"none","TOP_K":"4","MAX_SNIPPET":"900"})
    out = subprocess.run(
        ["python3","scripts/query.py", c["q"]],
        capture_output=True, text=True, env=env
    ).stdout
    miss = [s for s in c["must_contain"] if not re.search(re.escape(s), out, re.IGNORECASE)]
    if miss:
        print(f"❌ {c['q']}\n  Manque: {miss}\n")
    else:
        print(f"✅ {c['q']}")
        ok += 1
print(f"\n{ok}/{len(cases)} ok")
