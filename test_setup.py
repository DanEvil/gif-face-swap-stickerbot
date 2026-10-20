import os
import subprocess
import sys
import tempfile

tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, ".env"), "w", encoding="utf-8") as f:
    f.write("BOT_TOKEN=\nADMIN_IDS=111\nREQUIRED_CHANNELS=@old\n")

answers = "222,333\n\n2\nhttps://t.me/ch1\n@ch2\n\nn\n"
p = subprocess.run([sys.executable, os.path.join(os.getcwd(), "setup.py")], cwd=tmp, input=answers,
                   capture_output=True, text=True, timeout=30,
                   encoding="utf-8", errors="replace")
print("=== OUT ===")
print(p.stdout.encode("utf-8", errors="replace").decode("utf-8", errors="ignore"))
print("=== ERR ===")
print(p.stderr[-400:] if p.stderr else "(none)")
print("=== .env ===")
with open(os.path.join(tmp, ".env"), encoding="utf-8") as f:
    print(", ".join(l.strip() for l in f if l.strip() and not l.startswith("BOT_TOKEN")))
