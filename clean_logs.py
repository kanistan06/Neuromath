import os
# Read debug.log and output.log and rewrite as pure ASCII
for fname in ["debug.log", "output.log"]:
    path = os.path.join(os.path.dirname(__file__), fname)
    outpath = os.path.join(os.path.dirname(__file__), fname.replace(".", "_clean."))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Replace non-ASCII chars
        clean = content.encode("ascii", errors="replace").decode("ascii")
        with open(outpath, "w", encoding="ascii") as f:
            f.write(clean)
