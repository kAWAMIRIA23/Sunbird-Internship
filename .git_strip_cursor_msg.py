"""Strip Cursor co-author trailer from commit messages (stdin -> stdout)."""
import sys

DROP = "Co-authored-by: Cursor <cursoragent@cursor.com>"
text = sys.stdin.read()
lines = text.splitlines(keepends=True)
sys.stdout.write("".join(l for l in lines if l.rstrip("\r\n") != DROP))
