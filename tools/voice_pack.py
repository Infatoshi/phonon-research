#!/usr/bin/env python3
"""Check a folder of your recordings and zip it in the layout the Phonon voice fine-tune notebook expects.

Standard library only (Python 3.9+). Nothing is uploaded anywhere; the zip is written next to the folder.

  python3 voice_pack.py MY_FOLDER --script reading_script.txt     # you recorded the reading script: 001.m4a, 002.m4a, ...
  python3 voice_pack.py MY_FOLDER                                  # MY_FOLDER already has transcripts.csv or one .txt per clip

What it does:
  * finds the audio files (.wav .flac .m4a .mp3, also .ogg .opus .webm .aac);
  * finds what was said in each: transcripts.csv in the folder (columns: file,text), or the line of the reading
    script whose number is the first number in the file name (007.m4a, "Recording 7.m4a", line_007.wav), or a .txt
    file with the same name as the clip;
  * checks clip lengths (needs ffprobe for formats other than .wav; without it those lengths are not checked) and
    the total amount of speech;
  * writes MY_FOLDER.zip containing phonon_voice/ with the audio files, transcripts.csv and terms.txt (if you have one).
Upload that zip in step 5 of the notebook, or put it in the Google Drive folder My Drive/Phonon/voice.
"""
import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import wave
import zipfile

AUDIO = (".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".webm", ".aac")
MIN_CLIP, MAX_CLIP, MIN_TOTAL, OK_TOTAL, GOOD_TOTAL = 1.0, 30.0, 2.0, 5.0, 10.0


def duration(path):
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path) as w:
                return w.getnframes() / float(w.getframerate())
        except (wave.Error, EOFError):
            pass  # not plain PCM: try ffprobe
    if shutil.which("ffprobe"):
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                           capture_output=True, text=True, check=False)
        try:
            return float(json.loads(r.stdout)["format"]["duration"])
        except (ValueError, KeyError):
            return -1.0  # unreadable
    return None  # unknown


def read_script(path):
    lines = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            m = re.match(r"^\s*(\d+)\s+(.+?)\s*$", raw)
            if m and not raw.lstrip().startswith("#"):
                lines[int(m.group(1))] = m.group(2)
    return lines


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if rows and rows[0][0].strip().lower() in ("file", "filename", "audio", "path", "name"):
        rows = rows[1:]
    return {os.path.basename(r[0].strip()): ",".join(r[1:]).strip() for r in rows if len(r) >= 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="the folder with your recordings")
    ap.add_argument("--script", help="the reading script you recorded (numbered lines)")
    ap.add_argument("--terms", help="a terms.txt word list (default: terms.txt in the folder, if any)")
    ap.add_argument("-o", "--out", help="zip to write (default: FOLDER.zip next to the folder)")
    a = ap.parse_args()
    folder = os.path.abspath(a.folder)
    if not os.path.isdir(folder):
        sys.exit(f"Not a folder: {folder}")
    names = sorted(n for n in os.listdir(folder) if not n.startswith("."))
    audio = [n for n in names if n.lower().endswith(AUDIO)]
    if not audio:
        sys.exit(f"No audio files ({', '.join(AUDIO)}) in {folder}.")
    texts, source = {}, None
    if "transcripts.csv" in names:
        texts, source = read_csv(os.path.join(folder, "transcripts.csv")), "transcripts.csv"
    elif a.script:
        script = read_script(a.script)
        for n in audio:
            m = re.search(r"\d+", os.path.splitext(n)[0])
            if m and int(m.group()) in script:
                texts[n] = script[int(m.group())]
        source = os.path.basename(a.script)
        nums = [int(re.search(r"\d+", os.path.splitext(n)[0]).group()) for n in audio if re.search(r"\d+", n)]
        dups = sorted({x for x in nums if nums.count(x) > 1})
        if dups:
            sys.exit(f"Several recordings carry the same line number: {dups[:10]}. Keep one recording per line.")
        unread = sorted(set(script) - set(nums))
        if unread:
            print(f"note: {len(unread)} script line(s) have no recording (fine): {unread[:15]}{' ...' if len(unread) > 15 else ''}")
    else:
        for n in audio:
            t = os.path.join(folder, os.path.splitext(n)[0] + ".txt")
            if os.path.exists(t):
                with open(t, encoding="utf-8") as f:
                    texts[n] = " ".join(f.read().split())
        source = "one .txt per clip" if texts else None
    if not texts:
        sys.exit("No transcripts found. Use --script reading_script.txt if you recorded the reading script (files named by "
                 "line number), or add transcripts.csv (columns: file,text) or one .txt file per recording.")
    missing = [n for n in audio if n not in texts]
    if missing:
        print(f"warning: {len(missing)} recording(s) have no transcript and are left out: {missing[:8]}")
    extra = sorted(set(texts) - set(audio)) if source == "transcripts.csv" else []
    if extra:
        print(f"warning: transcripts.csv names {len(extra)} file(s) that are not in the folder: {extra[:8]}")
    keep, total, unknown, problems = [], 0.0, 0, []
    for n in audio:
        if n not in texts:
            continue
        d = duration(os.path.join(folder, n))
        if d is None:
            unknown += 1
        elif d < 0:
            problems.append(f"{n}: cannot be read as audio")
            continue
        elif d < MIN_CLIP:
            problems.append(f"{n}: only {d:.1f} s")
            continue
        elif d > MAX_CLIP:
            problems.append(f"{n}: {d:.0f} s is too long (record one line per file, at most {MAX_CLIP:.0f} s)")
            continue
        else:
            total += d
        keep.append(n)
    for p in problems:
        print("skipped:", p)
    if not keep:
        sys.exit("No usable recordings left.")
    terms = a.terms or (os.path.join(folder, "terms.txt") if "terms.txt" in names else None)
    out = a.out or folder.rstrip(os.sep) + ".zip"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["file", "text"])
    for n in keep:
        w.writerow([n, texts[n]])
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in keep:
            z.write(os.path.join(folder, n), "phonon_voice/" + n)
        z.writestr("phonon_voice/transcripts.csv", buf.getvalue())
        if terms:
            z.write(terms, "phonon_voice/terms.txt")
    mins = total / 60
    known = f"{mins:.1f} min of speech" + (f" (+ {unknown} clip(s) of unknown length: install ffmpeg to check them)" if unknown else "")
    print(f"wrote {out}: {len(keep)} recordings, {known}, transcripts from {source}"
          + (", terms.txt included" if terms else ""))
    if not unknown and mins < MIN_TOTAL:
        print(f"warning: the notebook needs at least {MIN_TOTAL:.0f} minutes; record more before uploading.")
    elif not unknown and mins < OK_TOTAL:
        print(f"warning: under {OK_TOTAL:.0f} minutes the notebook's before/after test is small; {GOOD_TOTAL:.0f} is recommended.")
    elif not unknown and mins < GOOD_TOTAL:
        print(f"note: {GOOD_TOTAL:.0f} minutes or more gives a more reliable result.")


if __name__ == "__main__":
    main()
