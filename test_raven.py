"""
Diagnostica per la wake word Raven.

Non usa il microfono: rigioca i template registrati dentro al detector e dice
se Raven li riconosce e con quanto margine. Serve a capire se il problema sta
nei template, nelle soglie, o nell'audio in ingresso.

    python test_raven.py

Su hardware reale conviene lanciarlo dopo ogni sessione di `--record`.
"""
import sys
import wave
import numpy as np
from pathlib import Path

import jarvis
from rhasspywake_raven import Raven, Template


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == jarvis.RATE, f"{path.name}: {wf.getframerate()} Hz"
        assert wf.getnchannels() == 1, f"{path.name}: non mono"
        return wf.readframes(wf.getnframes())


def build_raven(templates, average: bool) -> Raven:
    """Stesse impostazioni di jarvis.load_raven(), ma template scelti da noi."""
    if average and len(templates) > 1:
        templates = [Template.average_templates(templates, name="avg")]
    return Raven(
        templates=templates,
        keyword_name=jarvis.WAKE_WORD,
        recorder=jarvis.make_recorder(),
        probability_threshold=jarvis.RAVEN_PROBABILITY_THRESHOLD,
        distance_threshold=jarvis.RAVEN_DISTANCE_THRESHOLD,
        minimum_matches=jarvis.RAVEN_MINIMUM_MATCHES,
        refractory_sec=jarvis.RAVEN_REFRACTORY_SEC,
        failed_matches_to_refractory=jarvis.RAVEN_FAILED_MATCHES_TO_REFRACTORY,
        skip_probability_threshold=jarvis.RAVEN_SKIP_PROBABILITY_THRESHOLD,
    )


def feed(raven: Raven, audio: bytes, pad_sec: float = 0.7):
    """
    Manda l'audio a Raven a blocchi come farebbe il microfono.
    Il silenzio in coda serve a far scorrere la finestra fino in fondo.
    Ritorna (rilevato, distanza_migliore, probabilita_migliore).
    """
    pad = b"\x00\x00" * int(jarvis.RATE * pad_sec)
    stream = pad + audio + pad
    step = jarvis.CHUNK * 2

    detected = False
    best_dist, best_prob = None, None
    for i in range(0, len(stream) - step + 1, step):
        hit = raven.process_chunk(stream[i:i + step])
        for d in raven.last_distances:
            if d is not None and (best_dist is None or d < best_dist):
                best_dist = d
        for p in raven.last_probabilities:
            if p is not None and (best_prob is None or p > best_prob):
                best_prob = p
        if hit:
            detected = True
            break
    return detected, best_dist, best_prob


def main():
    paths = sorted(jarvis.TEMPLATE_DIR.glob("*.wav"))
    if not paths:
        print(f"❌ Nessun template in {jarvis.TEMPLATE_DIR}")
        return 1

    print(f"📁 {jarvis.TEMPLATE_DIR}  ({len(paths)} template)")
    print(f"⚙️  soglie: probabilità ≥ {jarvis.RAVEN_PROBABILITY_THRESHOLD}, "
          f"distanza ≤ {jarvis.RAVEN_DISTANCE_THRESHOLD}, "
          f"media template = {jarvis.RAVEN_AVERAGE_TEMPLATES}")

    # ---------- 1. salute dei file ----------
    print("\n=== 1. Qualità dei template ===")
    print(f"{'file':<17}{'durata':>8}{'DC':>8}{'RMS':>8}   note")
    audios = {}
    for p in paths:
        raw = read_wav(p)
        audios[p] = raw
        a = np.frombuffer(raw, dtype=np.int16)
        dur, dc, level = len(a) / jarvis.RATE, a.mean(), jarvis.rms(raw)
        notes = []
        if dur > jarvis.MAX_TEMPLATE_SEC: notes.append("troppo lungo")
        if dur < 0.3: notes.append("troppo corto")
        if abs(dc) > 500: notes.append(f"DC residuo {dc:+.0f}")
        if level < jarvis.MIN_TEMPLATE_RMS: notes.append("troppo debole")
        print(f"{p.name:<17}{dur:>7.2f}s{dc:>8.0f}{level:>8.0f}   {', '.join(notes) or 'ok'}")

    templates = {p: Raven.wav_to_template(str(p), name=p.name) for p in paths}

    # ---------- 2. si riconosce da solo? ----------
    # Un template DEVE essere rilevato dal detector costruito su se stesso.
    # Se fallisce qui, il problema è la registrazione, non le soglie.
    print("\n=== 2. Auto-riconoscimento (template contro se stesso) ===")
    self_ok = 0
    for p in paths:
        r = build_raven([templates[p]], average=False)
        det, dist, prob = feed(r, audios[p])
        self_ok += det
        print(f"  {p.name:<17} {'✅ RILEVATO' if det else '❌ NON rilevato'}"
              f"   distanza {dist if dist is None else round(dist,3)}"
              f"   probabilità {prob if prob is None else round(prob,3)}")
    print(f"  -> {self_ok}/{len(paths)}")

    # ---------- 3. configurazione di produzione ----------
    print("\n=== 3. Configurazione di produzione (come jarvis.py) ===")
    prod = build_raven(list(templates.values()), average=jarvis.RAVEN_AVERAGE_TEMPLATES)
    prod_ok = 0
    for p in paths:
        r = build_raven(list(templates.values()), average=jarvis.RAVEN_AVERAGE_TEMPLATES)
        det, dist, prob = feed(r, audios[p])
        prod_ok += det
        print(f"  {p.name:<17} {'✅' if det else '❌'}"
              f"   distanza {dist if dist is None else round(dist,3)}"
              f"   probabilità {prob if prob is None else round(prob,3)}")
    print(f"  -> {prod_ok}/{len(paths)} rilevati con le impostazioni attuali")

    # ---------- 4. media sì / media no ----------
    print("\n=== 4. Media dei template: aiuta o peggiora? ===")
    for avg in (True, False):
        n = 0
        for p in paths:
            r = build_raven(list(templates.values()), average=avg)
            n += feed(r, audios[p])[0]
        print(f"  RAVEN_AVERAGE_TEMPLATES = {str(avg):<5} -> {n}/{len(paths)}")

    # ---------- 5. quale soglia servirebbe ----------
    print("\n=== 5. Soglia di distanza necessaria ===")
    for thr in (0.10, 0.15, 0.22, 0.30, 0.40, 0.50):
        n = 0
        for p in paths:
            r = build_raven(list(templates.values()), average=jarvis.RAVEN_AVERAGE_TEMPLATES)
            r.distance_threshold = thr
            n += feed(r, audios[p])[0]
        mark = "  <-- attuale" if abs(thr - jarvis.RAVEN_DISTANCE_THRESHOLD) < 1e-9 else ""
        print(f"  distance_threshold {thr:.2f} -> {n}/{len(paths)}{mark}")

    # ---------- 6. falsi positivi ----------
    print("\n=== 6. Falsi positivi ===")
    silence = b"\x00\x00" * (jarvis.RATE * 3)
    rng = np.random.default_rng(0)
    noise = (rng.normal(0, 500, jarvis.RATE * 3)).astype(np.int16).tobytes()
    for name, audio in (("silenzio", silence), ("rumore di fondo", noise)):
        r = build_raven(list(templates.values()), average=jarvis.RAVEN_AVERAGE_TEMPLATES)
        det, dist, _ = feed(r, audio)
        print(f"  {name:<18} {'❌ FALSO POSITIVO' if det else '✅ nessuna attivazione'}")

    # ---------- verdetto ----------
    print("\n=== Verdetto ===")
    if prod_ok == len(paths):
        print("  Raven riconosce tutti i template. Se dal vivo non funziona, il")
        print("  problema è l'audio del microfono, non i template né le soglie.")
    elif self_ok < len(paths):
        print("  Alcuni template non riconoscono nemmeno se stessi: sono registrati")
        print("  male. Cancellali e rifai `python jarvis.py --record`.")
    else:
        print("  I template sono validi singolarmente ma la configurazione di")
        print("  produzione ne perde alcuni: guarda i punti 4 e 5 qui sopra.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
