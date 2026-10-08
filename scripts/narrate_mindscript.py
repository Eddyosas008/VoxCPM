"""Narrer les scripts écrits dans MindScript Audio, et rapporter l'audio dans l'app.

Le script est écrit là-bas (https://mindscriptaudio.com) : méditation, hypnose,
relaxation, avec ses silences en repères ``[PAUSE:…]``. La narration se fait
ici, avec les voix et la chaîne des livres audio. Ce script fait le trajet :

1. lister ou prendre les scripts du compte (clé d'API ou session) ;
2. écrire chacun dans ``output/mindscript/<slug>/script.txt`` ;
3. le narrer avec ``narrate_book.py`` — un MP3 par script, sans générique,
   repères de silence respectés, reprise au segment près grâce au cache ;
4. optionnellement le relire (``--relecture``) et le déposer dans la
   bibliothèque audio du compte (``--upload``, session seulement).

Un script déjà narré et inchangé est sauté. Un script modifié dans l'app est
renarré : les segments qui n'ont pas bougé viennent du cache, les autres sont
synthétisés. Un script en échec n'arrête pas les autres.

Entrée
------
Clé d'API (lecture seule, comptes Team/Enterprise) : ``MINDSCRIPT_API_KEY``
ou ``--api-key``. Session (tout, dépôt compris) : ``MINDSCRIPT_EMAIL`` et
``MINDSCRIPT_PASSWORD`` ou ``--email`` / ``--password``. ``MINDSCRIPT_URL``
pour une autre instance (par exemple ``http://localhost:5000`` en local).

Exemples
--------
  # Voir ce qu'il y a :
  python scripts/narrate_mindscript.py --list

  # Narrer un script (id ou début d'id), sur GPU :
  python scripts/narrate_mindscript.py --script 3f2a9c --device cuda

  # Tout ce qui a changé depuis une date, relu, puis déposé dans l'app :
  python scripts/narrate_mindscript.py --all --since 2026-10-01 --relecture --upload
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
PYTHON = sys.executable

sys.path.insert(0, str(REPO))

from narration import mindscript  # noqa: E402

LEXIQUE_FR = "conf/pronunciation_fr.json"

#: Les identifiants peuvent vivre ici plutôt que dans l'environnement : un
#: fichier ``CLÉ=valeur`` par ligne, ignoré par git, lisible dans un éditeur.
FICHIER_ENV_LOCAL = REPO / "conf" / "mindscript.local.env"


def charger_env_local(chemin: pathlib.Path = FICHIER_ENV_LOCAL) -> list[str]:
    """Poser dans l'environnement les ``MINDSCRIPT_*`` du fichier local.

    L'environnement prime : une valeur déjà posée n'est pas écrasée. Rend
    les noms chargés, pour pouvoir dire d'où vient l'accès.
    """
    charges: list[str] = []
    try:
        lignes = chemin.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return charges
    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        cle, valeur = ligne.split("=", 1)
        cle, valeur = cle.strip(), valeur.strip().strip('"').strip("'")
        if cle.startswith("MINDSCRIPT_") and valeur and not os.environ.get(cle):
            os.environ[cle] = valeur
            charges.append(cle)
    return charges


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def child_env() -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    return env


def run(cmd: list[str], logfile: pathlib.Path | None = None) -> tuple[int, str]:
    """Lancer une étape ; sa sortie va dans un fichier, pas en mémoire."""
    if logfile:
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=child_env())
        return p.returncode, logfile.read_text(encoding="utf-8", errors="replace")[-600:]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=REPO, env=child_env())
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    acces = ap.add_argument_group("accès à MindScript")
    acces.add_argument("--base-url", default=os.environ.get("MINDSCRIPT_URL", mindscript.URL_PAR_DEFAUT))
    acces.add_argument("--api-key", default=os.environ.get("MINDSCRIPT_API_KEY"))
    acces.add_argument("--email", default=os.environ.get("MINDSCRIPT_EMAIL"))
    acces.add_argument("--password", default=os.environ.get("MINDSCRIPT_PASSWORD"))

    quoi = ap.add_argument_group("quels scripts")
    quoi.add_argument("--list", action="store_true", help="lister les scripts et s'arrêter")
    quoi.add_argument("--script", action="append", metavar="ID",
                      help="un script par son id (ou le début de l'id) ; répétable")
    quoi.add_argument("--all", action="store_true", help="tous les scripts du compte")
    quoi.add_argument("--language", help="ne garder que cette langue (fr, en)")
    quoi.add_argument("--since", metavar="AAAA-MM-JJ",
                      help="ne garder que les scripts mis à jour depuis cette date")

    comment = ap.add_argument_group("narration")
    comment.add_argument("--outdir", default="output/mindscript")
    comment.add_argument("--voice", help="voix du catalogue (défaut : selon la langue — "
                                         f"{', '.join(f'{k} → {v}' for k, v in mindscript.VOIX_PAR_LANGUE.items())})")
    comment.add_argument("--device", default="cuda")
    comment.add_argument("--qc-retries", default="2")
    comment.add_argument("--cover", help="pochette à embarquer dans les MP3 (sinon aucune)")
    comment.add_argument("--export-only", action="store_true",
                         help="écrire les .txt et s'arrêter, sans narrer")
    comment.add_argument("--force", action="store_true", help="renarrer même un script inchangé")
    comment.add_argument("--relecture", action="store_true",
                         help="relire chaque script par reconnaissance vocale et réparer "
                              "les segments tronqués (GPU)")
    comment.add_argument("--relecture-retries", default="2")
    comment.add_argument("--upload", action="store_true",
                         help="déposer chaque MP3 produit dans la bibliothèque audio "
                              "du compte MindScript (session seulement)")
    return ap


def creer_client(args) -> mindscript.Client:
    """Le client, entré par clé ou par session — ou une explication claire."""
    client = mindscript.Client(args.base_url, api_key=args.api_key)
    if args.email and args.password:
        client.connecter(args.email, args.password)
    elif not args.api_key:
        raise SystemExit(
            "Aucun accès à MindScript. Soit dans l'environnement, soit dans "
            f"{FICHIER_ENV_LOCAL} (une ligne CLÉ=valeur par identifiant, fichier ignoré "
            "par git) :\n"
            "  MINDSCRIPT_EMAIL=...  et  MINDSCRIPT_PASSWORD=...   (tout, dépôt compris)\n"
            "  ou MINDSCRIPT_API_KEY=...   (lecture seule, comptes Team/Enterprise)"
        )
    if args.upload and client.utilisateur is None:
        raise SystemExit("--upload demande une session (courriel + mot de passe) : "
                         "l'API par clé ne sait pas déposer d'audio.")
    return client


def commande(script: mindscript.Script, txt: pathlib.Path, outdir: pathlib.Path,
             voix: str, args) -> list[str]:
    """La commande de narration d'un script — la même pour toute reprise."""
    cmd = [PYTHON, "scripts/narrate_book.py", str(txt),
           "--voice", voix, "--title", script.title,
           "--language", script.language if script.language in ("fr", "en") else "fr",
           "--assemble", "mp3", "--no-credits",
           "--outdir", str(outdir), "--device", args.device,
           "--qc-retries", str(args.qc_retries)]
    cmd += ["--cover", args.cover] if args.cover else ["--no-cover"]
    if script.language == "fr":
        cmd += ["--lexicon", LEXIQUE_FR]
    return cmd


def livrable(outdir: pathlib.Path) -> pathlib.Path | None:
    trouves = sorted(outdir.glob("*_complet.mp3")) if outdir.is_dir() else []
    return trouves[0] if trouves else None


def purger_sauf_cache(outdir: pathlib.Path, garder: set[str]) -> None:
    """Effacer la prise précédente, mais ni le cache ni le script lui-même."""
    for item in outdir.iterdir():
        if item.name in garder or item.name == ".cache":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def rapport_relecture(outdir: pathlib.Path) -> dict | None:
    try:
        r = json.loads((outdir / "relecture_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return r if isinstance(r, dict) and r.get("segments") else None


def afficher_liste(scripts: list[mindscript.Script]) -> None:
    print(f"{len(scripts)} script(s)\n")
    print(f"  {'id':<10} {'langue':<6} {'min':>4}  {'mis à jour':<10}  titre")
    for s in scripts:
        print(f"  {s.id[:8]:<10} {s.language:<6} {s.estimated_minutes:>4}  "
              f"{s.updated_at[:10]:<10}  {s.title[:60]}")


def main() -> int:
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    charges = charger_env_local()
    args = build_parser().parse_args()

    client = creer_client(args)
    log(f"MindScript : {args.base_url} ({client.mode}"
        + (f", identifiants lus dans {FICHIER_ENV_LOCAL.name}" if charges else "") + ")")

    scripts = client.scripts()
    if args.list or not (args.script or args.all):
        afficher_liste(scripts)
        if not args.list:
            print("\nRien à narrer : précisez --script ID ou --all.")
        return 0

    voulus = mindscript.selectionner(scripts, ids=args.script, langue=args.language,
                                     depuis=args.since)
    if args.script:
        connus = {s.id for s in voulus}
        for demande in args.script:
            if not any(i == demande or i.startswith(demande) for i in connus):
                log(f"script introuvable : {demande}")
    if not voulus:
        log("aucun script ne correspond")
        return 1

    outroot = pathlib.Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)
    state_path = outroot / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    def save() -> None:
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"{len(voulus)} script(s) à traiter")
    echecs = 0
    for i, s in enumerate(voulus, 1):
        # La liste ne porte pas toujours le contenu entier : on relit le script.
        if not s.content:
            s = client.script(s.id)
        dossier = outroot / s.slug()
        dossier.mkdir(parents=True, exist_ok=True)
        txt = dossier / "script.txt"
        txt.write_text(s.texte(), encoding="utf-8")
        (dossier / "script.json").write_text(json.dumps({
            "id": s.id, "title": s.title, "language": s.language,
            "estimated_minutes": s.estimated_minutes, "updated_at": s.updated_at,
            "project_id": s.project_id, "empreinte": s.empreinte(), "config": s.config,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[{i}/{len(voulus)}] {s.title} ({s.language}, ~{s.estimated_minutes} min) -> {dossier}")

        if args.export_only:
            continue

        st = state.get(s.id, {})
        mp3 = livrable(dossier)
        if (not args.force and st.get("status") == "done"
                and st.get("empreinte") == s.empreinte() and mp3 is not None):
            log("    à jour, sauté")
            continue

        voix = args.voice or s.voix_par_defaut()
        if not voix:
            log(f"    aucune voix par défaut pour la langue « {s.language} » — "
                "précisez --voice ; script écarté")
            state[s.id] = {"status": "failed", "stage": "voix", "slug": s.slug(),
                           "detail": f"aucune voix pour {s.language}"}
            save()
            echecs += 1
            continue

        if st.get("empreinte") not in (None, s.empreinte()) or args.force:
            # Le texte a changé : la prise d'avant ne vaut plus, le cache si.
            purger_sauf_cache(dossier, {"script.txt", "script.json"})
            log("    script modifié depuis la dernière prise — renarré (cache conservé)")

        state[s.id] = {"status": "running", "slug": s.slug(), "title": s.title,
                       "empreinte": s.empreinte(),
                       "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        save()
        journal = dossier / "narration.log"
        cmd = commande(s, txt, dossier, voix, args)
        t0 = time.time()
        rc, tail = run(cmd, journal)
        mins = (time.time() - t0) / 60
        mp3 = livrable(dossier)
        if rc != 0 or mp3 is None:
            log(f"    ÉCHEC après {mins:.0f} min (code {rc}) — voir {journal}")
            state[s.id].update(status="failed", stage="narration", detail=tail[-400:])
            save()
            echecs += 1
            continue
        log(f"    narré en {mins:.0f} min, voix « {voix} »")

        if args.relecture:
            rc, _ = run([PYTHON, "scripts/relire_livre.py", str(dossier), "--device", args.device,
                         "--repair", "--retries", str(args.relecture_retries)], journal)
            rapport = rapport_relecture(dossier)
            if rapport is None:
                log(f"    relecture impossible (code {rc}) — script gardé tel quel")
            else:
                log(f"    relecture : {rapport['segments']} relu(s), {rapport.get('tronques', 0)} "
                    f"tronqué(s), {rapport.get('repares', 0)} réparé(s)")
                if rapport.get("repares"):
                    rc, tail = run(cmd, journal)
                    if rc != 0 or livrable(dossier) is None:
                        log(f"    réassemblage échoué (code {rc})")
                        state[s.id].update(status="failed", stage="réassemblage",
                                           detail=tail[-400:])
                        save()
                        echecs += 1
                        continue
                    mp3 = livrable(dossier)

        state[s.id].update(status="done", mp3=str(mp3), voice=voix,
                           minutes=round(mins, 1),
                           finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        if args.upload:
            try:
                depot = client.televerser_audio(mp3, nom=s.title)
                state[s.id]["asset_id"] = str(depot.get("id", ""))
                log(f"    déposé dans MindScript (asset {depot.get('id', '?')})")
            except mindscript.ErreurMindScript as error:
                # Le MP3 existe et est bon : un dépôt raté ne défait pas la
                # narration, il se relance.
                log(f"    dépôt échoué : {error} — relancez avec --upload")
                state[s.id]["upload_error"] = str(error)
        save()
        log(f"    terminé -> {mp3}")

    faits = sum(1 for v in state.values() if v.get("status") == "done")
    log(f"fini — {faits} script(s) narrés au total, {echecs} échec(s) sur cette passe")
    return 1 if echecs else 0


if __name__ == "__main__":
    raise SystemExit(main())
