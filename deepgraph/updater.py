"""Mise à jour de DeepGraph à partir des releases GitHub (sans Qt : utilisable en fenêtre et en headless).

Vérification (dans l'ordre) :
  1. jeton dans l'environnement (DEEPGRAPH_TOKEN, GITHUB_TOKEN ou GH_TOKEN) -> API GitHub ;
  2. GitHub CLI (`gh`) connecté -> `gh api` ;
  3. clone git de DeepGraph -> `git ls-remote --tags origin` (droits SSH / HTTPS de git) ;
  sinon : avertissement expliquant quoi installer (le dépôt est privé).

Installation :
  - clone git : `git fetch --tags` puis avance rapide de `main` jusqu'au tag de la release,
    uniquement sur `main` sans modification locale (sinon rien n'est touché) ;
  - copie (archive .zip / .tar.gz) : téléchargement de l'archive de la release et remplacement des
    fichiers ; un fichier modifié localement (blocks.yaml…) est gardé, la nouvelle version est posée
    à côté (`.nouveau`) ; sauvegarde des fichiers remplacés dans un dossier temporaire.

Sans terminal interactif (export par script, éditeur en QT_QPA_PLATFORM=offscreen, tâche planifiée…),
la mise à jour trouvée est installée automatiquement, avec les mêmes garde-fous ; elle sert au lancement
suivant. Un verrou empêche deux processus lancés en même temps d'installer ensemble.

DEEPGRAPH_NO_UPDATE=1 désactive la vérification au démarrage ; DEEPGRAPH_NO_AUTO_UPDATE=1 garde la
vérification sans terminal interactif mais n'installe plus rien automatiquement (simple message).
"""
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request

from . import __version__

REPO = "Joffrey-Michaud/DeepGraph"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, ".deepgraph_install.json")
TOKEN_VARS = ("DEEPGRAPH_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
USER_FILES = ("blocks.yaml",)          # réglages que l'utilisateur modifie souvent
TIMEOUT = 8

HOW_TO_ENABLE = (
    "Le dépôt DeepGraph est privé : pour vérifier les mises à jour, l'application doit pouvoir "
    "s'identifier auprès de GitHub. Au choix :\n"
    "  • installer GitHub CLI (https://cli.github.com), puis lancer « gh auth login » ;\n"
    "  • ou définir la variable d'environnement DEEPGRAPH_TOKEN avec un jeton GitHub ayant accès en "
    "lecture au dépôt " + REPO + " (jeton « fine-grained », permission Contents : Read-only) ;\n"
    "  • ou utiliser un clone git du dépôt (git clone git@github.com:" + REPO + ".git).\n"
    "Pour ne plus vérifier au démarrage : DEEPGRAPH_NO_UPDATE=1."
)


# --------------------------------------------------------------------- résultats
class Release:
    def __init__(self, tag, notes="", name=""):
        self.tag = tag
        self.version = parse_version(tag)
        self.notes = notes or ""
        self.name = name or tag

    @property
    def label(self):
        return ".".join(map(str, self.version))


class CheckResult:
    """status : 'available', 'uptodate', 'noauth', 'denied', 'offline', 'error'."""

    def __init__(self, status, release=None, message="", method=""):
        self.status, self.release, self.message, self.method = status, release, message, method

    def __repr__(self):
        return f"CheckResult({self.status!r}, {self.release and self.release.tag!r}, via={self.method!r})"


class NoAuth(Exception):
    pass


class Denied(Exception):
    pass


class Offline(Exception):
    pass


def parse_version(s):
    nums = re.findall(r"\d+", str(s))[:3]
    return tuple(int(n) for n in nums + ["0"] * (3 - len(nums)))


def current_version():
    return parse_version(__version__)


def _env_true(name):
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "non", "no")


def disabled():
    return _env_true("DEEPGRAPH_NO_UPDATE")


def auto_install_disabled():
    """sans terminal interactif : DEEPGRAPH_NO_AUTO_UPDATE=1 -> message seulement, pas d'installation"""
    return _env_true("DEEPGRAPH_NO_AUTO_UPDATE")


# ------------------------------------------------------------------ transports
def _token():
    for v in TOKEN_VARS:
        t = os.environ.get(v, "").strip()
        if t:
            return t
    return None


def _have_gh():
    return shutil.which("gh") is not None


def _api(path, binary=False):
    """GET sur l'API GitHub (jeton, sinon gh). Lève NoAuth / Denied / Offline."""
    tok = _token()
    if tok:
        req = urllib.request.Request("https://api.github.com/" + path, headers={
            "Authorization": "Bearer " + tok, "User-Agent": "DeepGraph-updater",
            "Accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise Denied(f"le jeton ({_token_name()}) n'a pas accès au dépôt {REPO} (HTTP {e.code})")
            raise Offline(f"GitHub a répondu HTTP {e.code}")
        except (urllib.error.URLError, OSError) as e:
            raise Offline(f"GitHub injoignable ({getattr(e, 'reason', e)})")
        return data if binary else json.loads(data.decode("utf-8"))
    if _have_gh():
        try:
            p = subprocess.run(["gh", "api", path], capture_output=True, timeout=TIMEOUT * 4)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise Offline(f"gh ne répond pas ({e})")
        if p.returncode != 0:
            err = p.stderr.decode("utf-8", "replace").strip()
            low = err.lower()
            if "auth login" in low or "not logged" in low or "authentication" in low:
                raise NoAuth("GitHub CLI est installé mais pas connecté (lancer « gh auth login »)")
            if "404" in err or "not found" in low or "403" in err:
                raise Denied(f"le compte GitHub connecté à gh n'a pas accès au dépôt {REPO}")
            raise Offline(err.splitlines()[-1] if err else "erreur de gh")
        return p.stdout if binary else json.loads(p.stdout.decode("utf-8"))
    raise NoAuth("ni GitHub CLI (gh) ni jeton GitHub disponibles")


def _token_name():
    return next((v for v in TOKEN_VARS if os.environ.get(v, "").strip()), "jeton")


# -------------------------------------------------------------------------- git
def _git(*args, timeout=30):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=timeout)


def git_state():
    """État du clone git de DeepGraph ; None si ce n'est pas un clone de ce dépôt."""
    if shutil.which("git") is None or not os.path.exists(os.path.join(ROOT, ".git")):
        return None
    try:
        top = _git("rev-parse", "--show-toplevel").stdout.strip()
        if not top or os.path.realpath(top) != os.path.realpath(ROOT):
            return None
        url = _git("remote", "get-url", "origin").stdout.strip()
        if REPO.lower() not in url.lower().replace(":", "/"):
            return None
        branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        dirty = _git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {"branch": branch, "clean": not dirty, "dirty": dirty}


def _latest_tag_via_git():
    try:
        p = _git("ls-remote", "--tags", "--refs", "origin", "v*", timeout=TIMEOUT * 2)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Offline(f"git ne répond pas ({e})")
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        if "permission denied" in err.lower() or "not found" in err.lower():
            raise Denied("git n'a pas accès au dépôt distant (clé SSH ?)")
        raise Offline(err.splitlines()[-1] if err else "git ls-remote a échoué")
    tags = [l.split("refs/tags/")[-1] for l in p.stdout.splitlines() if "refs/tags/" in l]
    tags = [t for t in tags if re.fullmatch(r"v?\d+(\.\d+){0,2}", t)]
    return max(tags, key=parse_version) if tags else None


# ---------------------------------------------------------------- vérification
def check():
    """Cherche la dernière release. Ne lève jamais d'exception."""
    try:
        try:
            rel = _api(f"repos/{REPO}/releases/latest")
            r = Release(rel.get("tag_name", ""), rel.get("body", ""), rel.get("name", ""))
            method = "jeton" if _token() else "gh"
        except NoAuth as e:
            if git_state() is None:
                return CheckResult("noauth", message=f"{e}.\n\n{HOW_TO_ENABLE}")
            tag = _latest_tag_via_git()
            if tag is None:
                return CheckResult("uptodate", method="git")
            r, method = Release(tag), "git"
    except Denied as e:
        return CheckResult("denied", message=f"{e}.\n\n{HOW_TO_ENABLE}")
    except Offline as e:
        return CheckResult("offline", message=str(e))
    except Exception as e:  # noqa: BLE001
        return CheckResult("error", message=f"{type(e).__name__} : {e}")
    if r.version > current_version():
        return CheckResult("available", r, method=method)
    return CheckResult("uptodate", r, method=method)


# ----------------------------------------------------------------- installation
def install_mode():
    return "git" if git_state() is not None else "archive"


LOCK = os.path.join(tempfile.gettempdir(),
                    "deepgraph_update_" + hashlib.sha1(os.path.realpath(ROOT).encode()).hexdigest()[:12] + ".lock")
LOCK_STALE = 15 * 60                   # secondes : un verrou plus vieux vient d'un processus interrompu


def _acquire_lock():
    """Verrou d'installation (hors du dossier DeepGraph, pour ne pas salir le clone). False si pris."""
    import time
    try:
        if time.time() - os.path.getmtime(LOCK) > LOCK_STALE:
            os.remove(LOCK)
    except OSError:
        pass
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    return True


def _release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def install(release, log=print):
    """Installe la release. Renvoie (succès, message). Ne lève pas d'exception."""
    if not _acquire_lock():
        return False, ("Une autre installation de DeepGraph est en cours (verrou " + LOCK + ") : rien n'a été "
                       "fait ici.")
    try:
        if install_mode() == "git":
            return _install_git(release, log)
        return _install_archive(release, log)
    except Exception as e:  # noqa: BLE001
        return False, f"Mise à jour interrompue : {type(e).__name__} : {e}"
    finally:
        _release_lock()


def _install_git(release, log):
    st = git_state()
    if st["branch"] != "main":
        return False, (f"Clone git sur la branche « {st['branch']} » : mise à jour automatique faite "
                       f"seulement sur main. Pour mettre à jour : git checkout main && git pull --ff-only "
                       f"origin main (la release {release.tag} y est).")
    if not st["clean"]:
        return False, ("Clone git avec des modifications non enregistrées (commit / stash) :\n"
                       f"{st['dirty']}\n\nRien n'a été touché. Une fois le dossier propre : "
                       "git pull --ff-only origin main.")
    log(f"git fetch --tags origin …")
    p = _git("fetch", "--tags", "origin", timeout=120)
    if p.returncode != 0:
        return False, "git fetch a échoué :\n" + (p.stderr or p.stdout).strip()
    log(f"git merge --ff-only {release.tag} …")
    p = _git("merge", "--ff-only", release.tag, timeout=60)
    if p.returncode != 0:
        return False, ("Impossible d'avancer main jusqu'à " + release.tag + " (historique divergent) :\n"
                       + (p.stderr or p.stdout).strip() + "\n\nRien n'a été modifié.")
    return True, f"DeepGraph mis à jour en {release.label} (git, main → {release.tag})."


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _install_archive(release, log):
    if _token() is None and not _have_gh():
        return False, "Téléchargement impossible : " + HOW_TO_ENABLE
    log(f"Téléchargement de {release.tag} …")
    blob = _api(f"repos/{REPO}/tarball/{release.tag}", binary=True)
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f).get("files", {})
    except (OSError, ValueError):
        manifest = {}
    backup = tempfile.mkdtemp(prefix=f"deepgraph-{__version__}-sauvegarde-")
    kept, written, new_manifest = [], 0, {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for m in members:
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            rel = os.path.normpath(parts[1])
            if rel.startswith("..") or os.path.isabs(rel) or rel.split(os.sep)[0] == ".git":
                continue                          # chemin suspect : ignoré
            data = tar.extractfile(m).read()
            new_hash = hashlib.sha256(data).hexdigest()
            new_manifest[rel] = new_hash
            dest = os.path.join(ROOT, rel)
            if os.path.exists(dest):
                cur = _sha(dest)
                if cur == new_hash:
                    continue
                # modifié localement ? (différent de ce qui avait été installé)
                modified = (manifest.get(rel) not in (None, cur)) or (rel in USER_FILES and rel not in manifest)
                if modified:
                    with open(dest + ".nouveau", "wb") as f:
                        f.write(data)
                    kept.append(rel)
                    continue
                os.makedirs(os.path.dirname(os.path.join(backup, rel)), exist_ok=True)
                shutil.copy2(dest, os.path.join(backup, rel))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".tmp-maj"
            with open(tmp, "wb") as f:
                f.write(data)
            if os.path.exists(dest):
                shutil.copymode(dest, tmp)
            os.replace(tmp, dest)
            written += 1
    # fichiers installés par l'ancienne version et absents de la nouvelle (jamais ceux de l'utilisateur)
    for rel, h in manifest.items():
        dest = os.path.join(ROOT, rel)
        if rel not in new_manifest and os.path.exists(dest) and _sha(dest) == h:
            os.makedirs(os.path.dirname(os.path.join(backup, rel)), exist_ok=True)
            shutil.move(dest, os.path.join(backup, rel))
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"version": release.label, "files": new_manifest}, f, indent=1)
    msg = f"DeepGraph mis à jour en {release.label} ({written} fichier(s) remplacé(s)).\n" \
          f"Anciens fichiers sauvegardés dans : {backup}"
    if kept:
        msg += ("\n\nModifié(s) localement, donc conservé(s) ; la nouvelle version est à côté "
                "(« .nouveau ») :\n  " + "\n  ".join(kept))
    return True, msg


# --------------------------------------------------------------------- headless
def run_terminal(ask=True, apply=False, out=None):
    """Vérification en mode texte. apply : installer sans demander. ask : dans un terminal interactif, demander ;
    sans terminal interactif (script, offscreen, tâche planifiée), installer automatiquement, sauf si
    DEEPGRAPH_NO_AUTO_UPDATE=1. Renvoie 'installed', 'uptodate', 'declined', 'failed' ou 'unchecked'."""
    import sys
    out = out or sys.stderr
    res = check()
    if res.status == "available":
        r = res.release
        print(f"[mise à jour] DeepGraph {r.label} est disponible (version actuelle : {__version__}).", file=out)
        go = apply
        if not go and ask:
            interactive = sys.stdin is not None and sys.stdin.isatty()
            if interactive:
                try:
                    go = input("Installer maintenant ? [o/N] ").strip().lower() in ("o", "oui", "y", "yes")
                except EOFError:
                    go = False
            elif not auto_install_disabled():
                print("[mise à jour] pas de terminal interactif : installation automatique "
                      "(DEEPGRAPH_NO_AUTO_UPDATE=1 pour l'éviter).", file=out)
                go = True
        if not go:
            print("[mise à jour] Pour installer : python main.py --update", file=out)
            return "declined"
        ok, msg = install(r, log=lambda s: print("[mise à jour]", s, file=out))
        print(("[mise à jour] " if ok else "[mise à jour] ÉCHEC : ") + msg, file=out)
        if ok:
            print("[mise à jour] La nouvelle version sera utilisée au prochain lancement.", file=out)
        return "installed" if ok else "failed"
    if res.status in ("noauth", "denied"):
        print("[mise à jour] AVERTISSEMENT : impossible de vérifier les mises à jour — " + res.message, file=out)
    elif res.status in ("offline", "error"):
        print(f"[mise à jour] vérification impossible ({res.message})", file=out)
    else:
        if apply:
            print(f"[mise à jour] DeepGraph est à jour ({__version__}).", file=out)
        return "uptodate"
    return "unchecked"
