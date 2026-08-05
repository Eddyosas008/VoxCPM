# Narrer ailleurs que sur son poste

Ce guide sert à une chose : pouvoir produire un livre audio **depuis n'importe où**,
sans dépendre de la machine qu'on a sous la main. Trois routes, selon ce qu'on
cherche — la permanence, la vitesse, ou la gratuité.

Une seule commande les prépare toutes les trois, parce que la seule chose qui les
distingue est la présence d'un GPU, et le script la détecte au lieu de la demander :

```bash
curl -fsSL https://raw.githubusercontent.com/Eddyosas008/VoxCPM/claude/repo-analysis-improvement-dg0ies/scripts/cloud_setup.sh | bash
```

Il installe les paquets système, clone le dépôt, choisit la roue PyTorch adaptée
(CUDA ou CPU — les roues CPU sont bien plus légères), installe le projet et
télécharge le modèle. Relancé, il ne refait rien : chaque étape vérifie avant
d'agir.

## Ce que pèse l'installation

| | |
|---|---|
| Modèle VoxCPM2 (cache Hugging Face) | ~4,6 Go |
| Environnement Python avec PyTorch | ~1,6 Go |
| **Total** | **~6,2 Go** |

Autrement dit : l'espace disque n'est jamais le facteur limitant. **La RAM l'est.**
Le modèle demande environ **8,7 Go résidents en float32**, et c'est au chargement
des poids qu'il meurt quand ils manquent. En dessous de 12 Go de RAM, il faut
`VOXCPM_CPU_DTYPE=bfloat16` : empreinte divisée par deux (~4,4 Go), un peu plus
lent parce que le bfloat16 est émulé sur processeur.

## Route 1 — un VPS, pour la permanence

**Ce que ça apporte** : la machine tourne en continu. On lance une narration, on
ferme son portable, on récupère les chapitres deux jours plus tard. La chaîne
étant reprenable au segment près, une coupure ne coûte que le segment en cours.

**Ce que ça n'apporte pas** : de la vitesse. Un VPS d'entrée de gamme a 2 cœurs,
c'est-à-dire moins qu'un portable courant.

Exemple mesuré sur un Hostinger **KVM 2** — 2 cœurs, 8 Go de RAM, 100 Go de
disque, Ubuntu 24.04 : le disque est confortable, la RAM impose le bfloat16, et
les 2 cœurs rendent la narration environ **2 à 3 fois plus lente** qu'un portable
à 4 cœurs. Un livre de 3 heures y demande de l'ordre de **5 jours** — acceptable
seulement parce que personne n'attend devant.

Monter en gamme change la donne : 8 cœurs et 32 Go permettent le float32 et
divisent le temps par quatre. À comparer honnêtement au coût d'un GPU loué à
l'heure, qui fait le même livre en moins d'une heure.

## Route 2 — un GPU loué à l'heure, pour la vitesse

C'est la seule option qui change l'ordre de grandeur : **RTF ~0,3 contre ~18 à 40
sur processeur**, soit un livre de 3 heures en moins d'une heure de calcul.

Chez [RunPod](https://www.runpod.io/pricing), une RTX 4090 est à environ
**0,34 $/h** en Community Cloud, facturée à la seconde. Un livre entier coûte donc
moins qu'un café. Aucun engagement : on crée l'instance, on lance le script
ci-dessus, on narre, on rapatrie, on détruit.

```bash
# sur la machine louée
bash scripts/cloud_setup.sh
nohup ./.venv/bin/python scripts/narrate_book.py livre.epub \
    --voice "Narrateur profond & calme" --device cuda \
    --assemble m4b --export-acx > narration.log 2>&1 &

# depuis chez soi, quand c'est fini
rsync -avz root@<ip>:~/voxcpm/output/book_<nom>/ ./book_<nom>/
```

**Détruire l'instance en partant.** Elle est facturée tant qu'elle existe, même
inactive.

## Route 3 — Kaggle, pour ne rien payer

[Kaggle](https://www.kaggle.com/product-feedback/173129) donne **une trentaine
d'heures de GPU par semaine** (P100, ou deux T4), en sessions de **12 heures
maximum**. C'est gratuit, c'est un vrai GPU, et la limite de session n'est pas
bloquante ici : le cache par segment fait qu'une session reprend là où la
précédente s'est arrêtée. Un livre de 3 heures tient en une ou deux sessions.

La contrainte est le disque éphémère : il faut écrire les chapitres dans les
*outputs* du notebook, ou les pousser ailleurs avant la fin de session.

## Atteindre l'interface à distance, sans l'offrir à tout le monde

`app.py` écoute par défaut sur `0.0.0.0`, c'est-à-dire sur toutes les interfaces.
Sur une machine distante, **cela met le modèle à la disposition de quiconque
trouve le port** — et sur un GPU loué, c'est votre facture qui synthétise pour un
inconnu. L'application le signale désormais au démarrage.

Deux façons correctes :

**Le tunnel SSH** — rien n'est exposé, c'est la plus sûre :

```bash
# sur le serveur
./.venv/bin/python app.py --host 127.0.0.1 --port 8808 --device cuda --no-denoiser

# sur votre poste
ssh -N -L 8808:127.0.0.1:8808 root@<ip>
# puis http://127.0.0.1:8808
```

**Un mot de passe**, si l'accès direct est nécessaire :

```bash
VOXCPM_AUTH='edwin:motdepasse' ./.venv/bin/python app.py \
    --host 0.0.0.0 --port 8808 --device cuda --no-denoiser
```

Le mot de passe passe par la variable d'environnement plutôt que par
`--auth` en ligne de commande, pour qu'il n'atterrisse ni dans l'historique du
shell ni dans la liste des processus.

## Laisser tourner sans surveillance

Sur un VPS, une narration dure des jours : elle doit survivre à la fermeture de la
session SSH.

```bash
nohup ./.venv/bin/python scripts/narrate_book.py livre.epub \
    --voice "..." --outdir output/book_mon_livre > narration.log 2>&1 &
tail -f narration.log
```

Pour l'interface, qui elle doit repartir après un redémarrage du serveur, un
service systemd :

```ini
# /etc/systemd/system/voxcpm.service
[Unit]
Description=VoxCPM narration
After=network.target

[Service]
User=root
WorkingDirectory=/root/voxcpm
Environment=VOXCPM_AUTH=edwin:motdepasse
Environment=VOXCPM_CPU_DTYPE=bfloat16
ExecStart=/root/voxcpm/.venv/bin/python app.py --host 127.0.0.1 --port 8808 --device cpu --no-denoiser
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now voxcpm
journalctl -u voxcpm -f
```

## Couper la chaîne en deux

Le paquet `narration/` ne dépend ni de `torch` ni de `gradio` — c'est délibéré.
Tout ce qui n'est pas la synthèse tourne donc sur n'importe quelle petite machine,
en quelques secondes :

- contrôle qualité et listage des défauts (`repair_segment.py --list`)
- réparation d'un segment (celle-ci a besoin du modèle)
- assemblage M4B avec marqueurs et couverture
- contrôle de conformité et export de dépôt (`export_acx.py`)

L'architecture qui en découle : **le GPU loué ne fait que synthétiser**, quelques
dizaines de minutes, et tout le reste vit sur le VPS ou sur le poste local. C'est
ce qui rend la location à l'heure économique.

## Récapitulatif

| | Vitesse | Coût | Pour quoi |
|---|---|---|---|
| **VPS 2 cœurs** | ~3× plus lent qu'un portable | déjà payé | Permanence, stockage, tout le hors-synthèse |
| **VPS 8 cœurs** | ~4× un portable | abonnement mensuel | Narration sans surveillance, sans louer |
| **GPU à l'heure** | **~60× un portable** | ~0,34 $/h | Un livre entier en moins d'une heure |
| **Kaggle** | GPU, sessions de 12 h | gratuit | Essais, et livres entiers avec un peu de patience |
