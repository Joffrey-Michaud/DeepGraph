# Contribuer à DeepGraph

**Toute modification passe par une branche dédiée et une pull request.** On ne pousse jamais
directement sur `main` : l'administrateur du dépôt relit la pull request et décide de la fusionner
ou non. La règle vaut pour tout le monde, y compris pour un agent (LLM) qui modifie le dépôt.

## Nommage des branches

```
<type>/<sujet>
```

| type | quand |
|---|---|
| `feat` | nouvelle fonction : outil, option, bloc de bibliothèque, modèle type |
| `fix` | correction d'un bug |
| `docs` | documentation seulement |
| `refactor` | réorganisation du code, sans changement de comportement |
| `chore` | entretien : dépendances, configuration, nettoyage |

- `sujet` : 2 à 5 mots en minuscules, sans accents, séparés par des tirets, qui disent ce que fait
  la branche.
- Exemples : `feat/export-cli`, `fix/police-italique`, `docs/format-simplifie`, `feat/blocs-eeg`.
- Une branche = une modification cohérente. Si elle touche plusieurs types, prendre le type
  principal (dans l'ordre : `feat`, `fix`, `refactor`, `docs`, `chore`).

## Étapes

1. Partir de `main` à jour :
   ```bash
   git fetch origin
   git checkout -b <type>/<sujet> origin/main
   ```
2. Faire des commits courts, au présent, en français (« Ajoute… », « Corrige… »).
3. Vérifier avant de pousser :
   - l'éditeur s'ouvre (`python main.py`) et les modèles types s'ouvrent ;
   - l'export de tous les modèles types passe sans erreur :
     `python outils/exporter.py modeles/*.dgraph -f pdf -d /tmp/deepgraph_test` ;
   - si `blocks.yaml` a changé : `python outils/liste_blocs.py`, puis mettre à jour l'annexe de
     [docs/DGRAPH_POUR_LLM.md](docs/DGRAPH_POUR_LLM.md).
4. Pousser la branche et ouvrir une pull request vers `main` qui dit **quoi** et **pourquoi** :
   ```bash
   git push -u origin <type>/<sujet>
   gh pr create --base main
   ```
5. Ne pas fusionner soi-même : l'administrateur s'en charge.
