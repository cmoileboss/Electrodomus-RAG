# Electrodomus RAG

## Lancer l'API en local

### Prérequis

- Python 3.13, comme dans le `Dockerfile` du projet.
- Docker Desktop, pour lancer PostgreSQL avec l'extension `pg_textsearch` requise par la recherche BM25.

### 1. Démarrer PostgreSQL

Depuis la racine du projet :

```powershell
docker compose up -d postgres
```

Le service est accessible sur `localhost:5433`. Les paramètres PostgreSQL par défaut de l'application correspondent à ceux du service Docker Compose.

### 2. Créer et activer l'environnement virtuel

Dans PowerShell :

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Sous macOS ou Linux :

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

Avec l'environnement virtuel activé :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurer l'environnement

Si le fichier `.env` n'existe pas encore, crée-le à partir du modèle :

```powershell
Copy-Item .env.example .env
```

Dans `.env`, configure au minimum `CROSS_ENCODER_MODEL` (le modèle indiqué dans `.env.example` convient) et `GROQ_API_KEY` pour pouvoir générer des réponses avec le LLM. `HF_TOKEN` est facultatif avec les modèles publics par défaut. Ne partage pas et ne committe pas ce fichier : il peut contenir des clés secrètes.

### 5. Lancer l'API

À la racine du projet, avec l'environnement virtuel activé :

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Au premier démarrage, l'application initialise la base de données et peut télécharger les modèles d'embedding et de reranking ; cela peut prendre quelques minutes.

- Vérifier que l'API répond : <http://127.0.0.1:8000/health>
- Ouvrir la documentation interactive : <http://127.0.0.1:8000/docs>

Arrête le serveur avec `Ctrl+C`. Pour arrêter PostgreSQL, exécute `docker compose stop postgres`.

## Lancer le frontend en local

Le frontend nécessite Node.js 22 et npm. L'API doit déjà être lancée sur `http://localhost:8000` pour que les appels au backend aboutissent.

Dans un second terminal, depuis la racine du projet :

```bash
cd front
npm run dev
```

Ouvre ensuite l'adresse indiquée par Vite, généralement <http://localhost:5173>. Le serveur de développement redirige les requêtes `/api` vers l'API locale sur le port `8000`.

Arrête le serveur frontend avec `Ctrl+C`.