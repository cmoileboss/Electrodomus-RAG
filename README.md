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

Ne partage pas et ne committe pas ce fichier : il peut contenir des clés secrètes.

### Variables d'environnement

L'application charge les variables depuis `.env` à la racine du projet. Les valeurs de ce fichier remplacent les valeurs par défaut du code ; `.env.example` fournit déjà des valeurs pour plusieurs paramètres.

| Variable | Quand la renseigner | Rôle et valeur par défaut |
| --- | --- | --- |
| `CROSS_ENCODER_MODEL` | Obligatoire au démarrage de l'API | Modèle de reclassement des chunks. Utiliser `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| `GROQ_API_KEY` | Obligatoire pour générer des réponses | Clé API Groq. |
| `HF_TOKEN` | Jeton Hugging Face, nécessaire. |
| `EMBED_MODEL_ID` | Facultative | Modèle d'embedding ; défaut : `BAAI/bge-m3`. Il doit être identique à l'ingestion et à la recherche. |
| `MAX_TOKENS` | Facultative | Limite de tokens par chunk à l'ingestion ; défaut : `400`. |
| `GROQ_MODEL` | Facultative | Modèle LLM utilisé ; défaut du code : `openai/gpt-oss-20b`. `.env.example` choisit `openai/gpt-oss-120b`. |
| `GROQ_URL` | Facultative | URL de l'API Groq ; défaut : `https://api.groq.com/openai/v1/chat/completions`. |

Les paramètres PostgreSQL sont facultatifs pour le lancement local avec Docker Compose : par défaut, l'API utilise `DATABASE_HOST=localhost`, `DATABASE_PORT=5433`, `DATABASE_NAME=electrodomus`, `DATABASE_USER=postgres` et `DATABASE_PASSWORD=postgres`. Tu peux les définir dans `.env` pour utiliser une autre base. `DRIVER_NAME` vaut par défaut `postgresql+psycopg`.

Les limites de recherche peuvent aussi être ajustées : `CHUNK_LIMIT_VECTORIAL` et `CHUNK_LIMIT_BM25` (30 chacune par défaut), `CHUNK_LIMIT_RRF` (10) et `TOP_K_RERANKER` (3). Dans `.env.example`, elles sont respectivement définies à 100, 100, 100 et 20. `MISTRAL_API_KEY` n'est nécessaire que pour le script indépendant `mistral.py`, pas pour l'API FastAPI.

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
npm ci
npm run dev
```

Ouvre ensuite l'adresse indiquée par Vite, généralement <http://localhost:5173>. Le serveur de développement redirige les requêtes `/api` vers l'API locale sur le port `8000`.

Arrête le serveur frontend avec `Ctrl+C`.