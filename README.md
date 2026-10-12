# Application pour déployer un RAG avec une interface graphique Chainlit

### Installation 

Installer les paquets Python
```bash
pip install -r requiremtents.txt
```

### Ingestion des documents - Base documentaire (RAG)
Initialiser une base de donnée Chroma

Lancer l'ingestion 
```bash
python3 -m rag.ingest data/documents
```

### Lancement de l'application 

Executer
```bash
chainlit run app.py -w --port 32321
```


