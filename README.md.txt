Assistant Vocal Multilingue pour Pharmacie Paralabel
Ce projet implémente un assistant vocal intelligent et multilingue conçu pour la pharmacie Paralabel. Il permet aux utilisateurs de poser des questions sur les produits, les prix et d'obtenir des conseils de santé en utilisant leur voix, en arabe, français ou anglais.

Fonctionnalités
Reconnaissance Vocale (STT) : Convertit la parole de l'utilisateur en texte grâce au modèle Whisper (modèle medium pour un bon équilibre entre précision et performance).

Synthèse Vocale (TTS) : Génère des réponses vocales naturelles en utilisant Edge TTS, avec un support robuste pour l'arabe, le français et l'anglais, y compris des voix de secours pour l'arabe.

Compréhension du Langage Naturel (NLU) : Utilise l'API Google Gemini 1.5 Flash pour comprendre les requêtes complexes et générer des réponses pertinentes.

Recherche Sémantique de Produits : Recherche les produits pertinents dans une base de connaissances SQLite en utilisant des embeddings de phrases (Sentence Transformers) pour une meilleure compréhension contextuelle.

Recherche par Mots-clés (Fallback) : En cas d'échec de la recherche sémantique ou d'indisponibilité de la base de données, une recherche par mots-clés est effectuée sur les données brutes.

Gestion Intelligente des Requêtes : Distingue automatiquement les salutations, les remerciements et les questions générales des requêtes spécifiques aux produits pour fournir des réponses appropriées.

Base de Données Locale : Stocke les informations sur les produits et leurs embeddings dans une base de données SQLite pour des recherches rapides et efficaces.

Traitement Audio : Gère la conversion de formats audio (MP3 vers WAV) pour la compatibilité avec Whisper.

Technologies Utilisées
Python 3.10

Whisper : Pour la reconnaissance vocale (Speech-to-Text).

Edge TTS : Pour la synthèse vocale (Text-to-Speech).

Google Gemini API : Pour la génération de réponses et la compréhension du langage.

Sentence Transformers : Pour la génération d'embeddings de texte et la recherche de similarité.

SQLite3 : Base de données relationnelle légère pour le stockage des produits et de leurs embeddings.

pydub : Bibliothèque Python pour la manipulation audio.

numpy : Pour les opérations numériques, notamment avec les embeddings.

librosa : Pour le chargement et le traitement des fichiers audio.

python-dotenv : Pour la gestion des variables d'environnement (clés API).

re : Module pour les expressions régulières.

gtts (Google Text-to-Speech) : Utilisé dans les exemples pour générer des fichiers audio de test.

pygame : Peut être utilisé pour la lecture audio (non directement utilisé pour la lecture dans le script fourni, mais souvent utilisé dans de tels projets).

asyncio : Pour la gestion des opérations asynchrones (notamment avec Edge TTS).

tempfile, os, pickle : Pour la gestion des fichiers temporaires, des chemins et la sérialisation des embeddings.