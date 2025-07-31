from flask import Flask, request, jsonify
from agent_ai import VoiceAssistant
import logging
import os
import tempfile
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import sqlite3
from contextlib import contextmanager
import time
import json

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
CORS(app) 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

voice_assistant = None

def test_sqlite_connection():
    """Test rapide de la connexion SQLite"""
    try:
        db_path = "assistant_database.db"
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"SQLite non disponible: {e}")
        return False

def initialize_voice_assistant():
    """Initialise l'assistant vocal avec fallback automatique"""
    global voice_assistant
    try:
        logger.info("Initialisation de l'assistant vocal...")
        
        sqlite_available = test_sqlite_connection()
        
        if sqlite_available:
            logger.info("SQLite détecté, tentative d'initialisation normale...")
            try:
                voice_assistant = VoiceAssistant(fallback_only=False)
                if voice_assistant.db_available:
                    logger.info("✅ Assistant vocal initialisé avec SQLite")
                else:
                    logger.warning("⚠️ SQLite non fonctionnel, basculement vers fallback")
                    voice_assistant = VoiceAssistant(fallback_only=True)
            except Exception as e:
                logger.warning(f"Erreur avec SQLite: {e}, basculement vers fallback...")
                voice_assistant = VoiceAssistant(fallback_only=True)
        else:
            logger.warning("SQLite non disponible, initialisation avec fallback...")
            voice_assistant = VoiceAssistant(fallback_only=True)
        
        if hasattr(voice_assistant, 'fallback_only') and voice_assistant.fallback_only:
            logger.warning("⚠️  Assistant vocal initialisé avec base de données de secours")
        else:
            logger.info("✅ Assistant vocal initialisé avec SQLite")
            
        return True
        
    except Exception as e:
        logger.error(f"⚠️ Erreur lors de l'initialisation de l'assistant vocal: {e}")
        
        try:
            logger.info("Tentative d'initialisation en mode fallback uniquement...")
            voice_assistant = VoiceAssistant(fallback_only=True)
            logger.warning("🔄 Assistant vocal initialisé en mode fallback de secours")
            return True
        except Exception as fallback_error:
            logger.error(f"⚠️ Échec complet de l'initialisation: {fallback_error}")
            return False

def get_voice_assistant():
    """Fonction pour obtenir l'assistant vocal avec initialisation lazy"""
    global voice_assistant
    if voice_assistant is None:
        success = initialize_voice_assistant()
        if not success:
            raise RuntimeError("Impossible d'initialiser l'assistant vocal")
    return voice_assistant

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de santé amélioré avec vérification de la base de données"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError:
            return jsonify({
                "status": "error",
                "message": "Assistant vocal non initialisé"
            }), 503

        db_status = "sqlite"
        if hasattr(va, 'fallback_only') and va.fallback_only:
            db_status = "fallback"
        elif hasattr(va, 'db_available') and not va.db_available:
            db_status = "fallback"
        
        try:
            test_results = va.search_database("test", k=1)
            search_status = "ok"
        except Exception as e:
            search_status = f"error: {str(e)}"
            
        return jsonify({
            "status": "ok",
            "database": db_status,
            "search": search_status,
            "timestamp": time.time(),
            "sqlite_available": test_sqlite_connection()
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /health: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/database/status", methods=["GET"])
def database_status():
    """Endpoint pour vérifier le statut détaillé de la base de données"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError:
            return jsonify({"error": "Assistant vocal non initialisé"}), 503
            
        status = {
            "using_fallback": getattr(va, 'fallback_only', False) or not getattr(va, 'db_available', False),
            "sqlite_config": {
                "database_path": getattr(va, 'db_path', 'assistant_database.db'),
                "knowledge_base_path": getattr(va, 'knowledge_base_path', 'agent AI/database/all_docs.json')
            },
            "sqlite_available": test_sqlite_connection(),
            "connection_test": "unknown"
        }
        
        if hasattr(va, 'db_available') and va.db_available:
            try:
                cursor = va.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM products")
                product_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM product_embeddings")
                embedding_count = cursor.fetchone()[0]
                status["connection_test"] = "success"
                status["product_count"] = product_count
                status["embedding_count"] = embedding_count
            except Exception as e:
                status["connection_test"] = f"failed: {str(e)}"
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Erreur dans /database/status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/database/switch", methods=["POST"])
def switch_database():
    """Endpoint pour basculer entre SQLite et fallback"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError:
            return jsonify({"error": "Assistant vocal non initialisé"}), 503
            
        data = request.get_json()
        force_fallback = data.get("force_fallback", False)
        
        if force_fallback:
            va.fallback_only = True
            if hasattr(va, 'conn') and va.conn:
                va.conn.close()
            va.db_available = False
            logger.info("Basculement forcé vers la base de données de secours")
            
        else:
            if test_sqlite_connection():
                try:
                    va.setup_sqlite_database()
                    va.fallback_only = False
                    va.db_available = True
                    logger.info("Basculement réussi vers SQLite")
                except Exception as e:
                    logger.error(f"Impossible de basculer vers SQLite: {e}")
                    return jsonify({
                        "error": "SQLite non disponible",
                        "details": str(e)
                    }), 400
            else:
                return jsonify({
                    "error": "SQLite non disponible",
                    "details": "Connexion impossible"
                }), 400
        
        return jsonify({
            "status": "success",
            "using_fallback": va.fallback_only or not va.db_available
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /database/switch: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/database/reload", methods=["POST"])
def reload_database():
    """Endpoint pour recharger la base de données depuis all_docs.json"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError:
            return jsonify({"error": "Assistant vocal non initialisé"}), 503
            
        if not hasattr(va, 'db_available') or not va.db_available:
            return jsonify({"error": "Base de données SQLite non disponible"}), 400
            
        try:
            cursor = va.conn.cursor()
            cursor.execute("DELETE FROM product_embeddings")
            cursor.execute("DELETE FROM products")
            va.conn.commit()
            
            va.load_knowledge_base()
            
            cursor.execute("SELECT COUNT(*) FROM products")
            product_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM product_embeddings")
            embedding_count = cursor.fetchone()[0]
            
            return jsonify({
                "status": "success",
                "message": "Base de données rechargée avec succès",
                "product_count": product_count,
                "embedding_count": embedding_count
            })
            
        except Exception as e:
            logger.error(f"Erreur lors du rechargement: {e}")
            return jsonify({"error": f"Rechargement échoué: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Erreur dans /database/reload: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/database/stats", methods=["GET"])
def database_stats():
    """Endpoint pour obtenir les statistiques de la base de données"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError:
            return jsonify({"error": "Assistant vocal non initialisé"}), 503
            
        if hasattr(va, 'export_database_stats'):
            stats = va.export_database_stats()
            return jsonify(stats)
        else:
            return jsonify({"error": "Statistiques non disponibles en mode fallback"}), 400
            
    except Exception as e:
        logger.error(f"Erreur dans /database/stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Endpoint de transcription avec gestion d'erreurs améliorée"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError as e:
            logger.error(f"Assistant vocal non disponible: {e}")
            return jsonify({"error": "Assistant vocal non initialisé"}), 503
            
        if "audio" not in request.files:
            return jsonify({"error": "Aucun fichier audio fourni."}), 400

        file = request.files["audio"]
        logger.info(f"Fichier audio reçu : {file.filename}")

        if file.filename == '':
            return jsonify({"error": "Nom de fichier vide"}), 400
            
        allowed_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"Format non supporté. Formats autorisés: {', '.join(allowed_extensions)}"
            }), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            temp_path = tmp.name
            file.save(temp_path)
            logger.info(f"Fichier temporaire sauvegardé sous : {temp_path}")

        try:
            transcription = va.transcribe_audio_multilingual(temp_path)
            logger.info(f"Transcription obtenue : {transcription}")

            if not transcription or transcription.strip() == "":
                return jsonify({
                    "transcription": "",
                    "warning": "Aucun texte détecté dans l'audio"
                })

            return jsonify({
                "transcription": transcription,
                "status": "success"
            })

        finally:
            try:
                os.remove(temp_path)
                logger.info(f"Fichier temporaire supprimé : {temp_path}")
            except OSError as e:
                logger.warning(f"Impossible de supprimer le fichier temporaire: {e}")

    except Exception as e:
        logger.error(f"Erreur dans /transcribe : {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    """Endpoint de questions avec gestion améliorée de la recherche SQLite"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError:
            return jsonify({"error": "Assistant vocal non initialisé"}), 503
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données JSON requises"}), 400
            
        question = data.get("question", "").strip()

        logger.info(f"Question reçue : {question}")

        if not question:
            return jsonify({"error": "La question est vide."}), 400

        detected_language = va.detect_language_advanced(question)
        logger.info(f"Langue détectée : {detected_language}")

        is_simple, response_type = va.is_greeting_or_simple_query(question, detected_language)
        if is_simple:
            logger.info("Détection d'une salutation ou question simple.")
            simple_response = va.generate_simple_response(response_type, detected_language)
            logger.info(f"Réponse simple générée : {simple_response}")
            return jsonify({
                "answer": simple_response,
                "type": "simple",
                "language": detected_language,
                "database_used": "none"
            })

        try:
            context_passages = va.search_database(question)
            logger.info(f"Nombre de contextes trouvés : {len(context_passages)}")
            
            if hasattr(va, 'db_available') and va.db_available and not getattr(va, 'fallback_only', False):
                db_used = "sqlite"
            else:
                db_used = "fallback"
            
        except Exception as search_error:
            logger.error(f"Erreur lors de la recherche : {search_error}")
            context_passages = []
            db_used = "error"

        try:
            answer = va.generate_response(question, context_passages, detected_language)
            logger.info(f"Réponse générée : {answer[:100]}...")
            
            return jsonify({
                "answer": answer,
                "type": "contextual",
                "language": detected_language,
                "database_used": db_used,
                "context_count": len(context_passages)
            })
            
        except Exception as gen_error:
            logger.error(f"Erreur lors de la génération : {gen_error}")
            
            fallback_responses = {
                'arabic': "آسف، واجهت مشكلة في الإجابة. يمكنك إعادة المحاولة؟",
                'french': "Désolé, j'ai rencontré un problème. Pouvez-vous réessayer ?",
                'english': "Sorry, I encountered an issue. Can you try again?"
            }
            
            return jsonify({
                "answer": fallback_responses.get(detected_language, fallback_responses['french']),
                "type": "error_fallback",
                "language": detected_language,
                "database_used": db_used,
                "error": str(gen_error)
            }), 500

    except Exception as e:
        logger.error(f"Erreur dans /ask : {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    """Endpoint dédié à la recherche dans la base de données"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError:
            return jsonify({"error": "Assistant vocal non initialisé"}), 503
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données JSON requises"}), 400
            
        query = data.get("query", "").strip()
        limit = data.get("limit", 5)
        
        if not query:
            return jsonify({"error": "Requête de recherche vide"}), 400
            
        if not isinstance(limit, int) or limit < 1 or limit > 20:
            limit = 5
            
        logger.info(f"Recherche : '{query}' (limite: {limit})")
        
        results = va.search_database(query, k=limit)
        
        if hasattr(va, 'db_available') and va.db_available and not getattr(va, 'fallback_only', False):
            db_used = "sqlite"
        else:
            db_used = "fallback"
        
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results),
            "database_used": db_used
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /search : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tts", methods=["POST"])
def text_to_speech_endpoint():
    """Endpoint pour la synthèse vocale"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError:
            return jsonify({"error": "Assistant vocal non initialisé"}), 503
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données JSON requises"}), 400
            
        text = data.get("text", "").strip()
        language = data.get("language", "french")
        
        if not text:
            return jsonify({"error": "Texte vide"}), 400
            
        if language not in ['arabic', 'french', 'english']:
            language = 'french'
            
        logger.info(f"TTS demandé : '{text[:50]}...' en {language}")
        
        success = va.text_to_speech(text, language)
        
        return jsonify({
            "status": "success" if success else "error",
            "message": "Audio généré avec succès" if success else "Erreur lors de la génération audio",
            "text": text,
            "language": language
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /tts : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["GET"])
def test_endpoints():
    """Endpoint de test pour vérifier toutes les fonctionnalités"""
    try:
        try:
            va = get_voice_assistant()
        except RuntimeError:
            return jsonify({"error": "Assistant vocal non initialisé"}), 503
        
        test_results = {}
        
        test_phrases = {
            "Bonjour comment allez-vous ?": "french",
            "Hello how are you?": "english", 
            "أهلا وسهلا كيف الحال؟": "arabic"
        }
        
        language_tests = {}
        for phrase, expected_lang in test_phrases.items():
            detected = va.detect_language_advanced(phrase)
            language_tests[phrase] = {
                "expected": expected_lang,
                "detected": detected,
                "correct": detected == expected_lang
            }
        
        test_results["language_detection"] = language_tests
        
        search_test = va.search_database("écran solaire", k=2)
        test_results["search_test"] = {
            "query": "écran solaire",
            "results_count": len(search_test),
            "success": len(search_test) > 0
        }
        
        try:
            response = va.generate_response("Bonjour", [], "french")
            test_results["response_generation"] = {
                "success": True,
                "response_length": len(response)
            }
        except Exception as e:
            test_results["response_generation"] = {
                "success": False,
                "error": str(e)
            }
        
        test_results["database_status"] = {
            "sqlite_available": hasattr(va, 'db_available') and va.db_available,
            "fallback_mode": getattr(va, 'fallback_only', False)
        }
        
        return jsonify(test_results)
        
    except Exception as e:
        logger.error(f"Erreur dans /test : {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint non trouvé"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur interne du serveur: {error}")
    return jsonify({"error": "Erreur interne du serveur"}), 500

if __name__ == "__main__":
    logger.info("🚀 Démarrage du serveur Flask sur http://localhost:5000")
    logger.info(" Endpoints disponibles:")
    logger.info("  - GET  /health : État de santé du système")
    logger.info("  - GET  /database/status : Statut de la base de données")
    logger.info("  - POST /database/switch : Basculer entre SQLite et fallback")
    logger.info("  - POST /database/reload : Recharger depuis all_docs.json")
    logger.info("  - GET  /database/stats : Statistiques de la base")
    logger.info("  - POST /transcribe : Transcription audio")
    logger.info("  - POST /ask : Questions et réponses")
    logger.info("  - POST /search : Recherche dans la base")
    logger.info("  - POST /tts : Synthèse vocale")
    logger.info("  - GET  /test : Tests des fonctionnalités")
    logger.info("✅ Serveur prêt à recevoir des requêtes")
    app.run(host="0.0.0.0", port=5000, debug=True)