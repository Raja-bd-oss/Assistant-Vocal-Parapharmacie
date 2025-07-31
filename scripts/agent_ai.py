import sys
import json
import sqlite3
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import threading
import time
import subprocess
import google.generativeai as genai
from pydub import AudioSegment
from pydub.effects import normalize
import noisereduce as nr
import librosa
from langdetect import detect
import re
import gtts
from gtts import gTTS
import pygame
import edge_tts
import asyncio
import tempfile
import os
import pickle
from typing import List, Tuple, Optional

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class VoiceAssistant:
    def __init__(self, fallback_only=False):
        """
        Initialisation avec SQLite et support pour all_docs.json
        """
        try:
            self.fallback_only = fallback_only
            self.db_available = False
            self.db_path = "assistant_database.db"
            self.knowledge_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "all_docs.json")

            
            self.setup_models()
            if not fallback_only:
                try:
                    self.setup_sqlite_database()
                    self.load_knowledge_base()
                    self.db_available = True
                    logger.info("✅ Base de données SQLite initialisée")
                except Exception as e:
                    logger.warning(f"⚠️ Base de données non disponible, mode fallback activé: {e}")
                    self.db_available = False
            else:
                logger.info("🔄 Mode fallback uniquement activé")
            
            self.setup_language_patterns()
            self.setup_tts()
            
            logger.info("✅ Assistant vocal initialisé avec succès")
            
        except Exception as e:
            logger.error(f"⚠️ Erreur lors de l'initialisation: {e}")
            raise

    def setup_models(self):
        """Initialisation des modèles IA"""
        try:
            logger.info("Chargement du modèle Whisper...")
            self.whisper_model = whisper.load_model("medium")
            
            logger.info("Chargement du modèle d'embedding...")
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            
            logger.info("Configuration du modèle Gemini...")
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            
            logger.info("Modèles chargés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            raise

    def setup_sqlite_database(self):
        """Configuration de la base de données SQLite"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute('PRAGMA foreign_keys = ON')
            
            self.create_tables()
            
            logger.info("Base de données SQLite configurée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la configuration SQLite: {e}")
            raise

    def create_tables(self):
        """Créer les tables nécessaires"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                brand TEXT,
                category TEXT,
                price REAL,
                description TEXT,
                usage_instructions TEXT,
                ingredients TEXT,
                size TEXT,
                stock_quantity INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS product_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                text_content TEXT,
                embedding BLOB,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_product_name ON products(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_product_category ON products(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_embedding_product_id ON product_embeddings(product_id)')
        
        self.conn.commit()
        logger.info("Tables créées avec succès")

    def load_knowledge_base(self):
        """Charger all_docs.json dans SQLite si la base est vide"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM products")
            count = cursor.fetchone()[0]

            if count > 0:
                logger.info(f" La base SQLite contient déjà {count} produits. Chargement ignoré.")
                return

            if not os.path.exists(self.knowledge_base_path):
                logger.warning(f"⚠️ Fichier {self.knowledge_base_path} non trouvé.")
                return

            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)

            logger.info(f" Chargement de {len(docs_data)} produits depuis {self.knowledge_base_path}...")

            for i, doc in enumerate(docs_data):
                try:
                    product_data = self.parse_document(doc)
                    if product_data:
                        product_id = self.insert_product(product_data)

                        text_content = self.create_searchable_text(product_data)
                        embedding = self.model.encode([text_content])[0]
                        self.insert_embedding(product_id, text_content, embedding)

                        if (i + 1) % 10 == 0:
                            logger.info(f"✔️ {i + 1}/{len(docs_data)} produits traités")

                except Exception as e:
                    logger.error(f"⚠️ Erreur document {i} : {e}")
                    continue

            self.conn.commit()
            logger.info("✅ Base de connaissances importée avec succès dans SQLite.")

        except Exception as e:
            logger.error(f"🚨 Erreur load_knowledge_base : {e}")
            raise

    def parse_document(self, doc):
        """Parser un document JSON selon la structure Paralabel"""
        try:
            if isinstance(doc, dict):
                title = doc.get('title', 'Produit inconnu')
                brand = self.extract_brand_from_title(title)
                
                category = self.extract_category_from_url(doc.get('url', ''))
                
                return {
                    'name': title,
                    'brand': brand,
                    'category': category,
                    'price': self.extract_price(doc.get('price', '0')),
                    'description': doc.get('long_description', doc.get('short_description', '')),
                    'usage_instructions': '',  
                    'ingredients': '',  
                    'size': self.extract_size_from_title(title),
                    'stock_quantity': 10, 
                    'url': doc.get('url', '')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors du parsing du document: {e}")
            return None
        
    def extract_brand_from_title(self, title):
        """Extraire la marque du titre"""
        if not title:
            return "Paralabel"
        
        brands = [
            'FILORGA', 'LA ROCHE POSAY', 'VICHY', 'BIODERMA', 'AVENE', 'EUCERIN',
            'CERAVE', 'NIVEA', 'GARNIER', 'LOREAL', 'MAYBELLINE', 'REVLON',
            'NEUTROGENA', 'CETAPHIL', 'SEBAMED', 'DUCRAY', 'KLORANE', 'LIERAC',
            'CAUDALIE', 'NUXE', 'ROCHE POSAY', 'URIAGE', 'ISDIN', 'NOREVA'
        ]
        
        title_upper = title.upper()
        for brand in brands:
            if brand in title_upper:
                return brand.title()
        
        first_word = title.split()[0] if title.split() else "Paralabel"
        return first_word
    
    def extract_category_from_url(self, url):
        """Extraire la catégorie de l'URL"""
        if not url:
            return "Soins"
    
        category_mapping = {
        'soin-regard': 'Soins du regard',
        'soin-visage': 'Soins visage',
        'anti-age': 'Soins anti-âge',
        'hydratant': 'Soins hydratants',
        'nettoyant': 'Nettoyants',
        'protection-solaire': 'Protection solaire',
        'ecran-solaire': 'Protection solaire',
        'shampoing': 'Soins capillaires',
        'complement': 'Compléments alimentaires',
        'vitamine': 'Compléments alimentaires',
        'maquillage': 'Maquillage',
        'parfum': 'Parfumerie',
        'bebe': 'Produits bébé',
        'homme': 'Soins homme'
        }
    
        url_lower = url.lower()
        for keyword, category in category_mapping.items():
            if keyword in url_lower:
               return category
    
        return "Soins et beauté"

    def extract_size_from_title(self, title):
        """Extraire la taille du titre"""
        if not title:
            return ""
        
        import re
        size_patterns = [
            r'(\d+(?:\.\d+)?)\s*ml',
            r'(\d+(?:\.\d+)?)\s*g',
            r'(\d+(?:\.\d+)?)\s*mg',
            r'(\d+)\s*comprimés?',
            r'(\d+)\s*gélules?',
            r'(\d+)\s*sachets?'
        ]
        
        title_lower = title.lower()
        for pattern in size_patterns:
            match = re.search(pattern, title_lower)
            if match:
                return match.group(0)
        
        return ""

    def extract_price(self, price_str):
        """Extraire le prix numérique d'une chaîne"""
        try:
            if isinstance(price_str, (int, float)):
                return float(price_str)
            
            import re
            price_clean = re.sub(r'[^\d\.,]', '', str(price_str))
            price_clean = price_clean.replace(',', '.')
            
            if price_clean:
                return float(price_clean)
            
            return 0.0
        except:
            return 0.0

    def create_searchable_text(self, product_data):
        """Créer le texte de recherche pour un produit"""
        parts = []
        
        if product_data.get('name'):
            parts.append(f"Nom: {product_data['name']}")
        if product_data.get('brand'):
            parts.append(f"Marque: {product_data['brand']}")
        if product_data.get('category'):
            parts.append(f"Catégorie: {product_data['category']}")
        if product_data.get('description'):
            parts.append(f"Description: {product_data['description']}")
        if product_data.get('usage_instructions'):
            parts.append(f"Utilisation: {product_data['usage_instructions']}")
        if product_data.get('ingredients'):
            parts.append(f"Ingrédients: {product_data['ingredients']}")
        
        return " | ".join(parts)

    def insert_product(self, product_data):
        """Insérer un produit dans la base"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO products (name, brand, category, price, description, 
                                usage_instructions, ingredients, size, stock_quantity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            product_data['name'],
            product_data['brand'],
            product_data['category'],
            product_data['price'],
            product_data['description'],
            product_data['usage_instructions'],
            product_data['ingredients'],
            product_data['size'],
            product_data['stock_quantity']
        ))
        
        return cursor.lastrowid

    def insert_embedding(self, product_id, text_content, embedding):
        """Insérer un embedding dans la base"""
        cursor = self.conn.cursor()
        
        embedding_blob = pickle.dumps(embedding)
        
        cursor.execute('''
            INSERT INTO product_embeddings (product_id, text_content, embedding)
            VALUES (?, ?, ?)
        ''', (product_id, text_content, embedding_blob))

    def search_database_sqlite(self, query, k=5):
        """Recherche dans SQLite avec similarité cosinus"""
        logger.info(f"Recherche SQLite pour: {query}")
        
        try:
            query_embedding = self.model.encode([query])[0]
            
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT pe.product_id, pe.text_content, pe.embedding, 
                       p.name, p.brand, p.category, p.price, p.description,
                       p.usage_instructions, p.ingredients, p.size, p.stock_quantity
                FROM product_embeddings pe
                JOIN products p ON pe.product_id = p.id
            ''')
            
            results = cursor.fetchall()
            similarities = []
            
            for row in results:
                try:
                    stored_embedding = pickle.loads(row[2])
                    
                    similarity = self.cosine_similarity(query_embedding, stored_embedding)
                    
                    similarities.append((similarity, row))
                    
                except Exception as e:
                    logger.error(f"Erreur lors du calcul de similarité: {e}")
                    continue
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            passages = []
            for similarity, row in similarities[:k]:
                product_id, text_content, _, name, brand, category, price, description, usage_instructions, ingredients, size, stock_quantity = row
                
                passage = f"""
Produit: {name}
Marque: {brand}
Catégorie: {category}
Prix: {price} DT
Description: {description}
Utilisation: {usage_instructions}
Ingrédients: {ingredients}
Taille: {size}
Stock: {"Disponible" if stock_quantity > 0 else "Rupture de stock"}
Similarité: {similarity:.4f}
                """.strip()
                
                passages.append(passage)
            
            logger.info(f"Trouvé {len(passages)} produits similaires")
            return passages
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche SQLite: {e}")
            return []

    def cosine_similarity(self, a, b):
        """Calculer la similarité cosinus entre deux vecteurs"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0
            
            return dot_product / (norm_a * norm_b)
        except:
            return 0


    def search_database_fallback(self, query, k=5):
        """
        Recherche de fallback utilisant les données réelles de all_docs.json
        """
        logger.info(f" Recherche fallback pour: {query}")
        
        try:
            json_path = os.path.join("agent AI", "database", "all_docs.json")
            if not os.path.exists(json_path):
                # Essayer d'autres chemins possibles
                possible_paths = [
                    "all_docs.json",
                    "database/all_docs.json",
                    "agent_ai/database/all_docs.json",
                    os.path.join(os.path.dirname(__file__), "database", "all_docs.json")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        json_path = path
                        break
                else:
                    logger.error("Fichier all_docs.json introuvable")
                    return self._get_no_product_message(query)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                all_docs = json.load(f)
            
            logger.info(f"📁 Chargé {len(all_docs)} produits depuis all_docs.json")
            
        except Exception as e:
            logger.error(f"Erreur chargement all_docs.json: {e}")
            return self._get_no_product_message(query)
        
        query_lower = query.lower()
        relevant_products = []
        
        for doc in all_docs:
            try:
                score = 0
                title = doc.get('title', '').lower()
                description = doc.get('long_description', doc.get('short_description', '')).lower()
                url = doc.get('url', '').lower()
                
                combined_text = f"{title} {description} {url}"
                
                query_words = [word for word in query_lower.split() if len(word) > 2]
                
                for word in query_words:
                    if word in title:
                        score += 20
                    if word in description:
                        score += 10
                    if word in url:
                        score += 5
                
                if any(keyword in query_lower for keyword in ['ride', 'rides', 'anti-age', 'anti-aging']):
                    if any(keyword in combined_text for keyword in ['anti-age', 'ride', 'fermeté', 'lifting', 'ncef']):
                        score += 50
                
                if any(keyword in query_lower for keyword in ['regard', 'yeux', 'eye', 'contour']):
                    if any(keyword in combined_text for keyword in ['regard', 'eyes', 'contour', 'yeux']):
                        score += 50
                
                if any(keyword in query_lower for keyword in ['hydratant', 'hydration', 'sec', 'sèche']):
                    if any(keyword in combined_text for keyword in ['hydrat', 'moistur', 'sec']):
                        score += 30
                
                if score > 0:
                    relevant_products.append((doc, score))
                    
            except Exception as e:
                logger.error(f"Erreur lors du scoring du produit: {e}")
                continue
        
        if not relevant_products:
            return self._get_no_product_message(query)
        
        relevant_products.sort(key=lambda x: x[1], reverse=True)
        
        passages = []
        for doc, score in relevant_products[:k]:
            try:
                brand = self.extract_brand_from_title(doc.get('title', ''))
                category = self.extract_category_from_url(doc.get('url', ''))
                size = self.extract_size_from_title(doc.get('title', ''))
                
                passage = f"""
    Produit disponible chez Paralabel:

    Nom: {doc.get('title', 'Produit inconnu')}
    Marque: {brand}
    Catégorie: {category}
    Prix: {doc.get('price', 'Prix non disponible')}
    Description: {doc.get('long_description', doc.get('short_description', 'Description non disponible'))}
    Conditionnement: {size if size else 'Non spécifié'}
    Disponibilité: En stock chez Paralabel

    Lien produit: {doc.get('url', '')}
    Score de pertinence: {score}
                """.strip()
                
                passages.append(passage)
                
            except Exception as e:
                logger.error(f"Erreur formatage produit: {e}")
                continue
        
        logger.info(f"✅ Recherche fallback: {len(passages)} produits Paralabel trouvés")
        return passages

    def _get_no_product_message(self, query):
        """Message quand aucun produit n'est trouvé"""
        return [f"""
    Je suis désolé, mais je ne trouve pas de produit correspondant à "{query}" dans notre stock actuel chez Paralabel.

    Je vous recommande de :
    • Visiter notre site web www.paralabel.tn pour voir tous nos produits
    • Contacter notre pharmacie directement pour vérifier la disponibilité
    • Consulter notre pharmacien pour des conseils personnalisés

    Notre équipe sera ravie de vous aider à trouver le produit qui convient le mieux à vos besoins.
        """.strip()]
    def search_database(self, query, k=5):
        """Point d'entrée principal pour la recherche"""
        if self.db_available:
            try:
                results = self.search_database_sqlite(query, k)
                if results and len(results) > 0:
                    return results
                else:
                    logger.warning("Recherche SQLite sans résultats, passage au fallback")
                    return self.search_database_fallback(query, k)
            except Exception as e:
                logger.error(f"Erreur SQLite: {e}, passage au fallback")
                return self.search_database_fallback(query, k)
        else:
            return self.search_database_fallback(query, k)

    def setup_tts(self):
        """Initialisation du système Text-to-Speech"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            logger.info("✅ Système TTS (pygame) initialisé")
        except Exception as e:
            logger.error(f"⚠️ Erreur initialisation TTS (pygame): {e}")
            logger.warning("⚠️ TTS non disponible - Mode silencieux activé")

        try:
            import edge_tts
            self.edge_tts_available = True
            logger.info("✅ Module edge-tts détecté - Voix arabe activée")
        except ImportError:
            self.edge_tts_available = False
            logger.warning("⚠️ edge-tts non disponible - Lecture vocale arabe désactivée. Installer avec: pip install edge-tts")


    def setup_language_patterns(self):
        """Configuration des patterns de langue améliorés"""
        self.language_patterns = {
            'arabic': {
                'keywords': [
                    r'(أهلا|اهلا|مرحبا|السلام|عسلامة|يزيك|يا|كيف|شلونك|احوالك|لاباس|شنحوالك)',
                    r'(صباح|مساء|ليلة|نهار)(\s)?(الخير|طيبة|زين)',
                    r'(بونجور|بونسوار|سالي|هاي)',
                    r'(شنو|ايش|شيا|وينو|كيفاش|علاش|وقتاش|قداش|بقداش|بكداش)',
                    r'(تنجم|قادر|ممكن|يمكن)(\s)?(تعاونني|تساعدني|تقولي)',
                    r'(انا|اني)(\s)?(نحب|نحتاج|نبغي|نريد)',
                    r'(قولي|فهمني|وضحلي|شرحلي)',
                    r'(شكرا|مرسي|يعطيك|الله|برشا)',
                    r'(كريم|كرام|كريمة|واقي|شمس|حماية)',
                    r'(دوا|دواء|علاج)',
                    r'(شامبو|شامبوان|شمبوان)',
                    r'(صابون|سافون)',
                    r'(عطر|برفان|برفوم)',
                    r'(دودوران|ديودوران)',
                    r'(فيتامين|فيتامينة)',
                    r'(السعر|الثمن|الكلفة|بقداش)',
                    r'(برشا|ياسر|فما|ماكان|زادة|كيما|هكا|هكة)',
                    r'(عندكم|فيه|موجود|متاع)',
                    r'(بشرة|وجه|شعر|مختلط|دهني|جاف)',
                ],
                'tts_lang': 'ar',
                'greeting_responses': [
                    "أهلا وسهلا! كيف يمكنني مساعدتك في صيدلية باراليبل؟",
                    "مرحبا! لاباس؟ تنجم تسألني على أي منتج أو دواء تحتاجه.",
                    "يا هلا! كيفاش نجمت نساعدك اليوم؟",
                    "أهلا بيك في صيدلية باراليبل! شنحوالك؟"
                ],
                'thanks_responses': [
                    "العفو! أي وقت تحتاج مساعدة أنا هنا.",
                    "لا شكر على واجب! عندك حاجة أخرى؟",
                    "تسلم! إذا احتجت شيء آخر، أنا في الخدمة."
                ],
                'general_responses': [
                    "أنا مساعدك الافتراضي في صيدلية باراليبل. تنجم تسألني على الأدوية، المنتجات التجميلية، الأسعار ونصائح صحية.",
                    "نجمت نساعدك في معلومات على الأدوية، منتجات العناية، التجميل والأسعار."
                ]
            },
            
            'french': {
                'keywords': [
                    r'(salut|bonjour|bonsoir|hello|salaam|coucou|bonne\s+journée)',
                    r'(comment|ca|ça)(\s)?(va|allez|vous\s+portez)',
                    r'(bonne|bon)(\s)?(journée|matinée|soirée|nuit)',
                    r'(que|quoi|comment|pourquoi|quand|où|combien|quel|quelle)',
                    r'(pouvez|peux|peut|pourriez)(\s)?(vous|tu)(\s)?(m\'|me|nous)',
                    r'(j\'ai|je\s+veux|je\s+cherche|je\s+voudrais|j\'aimerais)',
                    r'(dites|expliquez|montrez|aidez|proposez|recommandez)',
                    r'(merci|thanks|remercie|je\s+vous\s+remercie)',
                    r'(crème|gel|shampoing|savon|parfum|déodorant|médicament|écran|solaire)',
                    r'(vitamine|complément|paracétamol|aspirine|sirop|protection|spf)',
                    r'(prix|coût|combien|euros?|dinars?)',
                    r'(tube|flacon|boîte|ml|grammes?)',
                    r'(est-ce\s+que|qu\'est-ce\s+que|avez-vous|y\s+a-t-il)',
                    r'(s\'il\s+vous\s+plaît|svp|merci\s+beaucoup)',
                    r'(peau|visage|cheveux|mixte|grasse|sèche|sensible)',
                ],
                'tts_lang': 'fr',
                'greeting_responses': [
                    "Bonjour ! Comment puis-je vous aider à la pharmacie Paralabel aujourd'hui ?",
                    "Salut ! Je suis là pour répondre à vos questions sur nos médicaments et produits.",
                    "Bonsoir ! Que puis-je faire pour vous ?",
                ],
                'thanks_responses': [
                    "De rien ! N'hésitez pas si vous avez d'autres questions.",
                    "Je vous en prie ! Je reste à votre disposition.",
                    "Avec plaisir ! Y a-t-il autre chose ?"
                ],
                'general_responses': [
                    "Je suis votre assistant virtuel Paralabel. Je peux vous renseigner sur les médicaments, produits cosmétiques, prix et conseils santé.",
                    "Je suis là pour vous aider avec tous vos besoins en pharmacie et parapharmacie."
                ]
            },
            
            'english': {
                'keywords': [
                    r'(hello|hi|hey|good\s+morning|good\s+evening|good\s+afternoon)',
                    r'(how\s+are\s+you|how\s+do\s+you\s+do|what\'s\s+up|how\'s\s+it\s+going)',
                    r'(nice\s+to\s+meet|pleased\s+to\s+meet|have\s+a\s+good\s+day)',
                    r'(what|how|why|when|where|which|who|whose)',
                    r'(can\s+you|could\s+you|would\s+you|will\s+you|do\s+you)',
                    r'(i\s+want|i\s+need|i\'m\s+looking|i\s+would\s+like|i\'d\s+like)',
                    r'(tell\s+me|show\s+me|explain|help|assist|recommend|suggest)',
                    r'(thank\s+you|thanks|appreciate|grateful)',
                    r'(cream|gel|shampoo|soap|perfume|deodorant|medicine|sunscreen)',
                    r'(vitamin|supplement|paracetamol|aspirin|syrup|protection|spf)',
                    r'(price|cost|how\s+much|dollars?|euros?)',
                    r'(tube|bottle|box|ml|grams?)',
                    r'(do\s+you\s+have|is\s+there|are\s+there|please|excuse\s+me)',
                    r'(skin|face|hair|combination|oily|dry|sensitive)',
                ],
                'tts_lang': 'en',
                'greeting_responses': [
                    "Hello! How can I help you at Paralabel pharmacy today?",
                    "Hi there! I'm here to assist with your medication and product questions.",
                    "Good day! What can I do for you?",
                    "Hello and welcome to Paralabel! How are you doing?"
                ],
                'thanks_responses': [
                    "You're welcome! Feel free to ask if you have more questions.",
                    "My pleasure! I'm here if you need anything else.",
                    "Glad to help! Is there anything else you'd like to know?"
                ],
                'general_responses': [
                    "I'm your Paralabel virtual assistant. I can help with medications, cosmetic products, prices and health advice.",
                    "I'm here to assist with all your pharmacy and parapharmacy needs."
                ]
            }
        }

    def detect_language_advanced(self, text):
        """Détection de langue améliorée - VERSION CORRIGÉE"""
        if not text or len(text.strip()) == 0:
            return 'french'

        text_clean = text.lower().strip()
        
        # 🔹 Détection des caractères arabes
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        if arabic_chars > 2:
            logger.info(f"Caractères arabes détectés: {arabic_chars}")
            return 'arabic'

        # 🔹 Score par langue basé sur mots-clés
        language_scores = {'arabic': 0, 'french': 0, 'english': 0}
        
        # Mots-clés spécifiques aux produits et questions
        enhanced_patterns = {
            'arabic': [
                r'(عندكم|فيه|موجود|تنجموا|وينو|كيفاش|شنو|ايش)',
                r'(كريم|شامبو|دوا|دواء|واقي|شمس|عطر|فيتامين|سيروم)',
                r'(بقداش|بكداش|السعر|الثمن|دينار)',
                r'(بشرة|وجه|شعر|دهني|جاف|مختلط)',
                r'(اقترحولي|انصحوني|قولولي)',
                r'(أهلا|مرحبا|السلام|شكرا|مرسي)'
            ],
            'french': [
                r'(avez-vous|proposez|recommandez|conseillez|suggérez)',
                r'(crème|écran|solaire|shampoing|médicament|parfum|vitamine|sérum)',
                r'(combien|prix|coût|euros?|dinars?)',
                r'(peau|visage|cheveux|grasse|sèche|mixte|sensible)',
                r'(propositions|suggestions|conseils|produits)',
                r'(bonjour|salut|bonsoir|merci|merci\s+beaucoup)',
                r'(pour\s+peau|anti-âge|protection|hydratant)',
                r'(quelles?|quel|que|comment|où|pourquoi)'
            ],
            'english': [
                r'(do\s+you\s+have|recommend|suggest|propose)',
                r'(cream|sunscreen|shampoo|medicine|perfume|vitamin|serum)',
                r'(how\s+much|price|cost|dollars?)',
                r'(skin|face|hair|oily|dry|combination|sensitive)',
                r'(products|suggestions|recommendations)',
                r'(hello|hi|good\s+morning|thank\s+you)',
                r'(for\s+skin|anti-aging|protection|moisturizing)',
                r'(what|how|where|why|which)'
            ]
        }
        
        for lang, patterns in enhanced_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_clean, re.IGNORECASE))
                if matches > 0:
                    language_scores[lang] += matches * 3

        # 🔹 Détection automatique avec langdetect
        try:
            from langdetect import detect
            detected = detect(text_clean)
            if detected == 'fr':
                language_scores['french'] += 3
            elif detected == 'en':
                language_scores['english'] += 3
            elif detected == 'ar':
                language_scores['arabic'] += 3
        except:
            pass
        
        # 🔹 Bonus pour certains patterns caractéristiques
        if re.search(r'(qu\'est-ce|est-ce\s+que)', text_clean, re.IGNORECASE):
            language_scores['french'] += 5
        elif re.search(r'(what\'s|how\'s|don\'t|can\'t)', text_clean, re.IGNORECASE):
            language_scores['english'] += 5

        # 🔹 Retourner la langue avec le meilleur score
        if max(language_scores.values()) > 0:
            detected_lang = max(language_scores, key=language_scores.get)
            logger.info(f"Langue détectée: {detected_lang} (scores: {language_scores})")
            return detected_lang
        
        # Par défaut, français
        return 'french'
    def is_greeting_or_simple_query(self, text, language):
        """Détecte UNIQUEMENT les salutations PURES et questions simples - VERSION CORRIGÉE"""
        if not text:
            return False, None

        text_clean = text.lower().strip()
        
        product_question_patterns = {
            'arabic': [
                r'(عندكم|فيه|موجود|تنجموا|قولولي|وينو|كيفاش)',
                r'(كريم|شامبو|دوا|دواء|واقي|شمس|عطر|فيتامين)',
                r'(بقداش|بكداش|السعر|الثمن)',
                r'(بشرة|وجه|شعر|دهني|جاف|مختلط)',
                r'(اقترحولي|انصحوني|شنو|ايش|شيا)'
            ],
            'french': [
                r'(avez-vous|proposez|recommandez|conseillez|suggérez|cherche|veux|voudrais|aimerais)',
                r'(crème|écran|solaire|shampoing|médicament|parfum|vitamine|sérum|gel)',
                r'(combien|prix|coût|coûte)',
                r'(peau|visage|cheveux|grasse|sèche|mixte|sensible)',
                r'(propositions|suggestions|conseils|produits|gamme)',
                r'(pour\s+(peau|visage|cheveux)|anti-|protection|hydratant)'
            ],
            'english': [
                r'(do\s+you\s+have|recommend|suggest|propose|looking\s+for|want|need)',
                r'(cream|sunscreen|shampoo|medicine|perfume|vitamin|serum|gel)',
                r'(how\s+much|price|cost)',
                r'(skin|face|hair|oily|dry|combination|sensitive)',
                r'(products|suggestions|recommendations|options)',
                r'(for\s+(skin|face|hair)|anti-|protection|moisturizing)'
            ]
        }
        
        for pattern in product_question_patterns.get(language, product_question_patterns['french']):
            if re.search(pattern, text_clean, re.IGNORECASE):
                return False, None
        
        thanks_patterns = {
            'arabic': [r'^(شكرا|مرسي|يعطيك\s+الصحة)(\s+برشا)?$'],
            'french': [r'^(merci|merci\s+beaucoup|je\s+vous\s+remercie)(\s+.*)?$'],
            'english': [r'^(thank\s+you|thanks|thank\s+you\s+very\s+much)(\s+.*)?$']
        }

        for pattern in thanks_patterns.get(language, thanks_patterns['french']):
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True, 'thanks'

        pure_greeting_patterns = {
            'arabic': [
                r'^(أهلا|اهلا|مرحبا|السلام\s+عليكم|عسلامة)(\s*[!.?]?\s*)?$',
                r'^(صباح|مساء)\s+(الخير|طيبة)(\s*[!.?]?\s*)?$',
                r'^(يا\s+هلا|هاي|سالي)(\s*[!.?]?\s*)?$',
                r'^(كيفك|شلونك|احوالك|لاباس|شنحوالك)(\s*[!.?]?\s*)?$'
            ],
            'french': [
                r'^(bonjour|salut|bonsoir|hello|coucou)(\s*[!.?]?\s*)?$',
                r'^(bonne?\s+(journée|matinée|soirée|nuit))(\s*[!.?]?\s*)?$',
                r'^(comment\s+(ça\s+va|allez-vous))(\s*[!.?]?\s*)?$',
                r'^(ça\s+va)(\s*[!.?]?\s*)?$'
            ],
            'english': [
                r'^(hello|hi|hey)(\s*[!.?]?\s*)?$',
                r'^(good\s+(morning|evening|afternoon))(\s*[!.?]?\s*)?$',
                r'^(how\s+are\s+you|what\'s\s+up|how\'s\s+it\s+going)(\s*[!.?]?\s*)?$'
            ]
        }

        for pattern in pure_greeting_patterns.get(language, pure_greeting_patterns['french']):
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True, 'greeting'

        greeting_with_question_patterns = {
            'french': [
                r'^(bonjour|salut|bonsoir)\s*,?\s+.{10,}',  # Salutation + texte long
                r'^(bonjour|salut|bonsoir)\s*,?\s+(quel|que|comment|combien|avez|proposez)'
            ],
            'arabic': [
                r'^(أهلا|مرحبا|السلام)\s*,?\s+.{10,}',
                r'^(أهلا|مرحبا|السلام)\s*,?\s+(شنو|ايش|كيف|عندكم)'
            ],
            'english': [
                r'^(hello|hi|good\s+morning)\s*,?\s+.{10,}',
                r'^(hello|hi|good\s+morning)\s*,?\s+(what|how|do\s+you|can\s+you)'
            ]
        }
        
        for pattern in greeting_with_question_patterns.get(language, greeting_with_question_patterns['french']):
            if re.search(pattern, text_clean, re.IGNORECASE):
                return False, None  
        return False, None


    def generate_simple_response(self, response_type, language):
        """Génère une réponse simple pour les salutations"""
        import random
        
        responses = self.language_patterns[language].get(f'{response_type}_responses', [])
        if responses:
            return random.choice(responses)
        
        if response_type == 'greeting':
            if language == 'arabic':
                return "أهلا! كيف يمكنني مساعدتك؟"
            elif language == 'english':
                return "Hello! How can I help you?"
            else:
                return "Bonjour ! Comment puis-je vous aider ?"
        elif response_type == 'thanks':
            if language == 'arabic':
                return "العفو! أي وقت."
            elif language == 'english':
                return "You're welcome!"
            else:
                return "De rien !"


    def generate_response(self, question, context_passages, language='french'):
        """Génération de réponse multilingue STRICTE avec Gemini - VERSION CORRIGÉE"""
        logger.info(f"Question reçue : {question} (langue: {language})")

        # 🔹 SEULEMENT pour les vraies salutations pures
        is_simple, response_type = self.is_greeting_or_simple_query(question, language)
        if is_simple:
            simple_response = self.generate_simple_response(response_type, language)
            logger.info(f"Réponse simple générée : {simple_response}")
            return simple_response

        # 🔹 Pour toutes les autres questions, traitement complet
        context_text = "\n\n".join(context_passages) if context_passages else ""
        
        # 🔸 Prompts améliorés avec meilleure détection des questions produits
        prompts = {
            'arabic': f"""أنت مساعد صوتي لصيدلية باراليبل فقط. جاوب باللهجة التونسية.

    🔹 السؤال:
    {question}

    🔸 المنتجات المتوفرة في باراليبل فقط:
    {context_text}

    ⚠️ تعليمات صارمة:
    - استعمل فقط المنتجات المذكورة أعلاه
    - إذا لم تجد منتج مناسب، اعتذر وقل أنه غير متوفر حالياً
    - لا تخترع منتجات أو أسعار
    - لا تذكر منتجات من صيدليات أخرى
    - كن صادق إذا لم تجد المنتج المطلوب

    🟢 الرد بالتونسي:""",

            'french': f"""Tu es l'assistant vocal de la pharmacie Paralabel UNIQUEMENT. 

    🔹 Question du client :
    {question}

    🔸 Produits RÉELLEMENT disponibles chez Paralabel :
    {context_text}

    ⚠️ Instructions STRICTES :
    - Utilise UNIQUEMENT les produits listés ci-dessus
    - Si aucun produit ne correspond, excuse-toi et dis qu'il n'est pas disponible actuellement
    - Ne JAMAIS inventer de produits, prix ou marques
    - Ne JAMAIS mentionner de produits d'autres pharmacies
    - Sois honnête si le produit demandé n'existe pas dans notre stock

    ✅ Si des produits correspondent :
    - Présente les options disponibles chez Paralabel
    - Mentionne les marques, prix et caractéristiques
    - Donne des conseils d'utilisation si pertinent

    ❌ Si aucun produit ne correspond exactement :
    "Je suis désolé, nous n'avons pas ce type de produit en stock actuellement chez Paralabel. Je vous invite à consulter notre pharmacien sur place ou à nous contacter pour plus d'informations sur nos arrivages."

    🟢 Réponse professionnelle et honnête :""",

            'english': f"""You are the voice assistant for Paralabel pharmacy ONLY.

    🔹 Customer question:
    {question}

    🔸 Products ACTUALLY available at Paralabel:
    {context_text}

    ⚠️ STRICT Instructions:
    - Use ONLY the products listed above
    - If no product matches, apologize and say it's not currently available
    - NEVER invent products, prices or brands
    - NEVER mention products from other pharmacies
    - Be honest if the requested product doesn't exist in our stock

    ✅ If products match:
    - Present available options at Paralabel
    - Mention brands, prices and features
    - Give usage advice if relevant

    ❌ If no product matches exactly:
    "I'm sorry, we don't currently have this type of product in stock at Paralabel. Please visit our pharmacy or contact us for information about new arrivals."

    🟢 Professional and honest answer:"""
        }

        try:
            prompt = prompts.get(language, prompts['french'])
            response = self.gemini_model.generate_content(prompt)
            generated_response = response.text.strip()
            
            logger.info(f"Réponse générée: {generated_response}")
            return generated_response

        except Exception as e:
            logger.error(f"Erreur génération réponse: {e}")
            error_responses = {
                'arabic': "آسف، ما قدرتش نجاوب. جرب مرة أخرى.",
                'french': "Désolé, je n'ai pas pu générer de réponse. Veuillez réessayer.",
                'english': "Sorry, I couldn't generate a response. Please try again."
            }
            return error_responses.get(language, error_responses['french'])

    def text_to_speech(self, text, language='french'):
        """Convertit le texte en parole dans la langue appropriée"""
        try:
            if not hasattr(pygame.mixer, 'music'):
                logger.warning("TTS non disponible - pygame non initialisé")
                return False

            logger.info(f" TTS en {language} pour: {text[:50]}...")

            if language.lower() == 'arabic':
                async def speak_with_edge():
                    try:
                        voice = "ar-TN-HediNeural"  # Voix arabe tunisienne (ou "ar-SA", etc.)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                            temp_path = tmp_file.name

                        communicate = edge_tts.Communicate(text=text, voice=voice)
                        await communicate.save(temp_path)

                        pygame.mixer.music.load(temp_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            await asyncio.sleep(0.1)

                        os.unlink(temp_path)
                        logger.info("✅ Lecture audio avec edge-tts réussie")
                        return True
                    except Exception as e:
                        logger.error(f"⚠️ Erreur TTS (edge-tts): {e}")
                        return False

                return asyncio.run(speak_with_edge())

            else:
                tts_lang_map = {
                    'french': 'fr',
                    'english': 'en'
                }
                tts_lang = tts_lang_map.get(language.lower(), 'fr')

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    temp_path = tmp_file.name

                tts = gTTS(text=text, lang=tts_lang, slow=False)
                tts.save(temp_path)

                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                os.unlink(temp_path)
                logger.info("✅ Lecture audio avec gTTS réussie")
                return True

        except Exception as e:
            logger.error(f"⚠️ Erreur globale TTS: {e}")
            return False

        
    def preprocess_audio(self, audio_path):
        """Prétraitement audio amélioré"""
        try:
            if audio_path.lower().endswith(".mp3"):
                logger.info("Conversion MP3 vers WAV...")
                mp3_audio = AudioSegment.from_mp3(audio_path)
                audio_path_wav = "converted.wav"
                mp3_audio.export(audio_path_wav, format="wav")
                audio_path = audio_path_wav

            audio, sr = librosa.load(audio_path, sr=16000)

            audio_clean = nr.reduce_noise(y=audio, sr=sr)

            audio_clean = audio_clean / np.max(np.abs(audio_clean))

            processed_path = "temp_processed.wav"
            write(processed_path, sr, (audio_clean * 32767).astype(np.int16))

            logger.info("Prétraitement audio terminé")
            return processed_path
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement audio: {e}")
            return audio_path

    def record_audio(self, duration=5, samplerate=16000):
        """Enregistrement audio"""
        logger.info(" Démarrage de l'enregistrement...")
        try:
            recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()
            
            recording = recording / np.max(np.abs(recording))
            
            audio_path = "temp.wav"
            write(audio_path, samplerate, (recording * 32767).astype(np.int16))
            logger.info("✅ Enregistrement terminé")
            return audio_path
        except Exception as e:
            logger.error(f"⚠️ Erreur lors de l'enregistrement: {e}")
            raise

    def transcribe_audio_multilingual(self, audio_path):
        """Transcription multilingue améliorée"""
        try:
            processed_audio = self.preprocess_audio(audio_path)
            
            result = self.whisper_model.transcribe(
                processed_audio,
                language=None,
                task="transcribe",
                temperature=0.0,
                beam_size=5,
                best_of=5,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False
            )
            
            text = result.get("text", "").strip()
            detected_lang_whisper = result.get("language", "")
            
            logger.info(f"Whisper détecté: {detected_lang_whisper}, Texte: {text}")
            
            corrected_text = self.post_process_transcription(text)
            
            if processed_audio != audio_path and os.path.exists(processed_audio):
                os.remove(processed_audio)
                
            return corrected_text
            
        except Exception as e:
            logger.error(f"⚠️ Erreur lors de la transcription: {e}")
            return ""

    def post_process_transcription(self, text):
        """Post-traitement avec corrections spécifiques"""
        if not text:
            return ""
            
        corrections = [
            (r'la roche posay|laroche posay|la roche-posay', 'La Roche-Posay'),
            (r'eucerin|eucérin|eucerine', 'Eucerin'),
            (r'bioderma|bio derma', 'Bioderma'),
            (r'vichy|vichi|vishy', 'Vichy'),
            (r'avène|avene|aven', 'Avène'),
            (r'cerave|cera ve|sera ve', 'CeraVe'),
            (r'nivea|nivéa', 'Nivea'),
            (r'garnier|garnyé', 'Garnier'),
            (r'loreal|l\'oreal|l\'oréal', "L'Oréal"),
            
            (r'كريم|krim|krym', 'crème'),
            (r'شامبو|شامبوان|shampo', 'shampoing'),
            (r'سيروم|sirum|sérom', 'sérum'),
            (r'جيل|jil', 'gel'),
            (r'ماسك|mask', 'masque'),
            (r'برفان|برفون|parfon', 'parfum'),
            (r'دودوران|deodoran', 'déodorant'),
            (r'صابون|saboun', 'savon'),
            (r'فيتامين|vitamin', 'vitamine'),
            (r'دوا|دواء', 'médicament'),
            (r'واقي\s+شمس|écran\s+solaire', 'écran solaire'),
            
            (r'بقداش|بكداش|b9adesh', 'combien'),
            (r'السعر|الثمن|prix', 'prix'),
            (r'دينار|dinar', 'dinar'),
            (r'يورو|euro', 'euro'),
            (r'مل|ml', 'ml'),
            (r'غرام|جرام|gram', 'gramme'),
            
            (r'paralabel|para label|بارالابل', 'Paralabel'),
            (r'صيدلية|pharmacie|pharmacy', 'pharmacie'),
        ]
        
        corrected_text = text
        for pattern, replacement in corrections:
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        
        corrected_text = re.sub(r'\s+', ' ', corrected_text).strip()
        
        return corrected_text

    def handle_command(self, command_json):
        """Gestionnaire de commandes principal"""
        try:
            command = json.loads(command_json)
            
            if command["cmd"] == "process_audio":
                start_time = time.time()
                logger.info(" Démarrage du traitement audio")
                
                audio_file = self.record_audio()
                question = self.transcribe_audio_multilingual(audio_file)
                
                if not question:
                    error_msg = "Désolé, je n'ai pas pu comprendre votre question."
                    self.text_to_speech(error_msg, 'french')
                    return json.dumps({"error": error_msg})

                detected_language = self.detect_language_advanced(question)
                logger.info(f" Langue détectée: {detected_language}")

                is_simple, response_type = self.is_greeting_or_simple_query(question, detected_language)
                if is_simple:
                    response = self.generate_simple_response(response_type, detected_language)
                    self.text_to_speech(response, detected_language)

                    elapsed = time.time() - start_time
                    logger.info(f" Réponse simple générée en {elapsed:.2f} secondes")

                    return json.dumps({
                        "question": question,
                        "response": response,
                        "language": detected_language,
                        "processing_time": elapsed
                    }, ensure_ascii=False)

                passages = self.search_database(question)
            
                response = self.generate_response(question, passages, detected_language)
            
                self.text_to_speech(response, detected_language)
            
                elapsed = time.time() - start_time
                logger.info(f"✅ Traitement complet en {elapsed:.2f} secondes")
            
                return json.dumps({
                    "question": question,
                    "response": response,
                    "language": detected_language,
                    "processing_time": elapsed
                }, ensure_ascii=False)
        
            elif command["cmd"] == "test_tts":
                test_messages = {
                    'arabic': "أهلا وسهلا في صيدلية باراليبل",
                    'french': "Bonjour et bienvenue à la pharmacie Paralabel",
                    'english': "Hello and welcome to Paralabel pharmacy"
                }
            
                for lang, message in test_messages.items():
                    logger.info(f" Test TTS {lang}: {message}")
                    self.text_to_speech(message, lang)
                    time.sleep(1)
            
                return json.dumps({"status": "TTS test completed"})

            elif command["cmd"] == "reload_database":
                try:
                    if self.db_available:
                        cursor = self.conn.cursor()
                        cursor.execute("DELETE FROM product_embeddings")
                        cursor.execute("DELETE FROM products")
                        self.conn.commit()
                        
                        self.load_knowledge_base()
                        
                        return json.dumps({"status": "Database reloaded successfully"})
                    else:
                        return json.dumps({"error": "Database not available"})
                except Exception as e:
                    logger.error(f"Erreur lors du rechargement: {e}")
                    return json.dumps({"error": f"Reload failed: {str(e)}"})

            elif command["cmd"] == "search_test":
                test_query = command.get("query", "écran solaire pour peau mixte")
                
                logger.info(f" Test de recherche pour: {test_query}")
                passages = self.search_database(test_query)
                
                return json.dumps({
                    "query": test_query,
                    "results": passages,
                    "count": len(passages)
                }, ensure_ascii=False)
        
            else:
                return json.dumps({"error": "Commande inconnue"})
    
        except Exception as e:
            logger.error(f"⚠️ Erreur dans handle_command: {e}")
            return json.dumps({"error": str(e)})

    def test_greeting_detection(self):
        """Test de la détection des salutations"""
        test_phrases = [
            "أهلا وسهلا",
            "مرحبا كيفك؟",
            "لاباس عليك؟",
            "شنحوالك؟",
            "صباح الخير",
            "مساء الخير",
            "شكرا برشا",
            "يعطيك الصحة",
            
            "Bonjour comment allez-vous ?",
            "Salut ça va ?",
            "Bonsoir",
            "Merci beaucoup",
            
            "Hello how are you?",
            "Hi there!",
            "Good morning",
            "Thank you",
            
            "Propose-moi un écran solaire pour peau mixte",
            "بقداش كريم La Roche Posay؟",
            "عندكم شامبو للشعر الدهني؟",
            "Combien coûte ce produit?",
            "Do you have vitamin C serum?"
        ]
        
        logger.info(" Test de détection des salutations:")
        for phrase in test_phrases:
            lang = self.detect_language_advanced(phrase)
            is_greeting_result, greeting_type = self.is_greeting_or_simple_query(phrase, lang)
            logger.info(f"'{phrase}' -> Salutation: {is_greeting_result}, Type: {greeting_type}, Langue: {lang}")
            
            if is_greeting_result:
                response = self.generate_simple_response(greeting_type, lang)
                logger.info(f"  Réponse: {response}\n")

    def export_database_stats(self):
        """Exporter les statistiques de la base de données"""
        if not self.db_available:
            return {"error": "Database not available"}
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM products")
            total_products = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM product_embeddings")
            total_embeddings = cursor.fetchone()[0]
            
            cursor.execute("SELECT category, COUNT(*) FROM products GROUP BY category")
            categories = cursor.fetchall()
            
            cursor.execute("SELECT brand, COUNT(*) FROM products GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 10")
            brands = cursor.fetchall()
            
            return {
                "total_products": total_products,
                "total_embeddings": total_embeddings,
                "categories": dict(categories),
                "top_brands": dict(brands)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export des stats: {e}")
            return {"error": str(e)}

    def __del__(self):
        """Fermer la connexion SQLite lors de la destruction de l'objet"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                logger.info("Connexion SQLite fermée")
        except:
            pass

def main():
    try:
        logger.info("🚀 Démarrage de l'assistant vocal Paralabel avec SQLite...")
        
        try:
            assistant = VoiceAssistant(fallback_only=False)
            logger.info("✅ Assistant vocal initialisé avec SQLite")
        except Exception as e:
            logger.warning(f"⚠️ SQLite non disponible, initialisation avec fallback...")
            try:
                assistant = VoiceAssistant(fallback_only=True)
                logger.info("✅ Assistant vocal initialisé en mode fallback")
            except Exception as e2:
                logger.error(f"⚠️ Échec complet de l'initialisation: {e2}")
                return None
        
        if len(sys.argv) > 1:
            if sys.argv[1] == "--test":
                assistant.test_greeting_detection()
                return assistant
            elif sys.argv[1] == "--stats":
                stats = assistant.export_database_stats()
                print(json.dumps(stats, indent=2, ensure_ascii=False))
                return assistant
        
        return assistant
        
    except KeyboardInterrupt:
        logger.info("⏹️ Assistant arrêté par l'utilisateur")
        return None
    except Exception as e:
        logger.error(f"⚠️ Erreur fatale: {e}")
        return None

if __name__ == "__main__":
    assistant = main()
    
    if assistant:
        logger.info(" Assistant vocal prêt - En attente de commandes...")
        logger.info(" Commandes disponibles:")
        logger.info("  - process_audio: Traitement vocal complet")
        logger.info("  - test_tts: Test du système TTS")
        logger.info("  - reload_database: Recharger depuis all_docs.json")
        logger.info("  - search_test: Test de recherche")
        
        try:
            for line in sys.stdin:
                response = assistant.handle_command(line.strip())
                print(response)
                sys.stdout.flush()
        except KeyboardInterrupt:
            logger.info("⏹️ Assistant arrêté par l'utilisateur")
        except Exception as e:
            logger.error(f"⚠️ Erreur pendant l'exécution: {e}")
    else:
        logger.error("⚠️ Assistant vocal non disponible")
        sys.exit(1)