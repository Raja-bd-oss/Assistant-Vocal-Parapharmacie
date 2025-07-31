import os
import requests
import certifi
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import re  # <-- import pour regex

BASE_URL = "https://www.paralabel.tn/"
OUTPUT_FILE = os.path.join("..", "database", "all_docs.json")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

def get_soup(url):
    resp = requests.get(url, headers=HEADERS, verify=certifi.where())
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def extract_category_links():
    print("Récupération des catégories depuis la page principale...")
    soup = get_soup(BASE_URL)
    menu_div = soup.find("div", id="_desktop_vegamenu")
    if not menu_div:
        print("Erreur: menu de catégories introuvable.")
        return []

    links = set()
    def recursive_extract(ul_element):
        for li in ul_element.find_all("li", recursive=False):
            a = li.find("a", href=True)
            if a:
                url = urljoin(BASE_URL, a['href'])
                if urlparse(url).netloc == urlparse(BASE_URL).netloc:
                    links.add(url)
            sub_ul = li.find("ul", class_="menu-dropdown")
            if sub_ul:
                recursive_extract(sub_ul)

    top_ul = menu_div.find("ul", class_="menu-content")
    if top_ul:
        recursive_extract(top_ul)
    else:
        print("Erreur: structure de menu inattendue.")
    
    print(f"Nombre total de catégories récupérées : {len(links)}")
    return list(links)

def get_product_links_from_category(category_url):
    product_links = set()
    page_url = category_url
    while page_url:
        print(f"Scraping page catégorie: {page_url}")
        soup = get_soup(page_url)
        product_cards = soup.select("article.product-miniature a.product-thumbnail")
        for a in product_cards:
            href = a.get("href")
            if href:
                full_url = urljoin(BASE_URL, href)
                product_links.add(full_url)
        next_page = soup.select_one("li.pagination_next a")
        if next_page and next_page.get("href"):
            next_url = urljoin(BASE_URL, next_page['href'])
            if next_url == page_url:
                break
            page_url = next_url
            time.sleep(1)
        else:
            break
    print(f"Total produits trouvés dans cette catégorie : {len(product_links)}")
    return list(product_links)

def extract_price(text):
    # Cherche un nombre (avec , ou . décimal)
    match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if match:
        number = match.group(1).replace(',', '.')  # normaliser décimal
        # On ajoute "DT" car le site n'affiche pas l'unité
        return f"{number} DT"
    else:
        return text.strip()

def scrape_product(product_url):
    print(f"Scraping produit : {product_url}")
    soup = get_soup(product_url)

    title = "N/A"
    for sel in [
        "h1.product-name-costumization",
        "h1.h1.namne_details.product-name-costumization",
        "h1.h1",
        "h1"
    ]:
        tag = soup.select_one(sel)
        if tag and tag.get_text(strip=True):
            title = tag.get_text(strip=True)
            break

    # Prix - version corrigée
    price = "N/A"
    price_tag = soup.select_one("span.current-price-value")
    if price_tag:
        content_attr = price_tag.get("content")
        if content_attr:
            price = f"{content_attr} DT"
        else:
            price = price_tag.get_text(strip=True) + " DT"

    short_desc = ""
    short_tag = soup.select_one("div.product-short-description, div.product-description-short, div.product-desc")
    if short_tag:
        short_desc = short_tag.get_text(separator="\n", strip=True)

    long_desc = ""
    long_tag = soup.select_one("div.product-description, section#description")
    if long_tag:
        long_desc = long_tag.get_text(separator="\n", strip=True)

    features = {}
    table = soup.find("table", class_="product-attributes")
    if table:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                features[label] = value

    return {
        "url": product_url,
        "title": title,
        "price": price,
        "short_description": short_desc,
        "long_description": long_desc,
        "features": features
    }


def main():
    all_products = []
    categories = extract_category_links()
    if not categories:
        print("Aucune catégorie trouvée, arrêt.")
        return

    for cat_url in categories:
        print(f"\nTraitement de la catégorie : {cat_url}")
        try:
            product_urls = get_product_links_from_category(cat_url)
            for prod_url in product_urls:
                try:
                    product_data = scrape_product(prod_url)
                    all_products.append(product_data)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Erreur scraping produit {prod_url} : {e}")
        except Exception as e:
            print(f"Erreur scraping catégorie {cat_url} : {e}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_products, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Terminé. {len(all_products)} produits enregistrés dans {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
