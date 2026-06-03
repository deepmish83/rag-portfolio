"""
downbload the wikipepedia articles into ./corpus as plain t6est files
These are out RAG document for today.

"""

import urllib.request, urllib.parse
import os
import json

os.makedirs('corpus', exist_ok=True)

TITLES = [
    "Hong KOng INternational Airport",
    "HK Express",
    "Cathay Pacific",
    "Low-cost carrier",
    "Airbus 320 family",
    "Microsoft 365",
    "Microsoft Copilot",
    "Generative Artificial Intelligence",
    "Retrieval Augmented Generation",
    "Large Language Models"
]

for title in TITLES:
    encoded = urllib.parse.quote(title.replace(' ', '_'))
    url = (f"https://en.wikipedia.org/w/api.php"
           f"?action=query&format=json&titles={encoded}"
           f"&prop=extracts&explaintext=true&redirects=1")
    
    print(f"Fetching: {title} {url}...")

    req = urllib.request.Request(url, headers={'User-Agent': 'RAGServiceBot/1.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))

    page = data['query']['pages']

    for _, page in page.items():
        content = page.get('extract', '')
        if content:
            safe = title.replace(' ', '_').replace('/', '_')
            with open(f'corpus/{safe}.txt', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f" saved: {safe}.txt ({len(content)} characters)")
        else:
            print(f" No content found for: {title}")
print("All articles have been processed.")
