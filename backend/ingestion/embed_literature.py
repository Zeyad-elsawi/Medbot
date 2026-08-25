"""
backend/ingestion/embed_literature.py
Fetches PubMed abstracts via Entrez API and generates PubMedBERT embeddings.
Populates the medical_literature Qdrant collection.
No API key required for Entrez — uses free public access.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from Bio import Entrez
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Entrez requires an email — use a placeholder (no registration needed for basic access)
Entrez.email = "medbot-research@portfolio.dev"
Entrez.tool = "MedbotClinicalCopilot"

# Chest X-ray related search terms for PubMed
LITERATURE_QUERIES = [
    "chest X-ray pneumonia findings radiology",
    "pulmonary consolidation chest radiograph",
    "pleural effusion chest X-ray diagnosis",
    "pneumothorax radiological findings",
    "cardiomegaly chest radiograph interpretation",
    "lung opacity differential diagnosis radiology",
    "atelectasis chest X-ray patterns",
    "pulmonary edema radiological features",
    "tuberculosis chest X-ray patterns",
    "lung nodule radiograph evaluation",
]

MAX_ABSTRACTS_PER_QUERY = 50  # 500 total abstracts across all queries


def fetch_pubmed_ids(query: str, max_results: int) -> list[str]:
    """
    Search PubMed and return a list of article PMIDs.

    Args:
        query: Search query string.
        max_results: Maximum number of PMIDs to return.

    Returns:
        List of PMID strings.

    Raises:
        RuntimeError: If Entrez search fails.
    """
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, rettype="uilist")
        record = Entrez.read(handle)
        handle.close()
        pmids = record.get("IdList", [])
        logger.debug("Query '%s' returned %d PMIDs", query, len(pmids))
        return pmids
    except Exception as exc:
        logger.error("Entrez search failed for query '%s': %s", query, exc)
        raise RuntimeError(f"Entrez search failed: {exc}") from exc


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """
    Fetch full abstracts for a list of PMIDs.

    Fetches in batches of 20 to respect Entrez rate limits (3 requests/sec).

    Args:
        pmids: List of PubMed article IDs.

    Returns:
        List of dicts with 'pmid', 'title', 'abstract', 'journal', 'year' keys.
    """
    articles = []
    batch_size = 20

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="abstract",
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()

            for article_data in records.get("PubmedArticle", []):
                article = article_data.get("MedlineCitation", {})
                pmid = str(article.get("PMID", ""))
                article_info = article.get("Article", {})

                title = str(article_info.get("ArticleTitle", ""))
                abstract_text = ""

                abstract_obj = article_info.get("Abstract", {})
                if abstract_obj:
                    abstract_parts = abstract_obj.get("AbstractText", [])
                    if isinstance(abstract_parts, list):
                        abstract_text = " ".join(str(p) for p in abstract_parts)
                    else:
                        abstract_text = str(abstract_parts)

                journal = str(article_info.get("Journal", {}).get("Title", ""))

                year = ""
                pub_date = article_info.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))

                if abstract_text and len(abstract_text) > 100:
                    articles.append(
                        {
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract_text,
                            "journal": journal,
                            "year": year,
                            "text_for_embedding": f"{title}. {abstract_text}",
                        }
                    )

            time.sleep(0.35)  # Stay under 3 requests/sec Entrez rate limit

        except Exception as exc:
            logger.warning("Failed to fetch batch starting at index %d: %s", i, exc)
            continue

    return articles


def main() -> None:
    """
    Main pipeline: search PubMed → fetch abstracts → embed → save.

    Outputs:
    - data/literature/abstracts.json — raw article data
    - data/literature/literature_embeddings.npy — PubMedBERT embeddings
    - data/literature/literature_metadata.json — metadata aligned with embeddings
    """
    settings = get_settings()
    output_dir = Path(settings.literature_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all PMIDs across queries (deduplicated)
    all_pmids: set[str] = set()
    logger.info("Searching PubMed with %d queries...", len(LITERATURE_QUERIES))

    for query in tqdm(LITERATURE_QUERIES, desc="Searching PubMed"):
        try:
            pmids = fetch_pubmed_ids(query, MAX_ABSTRACTS_PER_QUERY)
            all_pmids.update(pmids)
            time.sleep(0.35)  # Rate limiting
        except RuntimeError as exc:
            logger.warning("Skipping query '%s': %s", query, exc)

    logger.info("Total unique PMIDs found: %d", len(all_pmids))

    # Fetch all abstracts
    logger.info("Fetching abstracts for %d articles...", len(all_pmids))
    articles = fetch_abstracts(list(all_pmids))
    logger.info("Articles with valid abstracts: %d", len(articles))

    if not articles:
        logger.error("No abstracts fetched. Check network connection and Entrez access.")
        sys.exit(1)

    # Save raw abstracts
    with open(output_dir / "abstracts.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    logger.info("Raw abstracts saved to %s/abstracts.json", output_dir)

    # Load PubMedBERT and embed
    logger.info("Loading PubMedBERT for literature embedding...")
    model = SentenceTransformer(settings.pubmedbert_model)

    texts = [a["text_for_embedding"] for a in articles]
    logger.info("Generating embeddings for %d abstracts...", len(texts))

    embeddings = model.encode(
        texts,
        batch_size=settings.openi_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.save(output_dir / "literature_embeddings.npy", embeddings.astype(np.float32))

    literature_metadata = [
        {
            "pmid": a["pmid"],
            "title": a["title"],
            "abstract": a["abstract"],
            "journal": a["journal"],
            "year": a["year"],
        }
        for a in articles
    ]

    with open(output_dir / "literature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(literature_metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Literature embeddings saved — shape: %s | Articles: %d",
        embeddings.shape,
        len(articles),
    )


if __name__ == "__main__":
    main()
