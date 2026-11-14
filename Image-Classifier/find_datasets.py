
from huggingface_hub import HfApi

api = HfApi()

search_terms = ["books", "bags", "fashion products"]

for term in search_terms:

    print(f"\n{'='*50}")
    print(f"SEARCHING: {term.upper()}")
    print("="*50)

    results = api.list_datasets(
        search=term,
        limit=10
    )

    for dataset in results:
        print(dataset.id)
