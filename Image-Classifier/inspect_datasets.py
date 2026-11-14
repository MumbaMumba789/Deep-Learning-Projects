
from huggingface_hub import HfApi

api = HfApi()

candidates = [
    "vanlanenandr/Bags",
    "Rhynoa/Bags",
    "ceyda/fashion-products-small",
    "abdulrub/fashion-products",
    "AmandaMaduro/FashionProducts"
]

for dataset_id in candidates:

    print("\n" + "=" * 60)
    print(f"DATASET: {dataset_id}")
    print("=" * 60)

    try:
        info = api.dataset_info(
            dataset_id,
            files_metadata=True
        )

        print("Tags:", info.tags[:10])

        if info.card_data:
            print("Description:", info.card_data.get("description", "No description"))

        print("\nFiles:")

        for file in info.siblings[:20]:
            print(file.rfilename)

    except Exception as e:
        print("Could not inspect:", e)

print("\nINSPECTION COMPLETE")

print("\nSEARCHING FOR BOOK COVER DATASETS")

results = api.list_datasets(
    search="book covers",
    limit=15
)

for dataset in results:
    print(dataset.id)