
from pathlib import Path
from icrawler.builtin import BingImageCrawler

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"

searches = {
     "books": [
        "textbooks product photography",
        "novel books different covers",
        "school exercise books",
        "hardcover books",
        "paperback books",
        "fiction books collection",
        "university textbooks",
        "books stacked on table",
        "children story books"
    ],
    "bags": [
        "school backpacks product photography",
        "handbags different styles",
        "travel luggage bags",
        "leather bags",
        "laptop bags"
    ]
}

for class_name, keywords in searches.items():

    save_dir = RAW_DIR / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for index, keyword in enumerate(keywords):

        print(f"\nDownloading {class_name}: {keyword}")

        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={"root_dir": str(save_dir)}
        )

        crawler.crawl(
            keyword=keyword,
            max_num=50,
            min_size=(200, 200)
        )

print("\nADDITIONAL COLLECTION COMPLETE!")
