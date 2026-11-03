
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from tempfile import TemporaryDirectory

URL = (
    "https://www.python.org/ftp/python/doc/3.14.0/"
    "python-3.14.0-docs-text.zip"
)

DEST = Path("data/documents")

def main():
    DEST.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp:
        archive = Path(temp) / "python-docs.zip"

        print("Downloading Python documentation...")
        urlretrieve(URL, archive)

        print("Extracting documentation...")

        with ZipFile(archive) as z:
            for item in z.infolist():
                if not item.filename.endswith(".txt"):
                    continue

                # Preserve original folder structure.
                relative = Path(item.filename)

                if relative.is_absolute() or ".." in relative.parts:
                    continue

                target = DEST / relative
                target.parent.mkdir(parents=True, exist_ok=True)

                with z.open(item) as source:
                    target.write_bytes(source.read())

    print("Corpus downloaded successfully.")

if __name__ == "__main__":
    main()
