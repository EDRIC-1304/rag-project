from src.loader import load_documents


def main():
    docs = load_documents()

    print("Loaded documents:")
    print("-----------------")

    for d in docs:
        print(d.page_content)
        print()


if __name__ == "__main__":
    main()
