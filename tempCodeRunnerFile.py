print("\n📚 Retrieved Chunks:")
            for doc in source_docs:
                print("•", doc.page_content[:300].replace("\n", " "), "\n")