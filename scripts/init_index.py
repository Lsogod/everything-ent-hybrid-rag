from app.infra.es import ElasticsearchStore


if __name__ == "__main__":
    store = ElasticsearchStore()
    store.ensure_index()
    print("index ensured")
