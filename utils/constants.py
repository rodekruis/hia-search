class DocumentMetadata:
    """Document class metadata keys"""

    __slots__ = ()
    # database fields
    GOOGLE_INDEX = "google_index"
    CATEGORY = "category"
    SUBCATEGORY = "subcategory"
    SLUG = "slug"
    PARENT = "parent"
    QUESTION = "question"
    ANSWER = "answer"
    # splitting and embedding fields
    EMBEDDING_MODEL = "embedding_model"
    NTH_CHUNK = "nth_chunk"
