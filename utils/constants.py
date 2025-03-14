class DocumentMetadata:
    """Document class metadata keys"""

    __slots__ = ()
    # vector store fields
    GOOGLE_INDEX = "google_index"
    CATEGORY = "categoryID"
    SUBCATEGORY = "subcategoryID"
    SLUG = "slug"
    PARENT = "parent"
    QUESTION = "question"
    ANSWER = "answer"
    # splitting and embedding fields
    EMBEDDING_MODEL = "embedding_model"
    NTH_CHUNK = "nth_chunk"
    # search result fields
    SCORE = "score"
    CHILDREN = "children"
