class DocumentMetadata:
    """Document class metadata keys"""

    __slots__ = ()
    # vector store fields for all sheets
    GOOGLE_INDEX = "google_index"
    CATEGORY = "categoryID"
    SUBCATEGORY = "subcategoryID"
    SLUG = "slug"
    # vector store fields for Q&As sheet
    PARENT = "parent"
    QUESTION = "question"
    ANSWER = "answer"
    # vector store fields for Offers sheet
    NAME = "name"
    DESCRIPTION = "desciption"
    PHONENUMBERS = "phonenumbers"
    EMAILS = "emails"
    WEBURLS = "weburls"
    ADDRESS = "address"
    OPENWEEK = "openweek"
    OPENWEEKEND = "openwekend"
    NEEDTOKNOW = "needtoknow"
    MOREINFO = "moreinfo"
    # splitting and embedding fields
    EMBEDDING_MODEL = "embedding_model"
    NTH_CHUNK = "nth_chunk"
    # search result fields
    SCORE = "score"
    CHILDREN = "children"