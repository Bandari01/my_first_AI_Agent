class RAGConfig:
    def __init__(self, llm_model="gpt-4o-mini", temperature=0.1, max_tokens=4000,  # Lower the temperature to achieve a more consistent output
                 max_retries=3, knowledge_base_path="RAG_tool/knowledge_base",
                 top_k_retrieval=8, similarity_threshold=0.1, enable_validation=True,  # Increase the number of searches
                 openai_api_key=None):
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.knowledge_base_path = knowledge_base_path
        self.top_k_retrieval = top_k_retrieval
        self.similarity_threshold = similarity_threshold
        self.enable_validation = enable_validation
        self.openai_api_key = openai_api_key