"""
RAG Agent - completely compatible
"""

import time
import asyncio
import os

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available")


class RAGResult:
    def __init__(self, status, generated_code, retrieved_knowledge, reasoning_steps,
                 execution_time, retrieval_count, llm_calls, error_message=None,
                 submission_file_path=None):
        self.status = status
        self.generated_code = generated_code
        self.retrieved_knowledge = retrieved_knowledge
        self.reasoning_steps = reasoning_steps
        self.execution_time = execution_time
        self.retrieval_count = retrieval_count
        self.llm_calls = llm_calls
        self.error_message = error_message
        self.submission_file_path = submission_file_path


class RAGAgent:
    def __init__(self, config):
        self.config = config
        from .knowledge_retriever import KnowledgeRetriever
        self.retriever = KnowledgeRetriever(config.knowledge_base_path)
        self.retrieval_count = 0
        self.llm_call_count = 0

        # Initialize OpenAI client with API key
        if OPENAI_AVAILABLE:
            # Use API key from config if provided, otherwise use environment variable
            api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm_client = AsyncOpenAI(api_key=api_key)
                print("OpenAI client initialized with API key")
            else:
                self.llm_client = None
                print("Warning: No OpenAI API key provided")
        else:
            self.llm_client = None
            print("Warning: OpenAI client not available")

    async def run(self, problem_description, data_info):
        start_time = time.time()
        reasoning_steps = []

        try:
            # Check if OpenAI client is available
            if not self.llm_client:
                return RAGResult(
                    status="failed",
                    generated_code="",
                    retrieved_knowledge=[],
                    reasoning_steps=["OpenAI client not available. Please check API key."],
                    execution_time=0,
                    retrieval_count=0,
                    llm_calls=0,
                    error_message="OpenAI API key not configured properly"
                )

            # Step 1: Retrieve knowledge
            reasoning_steps.append("Analyzing problem and retrieving relevant knowledge...")
            retrieved_knowledge = await self._retrieve_relevant_knowledge(problem_description, data_info)

            # Step 2: Build enhanced prompt
            reasoning_steps.append("Building enhanced context with retrieved knowledge...")
            enhanced_prompt = self._build_enhanced_prompt(problem_description, data_info, retrieved_knowledge)

            # Step 3: Generate solution
            reasoning_steps.append("Generating solution with knowledge-enhanced context...")
            solution_result = await self._generate_solution(enhanced_prompt)

            # Step 4: Validate if enabled
            if self.config.enable_validation:
                reasoning_steps.append("Validating and refining solution...")
                final_code = await self._validate_and_refine(solution_result, retrieved_knowledge)
            else:
                final_code = solution_result

            execution_time = time.time() - start_time

            return RAGResult(
                status="completed",
                generated_code=final_code,
                retrieved_knowledge=retrieved_knowledge,
                reasoning_steps=reasoning_steps,
                execution_time=execution_time,
                retrieval_count=self.retrieval_count,
                llm_calls=self.llm_call_count
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return RAGResult(
                status="failed",
                generated_code="",
                retrieved_knowledge=[],
                reasoning_steps=reasoning_steps,
                execution_time=execution_time,
                retrieval_count=self.retrieval_count,
                llm_calls=self.llm_call_count,
                error_message=str(e)
            )

    async def _retrieve_relevant_knowledge(self, problem_description, data_info):
        query_parts = [problem_description]

        if 'columns' in data_info and data_info['columns']:
            query_parts.append("Data columns: " + ", ".join(data_info['columns'][:10]))

        if 'train_files' in data_info and data_info['train_files']:
            query_parts.append("Training files: " + ", ".join(data_info['train_files']))

        query = " ".join(query_parts)

        retrieved_docs = self.retriever.retrieve(
            query,
            top_k=self.config.top_k_retrieval,
            similarity_threshold=self.config.similarity_threshold
        )
        self.retrieval_count += 1

        return retrieved_docs

    def _build_enhanced_prompt(self, problem_description, data_info, retrieved_knowledge):
        knowledge_context = self._format_knowledge_context(retrieved_knowledge)
        data_context = self._format_data_context(data_info)

        # Check if it is Store Sales competition
        is_store_sales = "store-sales" in problem_description.lower()

        if is_store_sales:
            task_specific_instructions = """
## STORE SALES COMPETITION SPECIFIC REQUIREMENTS:

### CRITICAL REQUIREMENTS:
1. The code MUST generate a submission.csv file with columns 'id' and 'sales'
2. Handle categorical variables properly (use LabelEncoder or OneHotEncoder for 'family', 'city', 'state', 'type')
3. Use time-series appropriate validation (TimeSeriesSplit, not random split)
4. Merge data correctly: 
   - train/test with stores on 'store_nbr'
   - with oil on 'date' 
   - with transactions on ['date', 'store_nbr']
   - holidays should be handled as flags, not merged directly (to avoid row explosion)
5. Test set has 28512 records - ensure predictions match this count
6. Include proper error handling and logging
7. Ensure no negative predictions (use np.maximum(0, predictions))

### DATA FILES AND STRUCTURE:
- train.csv: id, date, store_nbr, family, sales, onpromotion
- test.csv: id, date, store_nbr, family, onpromotion  
- stores.csv: store_nbr, city, state, type, cluster
- oil.csv: date, oil
- holidays_events.csv: date, type, locale, locale_name, description, transferred
- transactions.csv: date, store_nbr, transactions

### FEATURE ENGINEERING REQUIREMENTS:
- Extract date features: year, month, day, day_of_week, is_weekend
- Create holiday flags from holidays_events.csv (only non-transferred holidays)
- Handle missing values in oil prices (forward fill)
- Include store metadata (city, state, type, cluster)
- Include transactions data aggregated by date and store
- Encode all categorical variables properly

### MODELING REQUIREMENTS:
- Use time-series appropriate model (RandomForest, XGBoost, LightGBM)
- Use time-series cross-validation or time-based split
- Target metric: RMSLE (Root Mean Squared Logarithmic Error)
- Ensure predictions are non-negative

### SUBMISSION REQUIREMENTS:
- File name: submission.csv
- Columns: id, sales
- Must have 28512 rows matching test set
- No index column in output

## CODE STRUCTURE REQUIREMENTS:
1. Complete, runnable Python code
2. Proper error handling with try/except
3. Informative logging/print statements
4. All necessary imports included
5. Code must execute without errors and produce submission.csv
"""
        else:
            task_specific_instructions = """
## GENERAL REQUIREMENTS:
- Generate appropriate predictions for the competition
- Create submission file in required format
- Ensure predictions are in correct range and scale
- Include proper error handling and logging
"""

        enhanced_prompt = f"""
# RAG-ENHANCED DATA ANALYTICS

## KNOWLEDGE CONTEXT:
{knowledge_context}

## DATA CONTEXT:
{data_context}

## PROBLEM:
{problem_description}

{task_specific_instructions}

## TASK:
Generate complete Python code that:
1. Loads and explores the data
2. Performs appropriate data preprocessing and feature engineering
3. Builds a time series forecasting model
4. Generates predictions for the test set
5. Creates a submission.csv file in the correct format
6. Includes proper validation and error handling

The code should be complete, runnable, and include all necessary imports.

Please provide the complete Python code:
```python
"""
        return enhanced_prompt

    def _format_knowledge_context(self, knowledge_docs):
        if not knowledge_docs:
            return "No relevant knowledge retrieved."

        sections = []
        for i, doc in enumerate(knowledge_docs, 1):
            section = f"{i}. [{doc['type']}] {doc['title']}\n"
            section += f"   {doc['content'][:200]}...\n"
            section += f"   Relevance: {doc.get('similarity', 0):.3f}"
            sections.append(section)

        return "\n".join(sections)

    def _format_data_context(self, data_info):
        context_parts = []

        if 'columns' in data_info and data_info['columns']:
            context_parts.append("Available columns: " + ", ".join(data_info['columns'][:15]))

        if 'train_files' in data_info and data_info['train_files']:
            context_parts.append("Training files: " + ", ".join(data_info['train_files']))

        if 'test_files' in data_info and data_info['test_files']:
            context_parts.append("Test files: " + ", ".join(data_info['test_files']))

        # Add detailed file information if available
        if 'all_files_info' in data_info and data_info['all_files_info']:
            context_parts.append("\nDetailed file information:")
            for file_name, file_info in data_info['all_files_info'].items():
                context_parts.append(f"- {file_name}: {len(file_info.get('columns', []))} columns")

        return "\n".join(context_parts) if context_parts else "No detailed data information."

    async def _generate_solution(self, enhanced_prompt):
        if not self.llm_client:
            return "# Mock solution - OpenAI not available\nprint('Hello World')"

        messages = [
            {
                "role": "system",
                "content": """You are an expert data scientist specializing in Kaggle competitions. 
Generate accurate, complete, and runnable Python code for data analysis and machine learning.
Ensure the code:
1. Is complete and can run without additional modifications
2. Includes all necessary imports
3. Handles errors gracefully
4. Produces the required output files
5. Follows best practices for the specific problem type
6. Includes informative print statements to track progress

IMPORTANT: For Store Sales competition, ensure the code handles data merging correctly and produces a submission.csv with exactly 28512 rows."""
            },
            {
                "role": "user",
                "content": enhanced_prompt
            }
        ]

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            self.llm_call_count += 1
            content = response.choices[0].message.content

            # Extract code from markdown if present
            if "```python" in content:
                start = content.find("```python") + 9
                end = content.find("```", start)
                code = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                code = content[start:end].strip()
            else:
                code = content

            # Ensure code ends with proper file generation
            if "submission.csv" not in code and "store-sales" in enhanced_prompt.lower():
                code += "\n\n# Ensure submission file is created\nif 'submission' in locals():\n    submission.to_csv('submission.csv', index=False)\n    print('Submission file created successfully!')\nelse:\n    print('Error: Submission dataframe not created')"

            return code

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    async def _validate_and_refine(self, initial_solution, knowledge):
        if not self.config.enable_validation or not self.llm_client:
            return initial_solution

        validation_prompt = f"""
## INITIAL CODE:
```python
{initial_solution}
        KNOWLEDGE FOR VALIDATION:
        {self._format_knowledge_context(knowledge)}

        VALIDATION TASKS:
        Please review and improve this code with focus on:

        Completeness: Ensure all necessary steps are included

        Correctness: Fix any logical errors or data handling issues

        Robustness: Add proper error handling

        Performance: Optimize for large datasets if needed

        Output: Ensure submission.csv is generated with correct format

        Specifically check for:

        Proper handling of categorical variables

        Correct data merging without row explosion

        Time-series appropriate validation

        Non-negative predictions for RMSLE

        Correct submission file format (id, sales)

        Please provide the improved code:
        """

        messages = [
            {
                "role": "system",
                "content": "You are a senior data scientist reviewing and improving code for Kaggle competitions. Focus on making the code robust, efficient, and correct."
            },
            {
                "role": "user",
                "content": validation_prompt
            }
        ]

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=self.config.max_tokens
            )

            self.llm_call_count += 1
            content = response.choices[0].message.content

            if "```python" in content:
                start = content.find("```python") + 9
                end = content.find("```", start)
                return content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                return content[start:end].strip()
            else:
                return content
        except Exception as e:
            print(f"Validation failed: {e}")
            return initial_solution  # Return original if validation fails_build_enhanced_prompt