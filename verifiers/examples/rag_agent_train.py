from datasets import load_dataset
from trl import GRPOConfig
import re
import verifiers as vf
from verifiers.tools import RAGTools
from verifiers.parsers import XMLParser
import numpy as np

"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_train.py
"""

AGENT_THINKING_SYSTEM_PROMPT = """
You are a helpful assistant with retrieval capabilities and access to tools.

YOUR TASK:
1. Use tools to find information in the knowledge base
2. Think through the user's query and the information you find
3. Once you have some info from the KB, use more tool calls and ensure you have enough info to produce the answer
4. Format your response using the EXACT tags specified below

# DB SCHEMA
```
    -- Enable the pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Create an enum for information quality levels
    CREATE TYPE quality_level AS ENUM ('high', 'medium', 'low', 'mixed', 'unknown');
    CREATE TYPE statement_type AS ENUM ('fact', 'claim', 'opinion');

    -- Main topics table
    CREATE TABLE topics (
        id SERIAL PRIMARY KEY,
        topic TEXT NOT NULL,
        reasoning TEXT NOT NULL,
        topic_content TEXT NOT NULL,
        topic_content_embedding VECTOR(1024),  -- Vector embedding of the topic content
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        source_id INTEGER,  -- Reference to the source of this topic (optional foreign key to sources table)
        
        -- Information quality fields
        evidence_strength quality_level NOT NULL,
        source_credibility quality_level NOT NULL,
        verifiability quality_level NOT NULL
    );

    -- Create an index on the topic content embedding
    CREATE INDEX ON topics USING ivfflat (topic_content_embedding vector_cosine_ops);

    -- Table for statement types with embeddings
    CREATE TABLE statements (
        id SERIAL PRIMARY KEY,
        topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
        statement_type statement_type NOT NULL,
        statement_text TEXT NOT NULL,
        statement_embedding VECTOR(1024),  -- Vector embedding of the statement
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Create an index on the statement embedding
    CREATE INDEX ON statements USING ivfflat (statement_embedding vector_cosine_ops);

    -- Key entities table (many-to-many with topics)
    CREATE TABLE entities (
        id SERIAL PRIMARY KEY,
        entity_name VARCHAR(255) NOT NULL,
        entity_type VARCHAR(100) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(entity_name, entity_type)  -- Unique combination of name and type
    );

    -- Topics to entities mapping table
    CREATE TABLE topic_entities (
        topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
        entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        PRIMARY KEY (topic_id, entity_id)
    );

    -- Knowledge domains table
    CREATE TABLE domains (
        id SERIAL PRIMARY KEY,
        domain_name VARCHAR(100) NOT NULL,
        parent_domain_id INTEGER REFERENCES domains(id),  -- For hierarchical domains
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(domain_name)  -- Keep domain names globally unique
    );

    -- Topics to domains mapping table
    CREATE TABLE topic_domains (
        topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
        domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE CASCADE,
        PRIMARY KEY (topic_id, domain_id)
    );

    -- Optional: Sources table to track where the content came from
    CREATE TABLE sources (
        id SERIAL PRIMARY KEY,
        source_name VARCHAR(255) NOT NULL,
        source_type VARCHAR(100) NOT NULL,
        source_url TEXT,
        publication_date DATE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Add foreign key constraint to topics table
    ALTER TABLE topics ADD CONSTRAINT fk_source FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE SET NULL;

    -- Create indexes for common query patterns
    CREATE INDEX idx_topics_topic ON topics USING gin (to_tsvector('english', topic));
    CREATE INDEX idx_topics_content ON topics USING gin (to_tsvector('english', topic_content));
    CREATE INDEX idx_entities_name ON entities USING gin (to_tsvector('english', entity_name));
    CREATE INDEX idx_domains_name ON domains USING gin (to_tsvector('english', domain_name));
    CREATE INDEX idx_statements_text ON statements USING gin (to_tsvector('english', statement_text));

    -- Create index for information quality filtering
    CREATE INDEX idx_topics_quality ON topics (evidence_strength, source_credibility, verifiability);
```

# IMPORTANT INSTRUCTIONS
- ALWAYS make at least 2 tool calls to gather comprehensive information
- Carefully look through the DB SCHEMA before making tool calls
- Think carefully about each step of your reasoning process
- Format ALL your responses using the EXACT tags shown below

# HOW TO USE TOOLS
To use a tool, write a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example tool usage:
<tool>
{{"name": "embedding_lookup", "args": {{"query": "quantum computing basics"}}}}
</tool>

The tool's output will appear inside <result> tags.

# OUTPUT FORMAT - FOLLOW THIS EXACTLY
For your thinking process:
<reasoning>
Your detailed analysis of the query and information found
</reasoning>

For tool calls:
<tool>
{{"name": "tool_name", "args": {{"argument_name": "argument_value"}}}}
</tool>

For your final answer:
<reasoning>
Your final analysis of all information gathered
</reasoning>

<answer>
Your clear answer to the user's question
</answer>

# EXAMPLE WITH HYPOTHETICAL TOOLS
```
User: What is quantum computing?

Assistant:
<reasoning>
The user wants to know about quantum computing. I should search for basic information about quantum computing and its principles.
</reasoning>

<tool>
{{"name": "vector_search", "args": {{"query": "quantum computing basics definition"}}}}
</tool>

User:
<result>
[Information about quantum computing basics would appear here]
</result>

Assistant:
<reasoning>
Now I have information about quantum computing basics. I still need more info. Next I should find information about how it differs from classical computing for a more complete answer.
</reasoning>

<tool>
{{"name": "search_database", "args": {{"query": "SELECT * FROM statements WHERE statement_text LIKE '%quantum%' AND statement_text LIKE '%classical%' LIMIT 10"}}}}
</tool>

User:
<result>
[Information about differences between quantum and classical computing would appear here]
</result>

Assistant:
<reasoning>
I now have sufficient information to answer the question. Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously, unlike classical bits. This gives quantum computers advantages for certain types of problems.
</reasoning>

<answer>
Quantum computing is a type of computing that uses quantum bits (qubits) instead of classical bits. While classical bits can only be in states of 0 or 1, qubits can exist in multiple states simultaneously due to quantum superposition, allowing quantum computers to solve certain problems much faster than classical computers.
</answer>

```
# AVAILABLE TOOLS
You have access to the following tools:

{tool_descriptions}

# REMEMBER
- You MUST use ALWAYS output <reasoning></reasoning> tags every response showing your thought process/analysis
- You MUST use <tool></tool> tags for tool calls
- You MUST use <answer></answer> tags ONLY for your final response
- The conversation will end if </answer> tag is found
- ALWAYS look carefully at the AVAILABLE TOOLS and DB SCHEMA before making tool calls
"""

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
embed_model.max_seq_length = 4096

xml_parser = XMLParser(fields=["reasoning", ("tool", "answer")])
env_parser = XMLParser(fields=["result"])


def get_last_answer(trajectory) -> str | None:
    """Extract the last answer from a trajectory."""
    for msg in reversed(trajectory):
        if msg['role'] == 'assistant':
            return msg['content']
    return None

def correctness_embedding_reward_func(completions, **kwargs) -> list[float]:
    graded_responses = []
    for completion, ground_truth in zip(completions, kwargs['answer']):
        tool_attempts = 0
        for i, msg in enumerate(completion):
            if msg['role'] == 'assistant':
                # Use parser to check for tool tag
                parsed = xml_parser.parse(msg['content'])
                if hasattr(parsed, 'tool') and parsed.tool is not None:
                    tool_attempts += 1

        tool_multiplier = 1.0
        if tool_attempts == 0:
            tool_multiplier = 0.1
        elif tool_attempts <= 2:
            tool_multiplier = 0.5
            

        last_answer = get_last_answer(completion)

        parsed = xml_parser.parse(last_answer)
        if hasattr(parsed, 'answer') and parsed.answer is not None:
            last_answer = parsed.answer
        
            ground_truth_embedding, last_answer_embedding = embed_model.encode([ground_truth, last_answer], task="retrieval.passage")

            similarity = np.dot(ground_truth_embedding, last_answer_embedding) / (np.linalg.norm(ground_truth_embedding) * np.linalg.norm(last_answer_embedding))
            if similarity > 0.92:
                score = 1.0
            elif similarity > 0.82:
                score = 0.5
            else:
                score = 0.0
        else:
            score = -0.3
        graded_responses.append(score * tool_multiplier)
    
    return graded_responses

def tool_execution_reward_func(completions, **kwargs):
    """
    Reward function that checks tool execution success.

    Uses XMLParser to identify proper tool calls.
    """
    def check_execution(trajectory):
        tool_attempts = 0
        successful_executions = 0
        
        # Find assistant messages with tools and their responses
        for i, msg in enumerate(trajectory):
            if msg['role'] == 'assistant':
                # Use parser to check for tool tag
                parsed = xml_parser.parse(msg['content'])
                if hasattr(parsed, 'tool') and parsed.tool is not None:
                    # Found a properly formatted tool message
                    if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                        tool_attempts += 1
                        # Check response with env_parser
                        multiplier = 1.0 
                        response = str(parsed.tool)
                        if (("vector_search_from_kb" in response) or ("query_db" in response)) and len(response) > 50:
                            multiplier = 1.5
                        else:
                            multiplier = 0.5
                        parsed_response = env_parser.parse(trajectory[i + 1]['content'])
                        if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                            successful_executions += 1 * multiplier
        
        # Calculate reward
        if tool_attempts == 0:
            return -0.2
        return (successful_executions / tool_attempts)
    
    return [check_execution(c) for c in completions]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""

    def check_execution(trajectory):
        reasoning_attempts = 0
        reason_and_tool_attempts = 0
        reason_and_answer_attempts = 0

        # Find assistant messages with tools and their responses
        num_assistant_messages = 0
        for i, msg in enumerate(trajectory):
            if msg['role'] == 'assistant':
                num_assistant_messages += 1
                # Use parser to check for tool tag
                parsed = xml_parser.parse(msg['content'])
                
                reason_formatting = hasattr(parsed, 'reasoning') and parsed.reasoning is not None
                if reason_formatting:
                    reasoning_attempts += 1
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        reason_and_tool_attempts += 1
                    elif hasattr(parsed, 'answer') and parsed.answer is not None and i + 1 == len(trajectory):
                        reason_and_answer_attempts += 1
        
        # Calculate reward
        if num_assistant_messages == 0:
            return -0.2
        score = (
            0.4 * (reasoning_attempts/num_assistant_messages) +
            0.3 * (reason_and_tool_attempts/max(1, num_assistant_messages - 1)) +
            0.3 * reason_and_answer_attempts
        )
        if score == 0.0:
            return -0.2
        return score
    
    scores = [check_execution(c) for c in completions]
    return scores

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    final_response_pattern = r"^<reasoning>\n[\s\S]*?\n</reasoning>\n\n<answer>\n[\s\S]*?\n</answer>$"

    responses = [get_last_answer(completion) for completion in completions]

    matches = [re.match(final_response_pattern, r) for r in responses]
    return [1.0 if match else -1.0 for match in matches]


dataset = load_dataset("Sulav/agent-rag-grpo-qa-2nd-trial")["train"]
# dataset = dataset.rename_column("question", "prompt")
dataset = dataset.train_test_split(test_size=0.12)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load tools

rag_tools = RAGTools(
    db_conn_string=f"host=citation-rag-postgres-do-user-12298230-0.f.db.ondigitalocean.com user=doadmin password={DB_PASSWORD} dbname=defaultdb port=25060",
    model_name="jinaai/jina-embeddings-v3",
    max_seq_length=4096
)

vf_env = vf.ToolEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt=AGENT_THINKING_SYSTEM_PROMPT,
    few_shot=[],
    tools=[rag_tools.vector_search_from_kb, rag_tools.query_db],
    max_steps=8
)
print(vf_env.system_prompt)

model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    num_train_epochs=1,
    top_p=0.95,
    repetition_penalty=1.07,
    temperature=0.7,
    max_steps=1000,
    bf16=True,
    max_grad_norm=1.0,
    num_iterations=4,
    beta=0.1,
    max_prompt_length=2048,
    max_completion_length=768,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_generations=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=100,
    eval_accumulation_steps=1,
    eval_on_start=False,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    vllm_max_model_len=16384,
    vllm_server_host="0.0.0.0", # replace with your inference server's host for multi-node setups
    vllm_server_port=8000,
    vllm_gpu_memory_utilization=0.98,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
    reward_weights=[1.0, 0.45, 0.25, 0.25],
    scale_rewards=False
)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_embedding_reward_func, tool_execution_reward_func, strict_format_reward_func, soft_format_reward_func],
    env=vf_env,
    args=training_args,
    train_dataset=vf_env.get_dataset(),
    eval_dataset=vf_env.get_eval_dataset()
)
trainer.train() 