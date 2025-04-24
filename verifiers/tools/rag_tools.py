from typing import List, Dict, Any
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from datetime import datetime
import re

class RAGTools:
    """
    Class for Retrieval Augmented Generation (RAG) database tools.
    Provides utilities for vector search and general database querying.
    """
    
    def __init__(self, db_conn_string: str, model_name: str, max_seq_length: int = 8192):
        """
        Initialize the RAG tools with database connection and embedding model.
        
        Args:
            db_conn_string: Database connection string
            model_name: Name of the sentence_transformers model to use
            max_seq_length: Maximum sequence length for embedding model
        """
        # Initialize database connection
        self.db_conn = psycopg.connect(db_conn_string)
        register_vector(self.db_conn)
        self.kb_cursor = self.db_conn.cursor(row_factory=dict_row)
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = max_seq_length
    
    def vector_search_from_kb(
        self,
        table_name: str,
        columns_to_select: List[str],
        embedding_column: str,
        query: str,
        top_k: int = 5,
        min_similarity_pct: float = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context using embeddings based cosine similarity search from the knowledge base.
        
        Args:
            table_name: Name of the table to search in
            columns_to_select: List of columns to select from the knowledge base
            embedding_column: Column name of the embedding column
            query: Query string
            top_k: Number of top results to return
            min_similarity_pct: Minimum similarity percentage
        Returns:
            List of dictionaries containing the query results
        """
        if not isinstance(columns_to_select, list):
            return "Error: columns_to_select must be a list"

        # Get allowed tables dynamically from the database
        self.kb_cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        table_search = self.kb_cursor.fetchall()
        allowed_tables = [row['table_name'] for row in table_search]
        if table_name not in allowed_tables:
            return f"Error: Table '{table_name}' is not allowed"

        # Get allowed columns dynamically from the database
        self.kb_cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
        allowed_columns = [row['column_name'] for row in self.kb_cursor.fetchall()]
        for col in columns_to_select + [embedding_column]:
            if col not in allowed_columns:
                return f"Error: Column '{col}' is not allowed"
        
        query_embedding = self.model.encode(query, task="retrieval.query", show_progress_bar=False).tolist()
        
        # Construct column selection safely
        columns_str = ", ".join(f'"{col}"' for col in columns_to_select)
        
        # Cast embedding to the proper type for comparison
        sql_query = f'''SELECT {columns_str}, (1 - ("{embedding_column}" <=> %s::vector)) * 100 AS similarity_percent
                FROM {table_name}
                WHERE (1 - ("{embedding_column}" <=> %s::vector)) * 100 > %s
                ORDER BY similarity_percent DESC
                LIMIT %s'''
        
        try:
            vector_results = self.kb_cursor.execute(
                sql_query, 
                (query_embedding, query_embedding, min_similarity_pct, top_k)
            ).fetchall()
            self.db_conn.commit()
        except Exception as e:
            self.db_conn.rollback()
            return f"Error: query execution failed: {str(e)}"
        
        return vector_results
    
    def query_db(self, sql_query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query on the database and return the results.
        Automatically converts datetime objects to ISO format strings for JSON serialization.
        
        Args:
            sql_query: SQL query to execute
            params: Parameters for the SQL query
            
        Returns:
            List of dictionaries containing the query results with datetime objects converted to strings
        """
        
        # First, check for dangerous operations
        if "CREATE TABLE" in sql_query.upper():
            return "Error: CREATE TABLE is not allowed, only SELECT queries are allowed"
        
        # Parse the query to determine if it's an aggregate query
        cleaned_query = sql_query.upper().strip()
        
        # Check if a LIMIT is required
        requires_limit = True
        
        # Aggregate functions naturally limit results
        aggregate_patterns = [
            r"SELECT\s+COUNT\s*\(",
            r"SELECT\s+SUM\s*\(",
            r"SELECT\s+AVG\s*\(",
            r"SELECT\s+MIN\s*\(",
            r"SELECT\s+MAX\s*\("
        ]
        
        # Check if it contains any aggregate function patterns
        for pattern in aggregate_patterns:
            if re.search(pattern, cleaned_query):
                requires_limit = False
                break
        
        # Check for GROUP BY with aggregate functions (still limited result set)
        if "GROUP BY" in cleaned_query and re.search(r"(COUNT|SUM|AVG|MIN|MAX)\s*\(", cleaned_query):
            requires_limit = False
        
        # Check for queries that select a single row by primary key
        if re.search(r"WHERE\s+\w+\s*=\s*['\"]?\w+['\"]?\s+(AND|OR|$|\s*LIMIT)", cleaned_query):
            requires_limit = False
        
        # If a LIMIT is already present, we don't need to enforce it
        if re.search(r"LIMIT\s+\d+", cleaned_query):
            requires_limit = False
        
        # Enforce LIMIT if needed
        if requires_limit:
            return "Error: LIMIT is required for non-aggregate queries. Please add a LIMIT clause."
        
        try:
            if params:
                results = self.kb_cursor.execute(sql_query, params).fetchall()
            else:
                results = self.kb_cursor.execute(sql_query).fetchall()
            self.db_conn.commit()

            # Convert datetime objects to ISO format strings
            serializable_results = []
            for row in results:
                serializable_row = {}
                for key, value in row.items():
                    if isinstance(value, datetime):
                        serializable_row[key] = value.isoformat()
                    else:
                        serializable_row[key] = value
                serializable_results.append(serializable_row)
            
            return serializable_results
        except Exception as e:
            self.db_conn.rollback()
            return f"Error: query execution failed: {str(e)}"
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()